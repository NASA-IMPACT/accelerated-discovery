"""Granite Guardian Tool implementing GuardrailProtocol."""

from __future__ import annotations

import asyncio
import math
import os
import re
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import Field, HttpUrl, model_validator
from typing_extensions import Self

from akd.guardrails._base import (
    GuardrailInput,
    GuardrailOperatorMixin,
    GuardrailOutput,
    RiskCategoryValidationMixin,
)
from akd.guardrails.categories._base import RiskCategory
from akd.guardrails.categories.granite import GraniteHarmCategory, GraniteRiskCategory
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils import async_lru_cache


class GuardianModelID(StrEnum):
    """Granite Guardian model identifiers."""

    GUARDIAN_2B = "granite3-guardian:2b"
    GUARDIAN_8B = "granite3-guardian:8b"
    GUARDIAN_3_3_8B = "ibm/granite3.3-guardian:8b"
    GUARDIAN_3_2_5B_MULTI_HARM = "hf.co/nishparadox/granite-guardian-3.2-5b-multi-harm-GGUF"


class OllamaType(StrEnum):
    """Ollama connection type."""

    CHAT = "chat"
    SERVER = "server"


# Prompt templates for multi-harm detection (2-step inference)
# Uses Granite model special tokens for proper turn formatting
HARM_RISK_DEFINITION = "The 'User' message contains content that is harmful, unethical, or dangerous."

# Granite special tokens for chat formatting
_END_OF_TEXT = "<|end_of_text|>"
_ASSISTANT_ROLE = "<|start_of_role|>assistant<|end_of_role|>"

# Step 1 templates: Harm detection (model generates Yes/No + confidence)
# Ends with special tokens to trigger model response
HARM_DETECTION_USER_TEMPLATE = f"""You are a safety agent trying to help determine whether the last user message in the conversation is harmful according to the safety risk definition below.

<start_of_conversation>
User Message: {{user_message}}
<end_of_conversation>

Our safety risk definition is defined below:
<start_of_risk_definition>
* {{risk_definition}}
<end_of_risk_definition>

Is the last user message harmful based on the risk definition? Your answer must be either 'Yes' or 'No'.{_END_OF_TEXT}
{_ASSISTANT_ROLE}"""

HARM_DETECTION_ASSISTANT_TEMPLATE = f"""You are a safety agent trying to help determine whether the last assistant message in the conversation is harmful according to the safety risk definition below.

<start_of_conversation>
User Message: {{user_message}}
Assistant Message: {{assistant_message}}
<end_of_conversation>

Our safety risk definition is defined below:
<start_of_risk_definition>
* {{risk_definition}}
<end_of_risk_definition>

Is the last assistant message harmful based on the risk definition? Your answer must be either 'Yes' or 'No'.{_END_OF_TEXT}
{_ASSISTANT_ROLE}"""


class GraniteGuardianBaseConfig(BaseToolConfig):
    """Base configuration for Granite Guardian tools (shared fields)."""

    ollama_base_url: HttpUrl = Field(
        default=HttpUrl(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
    )
    ollama_type: OllamaType = Field(
        default=OllamaType.SERVER,
        description="Ollama connection type.",
    )
    max_concurrency: int = Field(
        default=3,
        description="Max concurrent requests to Ollama (prevents memory/compute overload).",
    )
    timeout: float = Field(
        default=60.0,
        description="HTTP request timeout in seconds.",
    )
    validate_categories: bool = Field(
        default=True,
        description="Validate that input categories match the tool's supported types.",
    )
    think: bool = Field(
        default=False,
        description="Enable chain-of-thought reasoning. Returns thinking process in risk_results[<RiskCategory>]['thinking'].",
    )


class GraniteGuardianToolConfig(GraniteGuardianBaseConfig):
    """Configuration for Granite Guardian Tool (single-risk mode)."""

    model: GuardianModelID = Field(
        default=GuardianModelID.GUARDIAN_8B,
        description="Granite Guardian model to use.",
    )
    risk_categories: list[GraniteRiskCategory] = Field(
        default_factory=lambda: list(GraniteRiskCategory),
        description="Risk categories to check (defaults to all).",
    )

    @model_validator(mode="after")
    def validate_think_support(self) -> Self:
        """Auto-disable think if model doesn't support it."""
        # Only granite3.3-guardian:8b supports thinking mode
        if self.think and self.model != GuardianModelID.GUARDIAN_3_3_8B:
            logger.warning(
                f"[GraniteGuardianToolConfig] Model {self.model} does not support think=True. "
                f"Only {GuardianModelID.GUARDIAN_3_3_8B} supports chain-of-thought reasoning. "
                f"Automatically disabling think parameter.",
            )
            self.think = False
        return self


class GraniteGuardianTool(
    GuardrailOperatorMixin,
    RiskCategoryValidationMixin,
    BaseTool[GuardrailInput, GuardrailOutput],
):
    """
    Granite Guardian tool for single-risk detection.

    Uses granite3-guardian 2B/8B models to check content for a specific risk category.
    Implements GuardrailProtocol for unified guardrail interface.
    """

    name = "granite_guardian_tool"
    description = "Evaluates content for risks using Granite Guardian model."
    input_schema = GuardrailInput
    output_schema = GuardrailOutput
    config_schema = GraniteGuardianToolConfig

    _client: httpx.AsyncClient | None = None
    _semaphore: asyncio.Semaphore | None = None

    def _post_init(self) -> None:
        super()._post_init()
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Initialized with model={self.config.model}, "
                f"max_concurrency={self.config.max_concurrency}, timeout={self.config.timeout}s",
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            if self.debug:
                logger.debug(f"[{self.__class__.__name__}] HTTP client closed")

    async def _arun(self, params: GuardrailInput, **kwargs) -> GuardrailOutput:
        """Run risk detection for each requested category in parallel (with concurrency limit)."""
        # Input overrides config
        categories_to_check = list(params.risk_categories or self.config.risk_categories)

        # Validate category types (if enabled)
        self._validate_category_types(categories_to_check)

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Checking {len(categories_to_check)} risk categories: "
                f"{[c.value for c in categories_to_check]}",
            )

        # Run all category checks in parallel (limited by semaphore)
        tasks = [self._check_single_risk(params, cat) for cat in categories_to_check]
        results = await asyncio.gather(*tasks)

        # Collect detected risks and per-risk results
        detected_risks: list[GraniteRiskCategory] = []
        risk_results: dict[RiskCategory, dict[str, Any]] = {}

        for cat, result in zip(categories_to_check, results):
            risk_results[cat] = result
            if result.get("is_risky"):
                detected_risks.append(cat)

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Detected {len(detected_risks)} risks: "
                f"{[r.value for r in detected_risks]}",
            )

        return GuardrailOutput(
            detected_risks=detected_risks,
            risk_results=risk_results,
            provider=self.__class__.__name__,
        )

    async def _check_single_risk(
        self,
        params: GuardrailInput,
        risk_category: GraniteRiskCategory,
    ) -> dict[str, Any]:
        """Check a single risk category (with semaphore for concurrency control)."""
        assert self._semaphore is not None, "Tool not initialized"
        async with self._semaphore:
            if self.debug:
                logger.debug(f"[{self.__class__.__name__}] Checking risk: {risk_category.value}")

            if params.context:
                messages = (
                    ("system", risk_category.value),
                    ("user", params.context),
                    ("assistant", params.content),
                )
            else:
                messages = (
                    ("system", risk_category.value),
                    ("user", params.content),
                )
            result = await self._call_guardian_cached(messages)

            if self.debug:
                thinking_preview = result.get("thinking", "")[:100] if result.get("thinking") else "N/A"
                logger.debug(
                    f"[{self.__class__.__name__}] Risk {risk_category.value}: "
                    f"is_risky={result.get('is_risky')}, thinking={thinking_preview}...",
                )
            return result

    @async_lru_cache(maxsize=256)
    async def _call_guardian_cached(
        self,
        messages: tuple[tuple[str, str], ...],
    ) -> dict[str, Any]:
        """Call Ollama with guardian model (cached)."""
        try:
            # Convert tuple back to list of dicts for API
            messages_list = [{"role": role, "content": content} for role, content in messages]

            url = urljoin(str(self.config.ollama_base_url), "/api/chat")
            response = await self._client.post(
                url,
                json={
                    "model": self.config.model.value,
                    "messages": messages_list,
                    "stream": False,
                    "think": self.config.think,
                    "options": {"num_ctx": 8192, "temperature": 0, "seed": 42},
                },
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            # Parse yes/no from response
            match = re.search(r"\b(yes|no)\b", content or "", flags=re.IGNORECASE)
            if not match:
                return {"error": "Could not parse yes/no from response", "raw_response": content}

            label = match.group(1).lower()
            is_risky = label == "yes"

            result = {
                "risk_label": label,
                "is_risky": is_risky,
                "raw_response": content,
            }

            # Include thinking if present
            if thinking:
                result["thinking"] = thinking

            return result
        except Exception as e:
            logger.error(f"[GraniteGuardianTool] Error: {e}")
            return {"error": str(e)}

    # GuardrailProtocol implementation
    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Sync guardrail check."""
        return self.run(params)

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Async guardrail check."""
        return await self.arun(params)


class MultiRiskGraniteGuardianToolConfig(GraniteGuardianBaseConfig):
    """Configuration for Multi-Risk Granite Guardian Tool."""

    model: GuardianModelID = Field(
        default=GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM,
        description="Multi-risk model to use.",
    )
    risk_categories: list[GraniteHarmCategory] = Field(
        default_factory=lambda: [
            c
            for c in GraniteHarmCategory
            if c not in (GraniteHarmCategory.NOT_HARMFUL_PROMPT, GraniteHarmCategory.NOT_HARMFUL_RESPONSE)
        ],
        description="Harm categories to report (filters model output, defaults to all harmful).",
    )
    score_threshold: float = Field(
        default=0.0,
        description=(
            "Minimum per-category score required to include a detected risk. "
            "Range 0.0-1.0. Default 0.0 = no filtering (all detected categories pass). "
            "Set (e.g., 0.5) to drop low-confidence detections and reduce false positives. "
            "Has no effect if Ollama does not return logprobs (scores are None)."
        ),
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_multi_risk_model(self) -> Self:
        """Ensure model is multi-risk capable."""
        if self.model != GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM:
            logger.warning(
                f"[MultiRiskGraniteGuardianTool] Model {self.model} may not support multi-risk detection. "
                f"Using {GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM} is recommended.",
            )
        return self

    @model_validator(mode="after")
    def validate_think_support(self) -> Self:
        """Auto-disable think as multi-risk models don't support it."""
        if self.think:
            logger.warning(
                "[MultiRiskGraniteGuardianToolConfig] Multi-risk models do not support think=True. "
                "Chain-of-thought reasoning is not available in multi-harm detection mode. "
                "Automatically disabling think parameter.",
            )
            self.think = False
        return self


class MultiRiskGraniteGuardianTool(GraniteGuardianTool):
    """
    Multi-risk Granite Guardian tool.

    Uses granite-guardian-3.2-5b-multi-harm model to detect ALL risk categories at once.
    Returns multiple detected risks in a single call.
    """

    name = "multi_risk_granite_guardian_tool"
    description = "Evaluates content for multiple risk categories using multi-risk model."
    config_schema = MultiRiskGraniteGuardianToolConfig
    config: MultiRiskGraniteGuardianToolConfig  # type hint for pyright

    async def _arun(self, params: GuardrailInput, **kwargs) -> GuardrailOutput:
        """Run multi-harm detection using 2-step inference per model card.

        Step 1: Detect if content is harmful (Yes/No + confidence)
        Step 2: If harmful, get specific harm categories
        """
        # Input overrides config - validate early before model call
        categories_to_check = list(params.risk_categories or self.config.risk_categories)
        self._validate_category_types(categories_to_check)

        # Build Step 1 prompt (harm detection - no hardcoded answer)
        if params.context:
            step1_prompt = HARM_DETECTION_ASSISTANT_TEMPLATE.format(
                user_message=params.context,
                assistant_message=params.content,
                risk_definition=HARM_RISK_DEFINITION,
            )
        else:
            step1_prompt = HARM_DETECTION_USER_TEMPLATE.format(
                user_message=params.content,
                risk_definition=HARM_RISK_DEFINITION,
            )

        # Step 1: Detect if harmful (Yes/No + confidence)
        step1_result = await self._call_harm_detection(step1_prompt)

        label = step1_result.get("label", "").lower()
        confidence = step1_result.get("confidence", "")
        is_harmful = label == "yes"

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Step 1 - is_harmful={is_harmful}, label={label}, confidence={confidence}",
            )

        if not is_harmful:
            # Content is NOT harmful - return early with empty risks
            return GuardrailOutput(
                detected_risks=[],
                risk_results={},
                provider=self.__class__.__name__,
                extra={
                    "risk_label": label or "no",
                    "confidence": confidence,
                    "step1_raw": step1_result.get("raw_response"),
                },
            )

        # Step 2: Get specific categories (only if Step 1 = "Yes")
        # Append model's Step 1 output + <|end_of_text|> + <categories> (matches original format)
        step2_prompt = step1_prompt + step1_result["raw_response"] + _END_OF_TEXT + "\n<categories>"
        step2_result = await self._call_category_detection(step2_prompt)

        category_scores: dict[GraniteHarmCategory, float | None] = step2_result.get("categories", {})
        threshold = self.config.score_threshold
        _non_harmful = (GraniteHarmCategory.NOT_HARMFUL_PROMPT, GraniteHarmCategory.NOT_HARMFUL_RESPONSE)

        # Filter: configured categories only, exclude non-harmful markers, drop categories
        # whose per-category score is below the threshold. Categories without a score
        # (e.g., when Ollama doesn't return logprobs) pass through unfiltered.
        detected = [
            cat
            for cat, score in category_scores.items()
            if cat in categories_to_check and cat not in _non_harmful and (score is None or score >= threshold)
        ]

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Step 2 - Detected {len(detected)} risks "
                f"(filtered from {len(category_scores)}, threshold={threshold}): "
                f"{[(r.value, category_scores.get(r)) for r in detected]}",
            )

        # Build per-risk results with per-category score.
        risk_results: dict[RiskCategory, dict[str, Any]] = {
            cat: {"is_risky": True, "score": category_scores.get(cat)} for cat in detected
        }

        return GuardrailOutput(
            detected_risks=detected,
            risk_results=risk_results,
            provider=self.__class__.__name__,
            extra={
                "risk_label": "yes",
                "confidence": confidence,
                "step1_raw": step1_result.get("raw_response"),
                "step2_raw": step2_result.get("raw_response"),
                "unfiltered_categories": category_scores,
            },
        )

    @async_lru_cache(maxsize=256)
    async def _call_harm_detection(self, prompt: str) -> dict[str, Any]:
        """Step 1: Call model to detect harm. Stop at </confidence>."""
        try:
            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 1 - Sending harm detection prompt:\n"
                    f"--- PROMPT START ---\n{prompt}\n--- PROMPT END ---",
                )

            url = urljoin(str(self.config.ollama_base_url), "/api/generate")
            response = await self._client.post(
                url,
                json={
                    "model": self.config.model.value,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0,
                        "seed": 42,
                        "stop": ["</confidence>"],
                    },
                },
            )
            response.raise_for_status()

            content = response.json().get("response", "")

            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 1 - Ollama response:\n"
                    f"--- RESPONSE START ---\n{content}\n--- RESPONSE END ---",
                )

            # Parse: "Yes\n<confidence> High " or "No\n<confidence> Not Harmful "
            label = "yes" if content.strip().lower().startswith("yes") else "no"
            confidence_match = re.search(r"<confidence>\s*(.+?)(?:</confidence>|\s*$)", content)
            confidence = confidence_match.group(1).strip() if confidence_match else ""

            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 1 - Parsed: label={label}, confidence={confidence}",
                )

            return {
                "label": label,
                "confidence": confidence,
                "raw_response": content + "</confidence>",  # Include closing tag for Step 2
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Step 1 error: {e}")
            return {"error": str(e), "label": "no", "confidence": "", "raw_response": ""}

    @async_lru_cache(maxsize=256)
    async def _call_category_detection(self, prompt: str) -> dict[str, Any]:
        """Step 2: Call model to get categories. Stop at </categories>."""
        try:
            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 2 - Sending category detection prompt:\n"
                    f"--- PROMPT START ---\n{prompt}\n--- PROMPT END ---",
                )

            url = urljoin(str(self.config.ollama_base_url), "/api/generate")
            response = await self._client.post(
                url,
                json={
                    "model": self.config.model.value,
                    "prompt": prompt,
                    "stream": False,
                    # logprobs let us derive per-category model confidence from the
                    # first token emitted for each category name.
                    "logprobs": True,
                    "top_logprobs": 5,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0,
                        "seed": 42,
                        "stop": ["</categories>"],
                    },
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "")
            token_logprobs = data.get("logprobs") or []

            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 2 - Ollama response:\n"
                    f"--- RESPONSE START ---\n{content}\n--- RESPONSE END ---",
                )

            categories = self._parse_categories(content, token_logprobs)

            if self.debug:
                logger.debug(
                    f"[{self.__class__.__name__}] Step 2 - Parsed categories with scores: "
                    f"{[(c.value, s) for c, s in categories.items()]}",
                )

            return {
                "categories": categories,
                "raw_response": content,
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Step 2 error: {e}")
            return {"error": str(e), "categories": {}}

    def _parse_categories(
        self,
        content: str,
        token_logprobs: list[dict[str, Any]] | None = None,
    ) -> dict[GraniteHarmCategory, float | None]:
        """Parse categories from model output into a ``{category: score}`` mapping.

        The score is ``exp(first_token_logprob)`` (model confidence in emitting that
        category's first token), or ``None`` when ``token_logprobs`` is not provided.
        Python dicts preserve insertion order, so the model's emission order is kept.
        """
        cleaned = content.replace("</categories>", "").strip()
        if not cleaned:
            return {}

        raw_categories = [c.strip() for c in cleaned.split(",") if c.strip()]

        # Walk the token stream and find, for each emitted category, the logprob of
        # the first non-whitespace, non-comma token that starts it.
        first_token_logprobs = self._first_token_logprob_per_category(token_logprobs or [])

        results: dict[GraniteHarmCategory, float | None] = {}
        for i, raw_cat in enumerate(raw_categories):
            try:
                cat = GraniteHarmCategory(raw_cat)
            except ValueError:
                logger.warning(f"[{self.__class__.__name__}] Unknown category: {raw_cat}")
                continue
            lp = first_token_logprobs[i] if i < len(first_token_logprobs) else None
            results[cat] = math.exp(lp) if lp is not None else None

        return results

    @staticmethod
    def _first_token_logprob_per_category(
        token_logprobs: list[dict[str, Any]],
    ) -> list[float | None]:
        """Extract the logprob of the first token of each comma-separated category.

        The Step 2 output looks like ``Violence, Unethical Behavior``. We skip leading
        whitespace/commas and grab the logprob of each category's first token. Returns a
        list of logprobs aligned with the comma-separated category order.
        """
        if not token_logprobs:
            return []

        first_lps: list[float | None] = []
        expecting_start = True  # at position 0 and after each comma
        for entry in token_logprobs:
            tok = entry.get("token", "")
            stripped = tok.strip()
            # A comma ends the current category; next non-whitespace token starts the next one.
            if "," in tok:
                expecting_start = True
                continue
            # Skip pure whitespace/newline tokens between categories.
            if not stripped:
                continue
            if expecting_start:
                lp = entry.get("logprob")
                first_lps.append(lp if isinstance(lp, (int, float)) else None)
                expecting_start = False

        return first_lps
