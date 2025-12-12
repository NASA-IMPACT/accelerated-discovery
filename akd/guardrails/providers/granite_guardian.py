"""Granite Guardian Tool implementing GuardrailProtocol."""

from __future__ import annotations

import asyncio
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


# Prompt templates for multi-harm detection
HARM_RISK_DEFINITION = "The 'User' message contains content that is harmful, unethical, or dangerous."

MULTI_HARM_PROMPT_TEMPLATE = """You are a safety agent trying to help determine whether the last user message in the conversation is harmful according to the safety risk definition below.

<start_of_conversation>
User Message: {user_message}
<end_of_conversation>

Our safety risk definition is defined below:
<start_of_risk_definition>
* {risk_definition}
<end_of_risk_definition>

Is the last user message harmful based on the risk definition? Your answer must be either 'Yes' or 'No'.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Yes
<confidence> High </confidence><|end_of_text|>
<categories>"""

MULTI_HARM_RESPONSE_TEMPLATE = """You are a safety agent trying to help determine whether the last assistant message in the conversation is harmful according to the safety risk definition below.

<start_of_conversation>
User Message: {user_message}
Assistant Message: {assistant_message}
<end_of_conversation>

Our safety risk definition is defined below:
<start_of_risk_definition>
* {risk_definition}
<end_of_risk_definition>

Is the last assistant message harmful based on the risk definition? Your answer must be either 'Yes' or 'No'.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Yes
<confidence> High </confidence><|end_of_text|>
<categories>"""


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


class GraniteGuardianTool(RiskCategoryValidationMixin, BaseTool[GuardrailInput, GuardrailOutput]):
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
                logger.debug(
                    f"[{self.__class__.__name__}] Risk {risk_category.value}: is_risky={result.get('is_risky')}",
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
                    "options": {"num_ctx": 8192, "temperature": 0, "seed": 42},
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("message", {}).get("content", "")

            # Parse yes/no from response
            match = re.search(r"\b(yes|no)\b", content or "", flags=re.IGNORECASE)
            if not match:
                return {"error": "Could not parse yes/no from response", "raw_response": content}

            label = match.group(1).lower()
            is_risky = label == "yes"

            return {
                "risk_label": label,
                "is_risky": is_risky,
                "raw_response": content,
            }
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

    @model_validator(mode="after")
    def validate_multi_risk_model(self) -> Self:
        """Ensure model is multi-risk capable."""
        if self.model != GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM:
            logger.warning(
                f"[MultiRiskGraniteGuardianTool] Model {self.model} may not support multi-risk detection. "
                f"Using {GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM} is recommended.",
            )
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
        """Run multi-harm detection (single call detects all harms)."""
        # Input overrides config - validate early before model call
        categories_to_check = list(params.risk_categories or self.config.risk_categories)
        self._validate_category_types(categories_to_check)

        if params.context:
            prompt = MULTI_HARM_RESPONSE_TEMPLATE.format(
                user_message=params.context,
                assistant_message=params.content,
                risk_definition=HARM_RISK_DEFINITION,
            )
        else:
            prompt = MULTI_HARM_PROMPT_TEMPLATE.format(
                user_message=params.content,
                risk_definition=HARM_RISK_DEFINITION,
            )

        result = await self._call_multi_harm_cached(prompt)

        # Filter to only include configured categories
        detected = [cat for cat in result.get("categories", []) if cat in categories_to_check]

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Detected {len(detected)} risks "
                f"(filtered from {len(result.get('categories', []))}): {[r.value for r in detected]}",
            )

        # Build per-risk results
        risk_results: dict[RiskCategory, dict[str, Any]] = {cat: {"is_risky": True} for cat in detected}

        return GuardrailOutput(
            detected_risks=detected,
            risk_results=risk_results,
            provider=self.__class__.__name__,
            extra={
                "raw_response": result.get("raw_response"),
                "risk_label": result.get("risk_label"),
                "unfiltered_categories": result.get("categories", []),
            },
        )

    @async_lru_cache(maxsize=256)
    async def _call_multi_harm_cached(self, prompt: str) -> dict[str, Any]:
        """Call multi-harm model with completion endpoint (cached)."""
        try:
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
                        "stop": ["</categories>"],
                    },
                },
            )
            response.raise_for_status()

            content = response.json().get("response", "")
            categories = self._parse_categories(content)

            # Filter out non-harmful markers
            harmful_categories = [
                cat
                for cat in categories
                if cat not in (GraniteHarmCategory.NOT_HARMFUL_PROMPT, GraniteHarmCategory.NOT_HARMFUL_RESPONSE)
            ]

            return {
                "risk_label": "yes" if harmful_categories else "no",
                "is_risky": len(harmful_categories) > 0,
                "categories": harmful_categories,
                "raw_response": content,
            }
        except Exception as e:
            logger.error(f"[MultiHarmGraniteGuardianTool] Error: {e}")
            return {"error": str(e), "categories": []}

    def _parse_categories(self, content: str) -> list[GraniteHarmCategory]:
        """Parse comma-separated categories from model output."""
        content = content.replace("</categories>", "").strip()
        if not content:
            return []

        raw_categories = [cat.strip() for cat in content.split(",") if cat.strip()]

        categories = []
        for raw_cat in raw_categories:
            try:
                categories.append(GraniteHarmCategory(raw_cat))
            except ValueError:
                logger.warning(f"[MultiHarmGraniteGuardianTool] Unknown category: {raw_cat}")

        return categories
