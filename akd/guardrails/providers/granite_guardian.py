"""Granite Guardian Tool implementing GuardrailProtocol."""

from __future__ import annotations

import json
import os
import re
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin

import requests
from loguru import logger
from pydantic import Field, HttpUrl, model_validator
from typing_extensions import Self

from akd.guardrails._base import GuardrailInput, GuardrailOutput
from akd.guardrails.categories.granite import GraniteHarmCategory, GraniteRiskCategory
from akd.tools._base import BaseTool, BaseToolConfig


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


class GraniteGuardianToolConfig(BaseToolConfig):
    """Configuration for Granite Guardian Tool."""

    ollama_base_url: HttpUrl = Field(
        default=HttpUrl(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
    )
    model: GuardianModelID = Field(
        default=GuardianModelID.GUARDIAN_8B,
        description="Granite Guardian model to use.",
    )
    ollama_type: OllamaType = Field(
        default=OllamaType.SERVER,
        description="Ollama connection type.",
    )
    default_risk_category: GraniteRiskCategory = Field(
        default=GraniteRiskCategory.HARM,
        description="Default risk category to check.",
    )


class GraniteGuardianTool(BaseTool[GuardrailInput, GuardrailOutput]):
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

    async def _arun(self, params: GuardrailInput, **kwargs) -> GuardrailOutput:
        """Run single-risk detection."""
        # Determine which risk category to check
        risk_category = params.risk_categories[0] if params.risk_categories else self.config.default_risk_category

        # Convert to GraniteRiskCategory if string
        if isinstance(risk_category, str):
            risk_category = GraniteRiskCategory(risk_category)

        # Build messages for Ollama
        if params.context:
            # Multi-turn: context is user query, content is assistant response
            messages = [
                {"role": "system", "content": risk_category.value},
                {"role": "user", "content": params.context},
                {"role": "assistant", "content": params.content},
            ]
        else:
            # Single-turn: just content
            messages = [
                {"role": "system", "content": risk_category.value},
                {"role": "user", "content": params.content},
            ]

        # Call guardian model
        result = self._call_guardian(messages)

        # Build output
        detected_risks: list[GraniteRiskCategory] = []
        if result.get("is_risky"):
            detected_risks.append(risk_category)

        return GuardrailOutput(
            detected_risks=detected_risks,
            extra={
                "raw_response": result.get("raw_response"),
                "risk_label": result.get("risk_label"),
            },
        )

    def _call_guardian(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call Ollama with guardian model."""
        try:
            if self.config.ollama_type == OllamaType.CHAT:
                from ollama import chat

                result = chat(model=self.config.model.value, messages=messages)
                content = result.message.content
            else:
                result = self._ollama_server_chat(messages)
                content = result.get("content", "")

            # Parse yes/no from response
            match = re.search(r"\b(yes|no)\b", content, flags=re.IGNORECASE)
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

    def _ollama_server_chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call Ollama server /api/chat endpoint."""
        url = urljoin(str(self.config.ollama_base_url), "/api/chat")
        r = requests.post(
            url,
            json={
                "model": self.config.model.value,
                "messages": messages,
                "stream": True,
                "options": {"num_ctx": 8192, "temperature": 0, "seed": 42},
            },
            stream=False,
        )
        r.raise_for_status()

        output = ""
        for line in r.iter_lines():
            body = json.loads(line)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done") is False:
                message = body.get("message", {})
                output += message.get("content", "")
            if body.get("done", False):
                return {"content": output}

        return {"content": output}

    # GuardrailProtocol implementation
    def check(self, input: GuardrailInput) -> GuardrailOutput:
        """Sync guardrail check."""
        return self.run(input)

    async def acheck(self, input: GuardrailInput) -> GuardrailOutput:
        """Async guardrail check."""
        return await self.arun(input)


class MultiHarmGraniteGuardianToolConfig(GraniteGuardianToolConfig):
    """Configuration for Multi-Harm Granite Guardian Tool."""

    model: GuardianModelID = Field(
        default=GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM,
        description="Multi-harm model to use.",
    )

    @model_validator(mode="after")
    def validate_multi_harm_model(self) -> Self:
        """Ensure model is multi-harm capable."""
        if self.model != GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM:
            logger.warning(
                f"[MultiHarmGraniteGuardianTool] Model {self.model} may not support multi-harm detection. "
                f"Using {GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM} is recommended.",
            )
        return self


class MultiHarmGraniteGuardianTool(BaseTool[GuardrailInput, GuardrailOutput]):
    """
    Multi-harm Granite Guardian tool.

    Uses granite-guardian-3.2-5b-multi-harm model to detect ALL harm categories at once.
    Returns multiple detected risks in a single call.
    """

    name = "multi_harm_granite_guardian_tool"
    description = "Evaluates content for multiple harm categories using multi-harm model."
    input_schema = GuardrailInput
    output_schema = GuardrailOutput
    config_schema = MultiHarmGraniteGuardianToolConfig

    async def _arun(self, params: GuardrailInput, **kwargs) -> GuardrailOutput:
        """Run multi-harm detection."""
        # Build prompt based on single/multi-turn
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

        # Call multi-harm model
        result = self._call_multi_harm(prompt)

        return GuardrailOutput(
            detected_risks=result.get("categories", []),
            extra={
                "raw_response": result.get("raw_response"),
                "risk_label": result.get("risk_label"),
            },
        )

    def _call_multi_harm(self, prompt: str) -> dict[str, Any]:
        """Call multi-harm model with completion endpoint."""
        try:
            content = self._ollama_server_completion(prompt)
            categories = self._parse_categories(content)

            # Filter out non-harmful markers
            harmful_categories = [
                cat
                for cat in categories
                if cat not in (GraniteHarmCategory.NOT_HARMFUL_PROMPT, GraniteHarmCategory.NOT_HARMFUL_RESPONSE)
            ]

            is_risky = len(harmful_categories) > 0

            return {
                "risk_label": "yes" if is_risky else "no",
                "is_risky": is_risky,
                "categories": harmful_categories,
                "raw_response": content,
            }
        except Exception as e:
            logger.error(f"[MultiHarmGraniteGuardianTool] Error: {e}")
            return {"error": str(e), "categories": []}

    def _ollama_server_completion(self, prompt: str) -> str:
        """Call Ollama server /api/generate endpoint."""
        url = urljoin(str(self.config.ollama_base_url), "/api/generate")
        r = requests.post(
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
        r.raise_for_status()
        return r.json().get("response", "")

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

    # GuardrailProtocol implementation
    def check(self, input: GuardrailInput) -> GuardrailOutput:
        """Sync guardrail check."""
        return self.run(input)

    async def acheck(self, input: GuardrailInput) -> GuardrailOutput:
        """Async guardrail check."""
        return await self.arun(input)
