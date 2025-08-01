from typing import List

from pydantic import Field

from akd._base import BaseConfig
from akd.tools.granite_guardian_tool import GuardianModelID, OllamaType, RiskDefinition


class GuardrailsConfig(BaseConfig):
    """Configuration for Granite Guardian guardrails validation for agents and tools."""

    enabled: bool = Field(
        default=True, description="Whether Granite Guardian validation is enabled"
    )

    input_risk_types: List[RiskDefinition] = Field(
        default=[
            RiskDefinition.JAILBREAK,
            RiskDefinition.HARM,
            RiskDefinition.UNETHICAL_BEHAVIOR,
        ],
        description="Risk types to check on user inputs",
    )

    output_risk_types: List[RiskDefinition] = Field(
        default=[RiskDefinition.ANSWER_RELEVANCE, RiskDefinition.GROUNDEDNESS],
        description="Risk types to check on agent responses",
    )

    guardian_model: GuardianModelID = Field(
        default=GuardianModelID.GUARDIAN_8B, description="Granite Guardian model to use"
    )

    fail_on_risk: bool = Field(
        default=False,
        description="Whether to raise exception on risk detection (True) or just log warning (False)",
    )

    ollama_type: OllamaType = Field(
        default=OllamaType.CHAT, description="Ollama interface type to use"
    )

    snippet_n_chars: int = Field(
        default=200, description="Number of characters to include in log snippets"
    )
