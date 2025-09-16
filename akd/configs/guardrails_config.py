from typing import List

from pydantic import Field

from akd._base import BaseConfig
from akd.tools.granite_guardian_tool import GuardianModelID, OllamaType, RiskDefinition


class GuardrailsConfig(BaseConfig):
    """Configuration for Granite Guardian guardrails validation for agents and tools."""

    enabled: bool = Field(
        default=True,
        description="Whether Granite Guardian validation is enabled",
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
        default=GuardianModelID.GUARDIAN_8B,
        description="Granite Guardian model to use",
    )

    fail_on_risk: bool = Field(
        default=False,
        description="Whether to raise exception on risk detection (True) or just log warning (False)",
    )

    ollama_type: OllamaType = Field(
        default=OllamaType.SERVER,
        description="Ollama interface type to use",
    )

    snippet_n_chars: int = Field(
        default=200,
        description="Number of characters to include in log snippets",
    )

    input_fields: list[str] = Field(
        default_factory=lambda: [
            "query",
            "content",
            "text",
            "user_input",
            "message",
            "queries",
        ],
        description="List of input field names to check for risks. If empty, all input fields will be checked.",
    )

    output_fields: list[str] = Field(
        default_factory=lambda: [
            "response",
            "answer",
            "content",
            "text",
            "result",
            "results",
            "search_results",
        ],
        description="List of output field names to check for risks. If empty, all output fields will be checked.",
    )
