from typing import Dict, List, Optional

import yaml
from loguru import logger
from pydantic import Field, model_validator

from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent
from akd.agents._base import BaseAgentConfig
from akd.configs.prompts import RISK_REPORT_SYSTEM_PROMPT
from akd.utils import get_akd_root


class RiskReportAgentInputSchema(InputSchema):
    """
    Input schema for the Risk Report Agent.
    """

    failed_criteria: Dict[str, List[str]] = Field(
        default=None,
        description=(
            "Failed criteria per risk."
            "Keys are risk IDs for risks that failed a run of the Risk Agent."
            "These should match keys in the risk definitions YAML file or science risks yaml file."
        ),
    )
    risky_content: str = Field(
        ...,
        description="Risky gererated content, optionally including the query/queries used for generation.",
    )


class RiskReportAgentOutputSchema(OutputSchema):
    """
    Output schema for Risk Report Agent.
    """

    risk_report: str = Field(
        ...,
        description="A report documenting detected risks generated content.",
    )


class RiskReportAgentConfig(BaseAgentConfig):
    """Configuration for the RiskReportAgent."""

    system_prompt: str = RISK_REPORT_SYSTEM_PROMPT
    risk_yaml_paths: List[str] = Field(
        default_factory=lambda: [
            str(
                get_akd_root() / "akd/agents/risk/risk_atlas_data.yaml",
            ),
            str(
                get_akd_root() / "akd/agents/risk/science_lit_risks.yaml",
            ),
        ],
        description="List of yaml files defining risks. "
        "Each file must contain a `risks` key with `id` and `description` fields.",
    )
    agent_description: Optional[str] = Field(
        default=None,
        description="Description of agent being evaluated - used as behavioral context.",
    )

    @model_validator(mode="after")
    def _inject_agent_description(self) -> "RiskReportAgentConfig":
        """Dynamically enrich the system prompt if a description is provided."""
        if self.agent_description:
            self.system_prompt = (
                RISK_REPORT_SYSTEM_PROMPT + "\n\nAgent Behavioral Context:\n" + self.agent_description.strip() + "\n"
            )
        return self


class RiskReportAgent(
    LiteLLMInstructorBaseAgent[RiskReportAgentInputSchema, RiskReportAgentOutputSchema],
):
    """
    Agent that generates a report detailing detected risks in generated content
    """

    input_schema = RiskReportAgentInputSchema
    output_schema = RiskReportAgentOutputSchema
    config_schema = RiskReportAgentConfig

    _risk_map: Optional[Dict[str, str]] = None  # Cached risk ID -> description mapping

    def __init__(
        self,
        config: RiskReportAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the RiskReportAgent with configuration."""
        config = config or RiskReportAgentConfig()
        super().__init__(config=config, debug=debug)
        self._risk_map = self.load_risks_from_yaml(
            config.risk_yaml_paths,
        )
        logger.info("Risk report agent created.")

    @staticmethod
    def load_risks_from_yaml(yaml_paths: List[str]) -> Dict[str, str]:
        """
        Load risks from YAML file and return a dict of {risk_id: description}
        """

        risk_def_dicts = []
        for risk_def_path in yaml_paths:
            with open(risk_def_path, "r", encoding="utf-8") as f:
                risk_def_dicts.append(yaml.safe_load(f))

        merged_risks = {}
        for i, data in enumerate(risk_def_dicts):
            if "risks" not in data:
                logger.warning(f"`risks` key is missing from {risk_def_path[i]}.")
            for risk in data.get("risks", []):
                risk_id = risk.get("id")
                risk_description = risk.get("description")
                risk_concern = risk.get("concern")
                risk_isPartOf = risk.get("isPartOf")
                if risk_id and risk_description:
                    merged_risks[risk_id] = {
                        "description": risk_description,
                        "concern": risk_concern,
                        "isPartOf": risk_isPartOf,
                    }

        return merged_risks

    async def arun(
        self,
        params: RiskReportAgentInputSchema,
        **kwargs,
    ) -> RiskReportAgentOutputSchema:
        unknown_ids = [r for r in params.failed_criteria.keys() if r not in self._risk_map]
        if unknown_ids:
            logger.error(f"Unknown risk IDs provided in `failed_criteria`: {unknown_ids}")
            raise ValueError(f"Unknown risk IDs provided in `failed_criteria`: {unknown_ids}")

        relevant_risk_definitions = {risk_id: self._risk_map[risk_id] for risk_id in params.failed_criteria}
        object.__setattr__(
            params,
            "relevant_risk_definitions",
            relevant_risk_definitions,
        )

        response = await super().arun(params)

        return response
