from pathlib import Path
from typing import Dict, List, Optional, Self

import yaml
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel, ConfigDict, Field, model_validator

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.agents._base import BaseAgentConfig
from akd.configs.prompts import RISK_SYSTEM_PROMPT


class RiskAgentInputSchema(InputSchema):
    """
    Input schema for the Risk Agent.
    """

    inputs: List[str] = Field(
        ...,
        description="A list of user inputs/messages, ordered chronologically as part of a conversation.",
    )
    outputs: List[str] = Field(
        ...,
        description="A list of model outputs/responses, aligned with the `inputs` by index.",
    )
    risk_ids: List[str] = Field(
        ...,
        description="A list of risk IDs to evaluate against. These should match keys in the risk definitions YAML file.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata such as model name, temperature, or any other context that may inform risk assessment.",
    )

    @model_validator(mode="after")
    def check_inputs(self) -> Self:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"'inputs' and 'outputs' must be of equal length. Got {len(self.inputs)} inputs and {len(self.outputs)} outputs.",
            )
        return self


class Criterion(BaseModel):
    """
    Single evaluation criterion
    """

    description: str = Field(
        ...,
        description="A specific, verifiable evaluation criterion.",
    )


class RiskAgentOutputSchema(InputSchema):
    """
    Output schema for Risk Agent.
    """

    criteria_by_risk: Dict[str, List[Criterion]] = Field(
        ...,
        description="A mapping of risk IDs to sructured evaluation criteria.",
    )
    dag_metric: DAGMetric = Field(
        ...,
        description="A DeepEval DAG metric constructed from the risk criteria.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RiskCriteriaOutputSchema(OutputSchema):
    """
    Schema used for instructor model (per risk as opposed to that of full output schema)
    """

    criteria: List[Criterion] = Field(
        ...,
        description="Criteria for a single risk",
    )


class RiskAgentConfig(BaseAgentConfig):
    """Configuration for the RiskAgent."""

    system_prompt: str = RISK_SYSTEM_PROMPT
    default_risk_yaml_path: str = str(Path(__file__).parent / "risk_atlas_data.yaml")


class RiskAgent(InstructorBaseAgent[RiskAgentInputSchema, RiskAgentOutputSchema]):
    """
    Agent that generates tailored risk evaluation criteria and a DAGMetric
    based on a predefined risk atlas and specific model inputs and outputs.

    Unlike approaches that convert static risk definitions into fixed criteria,
    this agent produces criteria that are context-sensitive - grounded not only
    in the risk definitions (from the YAML atlas) but also in the actual
    conversation (inputs/outputs) being evaluated.

    This allows the resulting DAGMetric to reflect how a given risk might
    manifest in a specific interaction, and enables generation of precise,
    relevant LLMTestCases for downsteam evaluation.

    The DAGMetric is structured hierarchically: each set of criteria derived
    for a particular risk is grouped under an aggregation node specific to that
    risk. These risk-specific aggregation nodes then feed into a final aggregation
    node that combines the results. If any of the risk criteria are violated,
    the final aggregation node reflects that in its verdict, effectively penalizing
    the overall score if risks are detected.
    """

    input_schema = RiskAgentInputSchema
    output_schema = RiskAgentOutputSchema
    config_schema = RiskAgentConfig

    _risk_map: Optional[Dict[str, str]] = None  # Cached risk ID -> description mapping

    def __init__(
        self,
        config: RiskAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the RiskAgent with configuration."""
        config = config or RiskAgentConfig()
        self.config = config
        super().__init__(config=config, debug=debug)

    @classmethod
    def load_risks_from_yaml(cls, path: str) -> Dict[str, str]:
        """
        Load risks from YAML file and return a dict of {risk_id: description}
        """
        if cls._risk_map is not None:
            return cls._risk_map

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cls._risk_map = {
            risk["id"]: risk["description"]
            for risk in data.get("risks", [])
            if "id" in risk and "description" in risk
        }
        return cls._risk_map

    def build_dag_from_criteria(
        self,
        criteria_by_risk: dict[str, list[Criterion]],
    ) -> DAGMetric:
        root_nodes = []
        final_risk_nodes = []
        num_risks = len(criteria_by_risk)

        for risk_id, criteria in criteria_by_risk.items():
            child_nodes = []

            for i, criterion in enumerate(criteria):
                node = TaskNode(
                    output_label=f"{risk_id}_{i + 1}",
                    instructions=f"{criterion.description} (True/False)",
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    children=[],  # Will link later
                    label=f"{risk_id}: {criterion.description}",
                )
                child_nodes.append(node)
                root_nodes.append(node)

            # Combine all individual criterion nodes under a risk node
            risk_agg_node = TaskNode(
                output_label=f"{risk_id}_all_true",
                instructions=f"Are all of the following true: {', '.join(n.output_label for n in child_nodes)}?",
                evaluation_params=[],
                children=[],  # Will link to global node later
                label=f"{risk_id} aggregation node",
            )
            for node in child_nodes:
                node.children = [risk_agg_node]

            final_risk_nodes.append(risk_agg_node)

        # Final aggregator node
        all_risk_outputs = [n.output_label for n in final_risk_nodes]

        verdicts = []
        for num_true in range(num_risks + 1):
            score = round((10 / num_risks) * num_true, 2) if num_risks > 0 else 0
            label = (
                f"All {num_risks} are True"
                if num_true == num_risks
                else f"None of {num_risks} are true"
                if num_true == 0
                else f"Only {num_true} of {num_risks} are True"
            )
            verdicts.append(VerdictNode(verdict=label, score=score))

        risk_summary_node = NonBinaryJudgementNode(
            criteria=f"Are values of {', '.join(all_risk_outputs)} all True?",
            children=verdicts,
            label="Final risk aggregation node.",
        )

        for risk_node in final_risk_nodes:
            risk_node.children = [risk_summary_node]

        dag_metric = DAGMetric(
            name=f"Evaluate result based on risks: {', '.join(criteria_by_risk.keys())}",
            dag=DeepAcyclicGraph(root_nodes=root_nodes),
            verbose_mode=True,
        )

        return dag_metric

    async def _arun(
        self,
        params: RiskAgentInputSchema,
        **kwargs,
    ) -> RiskAgentOutputSchema:
        # Load risk definitions
        risk_map = self.load_risks_from_yaml(self.config.default_risk_yaml_path)

        # Validate risk_ids
        unknown_ids = [r for r in params.risk_ids if r not in risk_map]
        if unknown_ids:
            raise ValueError(f"Unknown risk IDs provided: {unknown_ids}")

        criteria_by_risk = {}

        for risk_id in params.risk_ids:
            self.memory.clear()

            # Combine risk definition and conversation into one user message
            risk_description = risk_map[risk_id]

            conversation_text = "\n".join(
                f"Turn {i + 1}:\nUser: {inp}\nModel: {outp}"
                for i, (inp, outp) in enumerate(zip(params.inputs, params.outputs))
            )

            user_prompt = f"""\
            Risk ID: {risk_id}
            Risk Description: {risk_description}

            Conversation:
            {conversation_text}
            """

            self.memory.append(
                {
                    "role": "user",
                    "content": user_prompt,
                },
            )

            # Use the per-risk schema here instead of the full Output Schema for the Risk Agent
            response = await self.get_response_async(
                response_model=RiskCriteriaOutputSchema,
            )
            criteria_by_risk[risk_id] = response.criteria

        dag_metric = self.build_dag_from_criteria(criteria_by_risk)

        return RiskAgentOutputSchema(
            criteria_by_risk=criteria_by_risk,
            dag_metric=dag_metric,
        )
