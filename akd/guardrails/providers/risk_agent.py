"""RiskAgent guardrail provider implementing GuardrailProtocol.

This module provides RiskAgent, which generates evaluation criteria for risks
and builds DAG metrics for hierarchical evaluation. It directly uses
GuardrailInput/GuardrailOutput for unified interface with other guardrail providers.
"""

import asyncio
from collections.abc import Sequence
from enum import Enum
from typing import Any

from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from loguru import logger
from pydantic import BaseModel, Field

from akd._base import OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent
from akd.agents._base import BaseAgentConfig
from akd.configs.prompts import RISK_SYSTEM_PROMPT
from akd.guardrails._base import (
    GuardrailInput,
    GuardrailOperatorMixin,
    GuardrailOutput,
    RiskCategoryValidationMixin,
)
from akd.guardrails.categories._base import RiskCategory

# Dynamically created from YAML - may be None if file doesn't exist
from akd.guardrails.categories.atlas import ScienceRiskCategory


class CriterionImportance(Enum):
    """Importance level for evaluation criteria."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Criterion(BaseModel):
    """Single evaluation criterion."""

    description: str = Field(
        ...,
        description="A specific, verifiable evaluation criterion.",
    )
    importance: CriterionImportance = Field(
        description="Importance of criterion",
        default=CriterionImportance.MEDIUM,
    )


class RiskCriteriaOutputSchema(OutputSchema):
    """Schema used for instructor model (per risk criteria generation)."""

    criteria: list[Criterion] = Field(
        ...,
        description="Criteria for a single risk",
    )


class RiskAgentConfig(BaseAgentConfig):
    """Configuration for the RiskAgent."""

    system_prompt: str = RISK_SYSTEM_PROMPT
    io_hints: bool = Field(
        default=False,
        description="Overriding this to suppress error in json schema conversion of DAG metric.",
    )
    pass_threshold: float = Field(
        default=0.9,
        description="Score threshold for passing (0.0-1.0).",
    )
    validate_categories: bool = Field(
        default=True,
        description="Validate that input categories match supported types.",
    )
    risk_categories: list[RiskCategory] = Field(
        default_factory=lambda: [
            ScienceRiskCategory.HALLUCINATION_IDENTIFICATION,
            ScienceRiskCategory.ATTRIBUTION,
            ScienceRiskCategory.CONSISTENCY,
            ScienceRiskCategory.UNCERTAINTY_IDENTIFICATION,
            ScienceRiskCategory.OVERGENERALIZATION,
        ]
        if ScienceRiskCategory
        else [],
        description="Default risk categories for scientific discovery (AtlasRiskCategory, ScienceRiskCategory).",
    )
    risk_weights: dict[str, float] | None = Field(
        default=None,
        description="Optional per-risk weight overrides. Keys are risk_id values.",
    )
    include_dag_metric: bool = Field(
        default=False,
        description="Include full DAGMetric object in output extra (for advanced inspection).",
    )


class RiskAgent(
    GuardrailOperatorMixin,
    RiskCategoryValidationMixin,
    LiteLLMInstructorBaseAgent[GuardrailInput, GuardrailOutput],
):
    """
    Agent that generates tailored risk evaluation criteria and a DAGMetric
    based on risk category metadata and specific model inputs/outputs.

    Uses preloaded RiskCategory enums (AtlasRiskCategory, ScienceRiskCategory, etc.)
    which already have metadata (description, name, extra) attached - no YAML loading needed.

    Implements GuardrailProtocol via check()/acheck() for unified interface.

    Usage:
        from akd.guardrails import GuardrailInput, AtlasRiskCategory
        from akd.guardrails.providers import RiskAgent

        agent = RiskAgent()
        output = await agent.acheck(GuardrailInput(
            content="The capital of France is Berlin.",
            context="What is the capital of France?",
            risk_categories=[AtlasRiskCategory.HALLUCINATION],
        ))
        print(output.passed)  # False
        print(output.detected_risks)  # [AtlasRiskCategory.HALLUCINATION]
    """

    input_schema = GuardrailInput
    output_schema = GuardrailOutput
    config_schema = RiskAgentConfig
    config: RiskAgentConfig  # For type checker

    def __init__(
        self,
        config: RiskAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the RiskAgent with configuration."""
        config = config or RiskAgentConfig()
        super().__init__(config=config, debug=debug)
        logger.info("RiskAgent created.")

    def _get_risk_agg_instructions(
        self,
        high_clause: str,
        medium_clause: str,
        m_required: int,
        borderline_expr: str,
        low_clause: str,
    ) -> str:
        return (
            "Evaluate the risk pass/fail using the following rules:\n"
            f"1) HIGH: {high_clause}\n"
            f"2) MEDIUM: {medium_clause}\n"
            f"   - Define MEDIUM_pass = True if count(True in MEDIUM) >= {m_required}; else False.\n"
            f"   - Define MEDIUM_borderline = {borderline_expr}\n"
            f"3) LOW (tiebreaker only): {low_clause}\n\n"
            "Decision logic:\n"
            "- If any HIGH is False -> return False.\n"
            "- Else if MEDIUM_pass is True -> return True.\n"
            "- Else if MEDIUM_borderline is True ->\n"
            "      If any LOW is True -> return True; else return False.\n"
            "- Else -> return False.\n\n"
            "Return strictly 'True' or 'False'."
        )

    def _get_summary_instructions(
        self,
        all_risk_outputs: list[str],
        weights_text: str,
        denom: float,
    ) -> str:
        return (
            "Compute a weighted pass ratio over risks using their pass/fail outputs and the provided weights.\n"
            "Steps:\n"
            f"1) Risk pass/fail outputs: [{', '.join(all_risk_outputs)}]\n"
            "   Treat each as True (passed) or False (failed).\n"
            "2) Weights:\n"
            f"{weights_text}\n"
            f"3) Let total_weight = {denom} (sum of the weights).\n"
            "4) Let passed_weight = sum of weights for risks that passed (True).\n"
            "5) weighted_ratio = passed_weight / total_weight.\n"
            "6) Select the verdict that matches the weighted_ratio bucket:\n"
            "- Choose the **highest threshold** that weighted_ratio meets.\n"
            "- Example: If weighted_ratio = 1.0, it meets >= 0.25, >= 0.50, >= 0.75, and >= 0.90, "
            "but you must select only the >= 0.90 verdict.\n"
            "7) Do not select lower thresholds once a higher one applies."
        )

    def _resolve_risk_weights(self, risk_categories: Sequence[RiskCategory]) -> dict[str, float]:
        """Resolve weights for all risk categories, defaulting to 1.0."""
        overrides = self.config.risk_weights or {}
        return {cat.value: float(overrides.get(cat.value, 1.0)) for cat in risk_categories}

    def _build_dag_from_criteria(
        self,
        criteria_by_risk: dict[str, list[Criterion]],
        risk_weights: dict[str, float] | None = None,
    ) -> tuple[DAGMetric, dict[str, list[TaskNode]]]:
        """Build a DAGMetric from criteria grouped by risk.

        Returns:
            Tuple of (DAGMetric, criterion_nodes_by_risk) where criterion_nodes_by_risk
            maps risk_id to list of TaskNodes in same order as criteria.
        """
        root_nodes: list[TaskNode] = []
        final_risk_nodes: list[TaskNode] = []
        criterion_nodes_by_risk: dict[str, list[TaskNode]] = {}

        for risk_id, criteria in criteria_by_risk.items():
            child_nodes = []
            criterion_nodes_by_risk[risk_id] = []

            # Create root nodes (binary True/False) with importance-aware labeling
            for i, criterion in enumerate(criteria):
                importance_str = getattr(
                    criterion,
                    "importance",
                    CriterionImportance.MEDIUM,
                ).value

                node = TaskNode(
                    output_label=f"{risk_id}_{i + 1}",
                    instructions=(f"{criterion.description}\nAnswer strictly with True or False."),
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    children=[],
                    label=f"{risk_id} | importance: {importance_str} | {criterion.description}",
                )
                child_nodes.append((node, importance_str))
                root_nodes.append(node)
                criterion_nodes_by_risk[risk_id].append(node)
                if self.debug:
                    logger.debug(
                        f"Created DAG root node for criterion: `{criterion}` for risk: {risk_id}",
                    )

            # Group nodes by importance for aggregation logic
            high_nodes = [n for n, imp in child_nodes if imp == "high"]
            medium_nodes = [n for n, imp in child_nodes if imp == "medium"]
            low_nodes = [n for n, imp in child_nodes if imp == "low"]

            high_labels = [n.output_label for n in high_nodes]
            medium_labels = [n.output_label for n in medium_nodes]
            low_labels = [n.output_label for n in low_nodes]

            # Compute threshold text for MEDIUM (ceil(len(medium)/2))
            m_total = len(medium_labels)
            m_required = (m_total + 1) // 2
            borderline_expr = (
                f"True if the count of True among the MEDIUM set equals {max(m_required - 1, 0)}; otherwise False."
                if m_total > 0
                else "False"
            )

            # Build readable fragments used in instructions
            high_clause = (
                f"All HIGH must be True. HIGH set: [{', '.join(high_labels)}]."
                if high_labels
                else "No HIGH criteria (treat as satisfied)."
            )
            medium_clause = (
                "MEDIUM requires at least half (rounded up) to be True. "
                f"MEDIUM set: [{', '.join(medium_labels)}]. "
                f"Total MEDIUM = {m_total}; required True = {m_required}."
                if m_total
                else "No MEDIUM criteria (treat as satisfied)."
            )
            low_clause = (
                "LOW criteria are used only as a tiebreaker if MEDIUM is borderline. "
                f"LOW set: [{', '.join(low_labels)}]."
                if low_labels
                else "No LOW criteria."
            )

            # Aggregation rule (per-risk)
            risk_agg_instructions = self._get_risk_agg_instructions(
                high_clause,
                medium_clause,
                m_required,
                borderline_expr,
                low_clause,
            )

            # Create the per-risk aggregation node
            risk_agg_node = TaskNode(
                output_label=f"{risk_id}_importance_aware_pass",
                instructions=risk_agg_instructions,
                evaluation_params=[],
                children=[],
                label=f"{risk_id} aggregation node (importance-aware)",
            )
            if self.debug:
                logger.debug(f"Created aggregation node for risk: {risk_id}")

            # Link every criterion node to the per-risk aggregation node
            for n, _imp in child_nodes:
                n.children = [risk_agg_node]

            final_risk_nodes.append(risk_agg_node)

        # WEIGHTED FINAL AGGREGATION
        weights: dict[str, float] = {rid: 1.0 for rid in criteria_by_risk.keys()}
        if risk_weights:
            for rid, w in risk_weights.items():
                if rid in weights:
                    weights[rid] = float(w)

        weight_lines = [f"- {rid}: {weights[rid]}" for rid in criteria_by_risk.keys()]
        weights_text = "\n".join(weight_lines)
        denom = sum(weights.values())
        if denom == 0:
            logger.error("Risk weights sum to zero; cannot compute weighted ratio.")
            raise ValueError("Risk weights sum to zero; cannot compute weighted ratio.")

        all_risk_outputs = [n.output_label for n in final_risk_nodes]

        verdicts = [
            VerdictNode(verdict="Weighted pass ratio >= 0.90", score=10.0),
            VerdictNode(verdict="Weighted pass ratio >= 0.75", score=7.5),
            VerdictNode(verdict="Weighted pass ratio >= 0.50", score=5.0),
            VerdictNode(verdict="Weighted pass ratio >= 0.25", score=2.5),
            VerdictNode(verdict="Weighted pass ratio < 0.25", score=0.0),
        ]

        summary_instructions = self._get_summary_instructions(
            all_risk_outputs,
            weights_text,
            denom,
        )

        risk_summary_node = NonBinaryJudgementNode(
            criteria=summary_instructions,
            children=verdicts,
            label="Final risk aggregation node (weighted).",
        )

        for risk_node in final_risk_nodes:
            risk_node.children = [risk_summary_node]

        if self.debug:
            logger.debug("Created final risk aggregation node and linked its parent nodes.")

        dag_metric = DAGMetric(
            name=f"Evaluate result based on risks (weighted): {', '.join(criteria_by_risk.keys())}",
            dag=DeepAcyclicGraph(root_nodes=root_nodes),
            verbose_mode=True,
        )
        return dag_metric, criterion_nodes_by_risk

    async def _generate_criteria_for_risk(
        self,
        risk_category: RiskCategory,
        context: str | None,
        content: str,
    ) -> tuple[str, list[Criterion]]:
        """Generate evaluation criteria for a single risk category."""
        risk_id = risk_category.value
        risk_description = risk_category.metadata.description

        logger.info(f"Processing risk: {risk_id}")

        messages = [self._default_system_message()]

        user_prompt = f"""
Risk ID: {risk_id}
Risk Description: {risk_description}

User Input: {context or "(no context provided)"}
Model Output: {content}
"""
        messages.append({"role": "user", "content": user_prompt})

        response: RiskCriteriaOutputSchema = await self.get_response_async(
            response_model=RiskCriteriaOutputSchema,
            messages=messages,
        )  # type: ignore[assignment]

        logger.info(f"Judge criteria obtained for risk: {risk_id}")
        return (risk_id, response.criteria)

    async def _arun(
        self,
        params: GuardrailInput,
        **kwargs: Any,
    ) -> GuardrailOutput:
        """
        Run the agent to generate criteria, build DAG metric, and evaluate.

        This is the core implementation that:
        1. Generates evaluation criteria for each risk category
        2. Builds a DAG metric from the criteria
        3. Evaluates the content against the DAG metric
        4. Returns GuardrailOutput with detected risks
        """
        # Use input categories or fallback to config defaults
        risk_categories = list(params.risk_categories or self.config.risk_categories)

        if not risk_categories:
            return GuardrailOutput(
                detected_risks=[],
                provider=self.__class__.__name__,
                extra={"score": 1.0, "reason": "No risk categories specified"},
            )

        # Validate category types (if enabled)
        self._validate_category_types(risk_categories)

        risk_weights = self._resolve_risk_weights(risk_categories)

        # Generate criteria for all risk categories in parallel
        results = await asyncio.gather(
            *[self._generate_criteria_for_risk(rc, params.context, params.content) for rc in risk_categories],
        )
        criteria_by_risk: dict[str, list[Criterion]] = dict(results)

        # Build DAG metric from criteria
        dag_metric, criterion_nodes_by_risk = self._build_dag_from_criteria(
            criteria_by_risk,
            risk_weights=risk_weights,
        )
        logger.info("DAG metric created.")

        # Evaluate the DAG metric
        test_case = LLMTestCase(
            input=params.context or "",
            actual_output=params.content,
        )
        dag_metric.measure(test_case)

        # Get score (normalized to 0-1 from 0-10)
        raw_score = dag_metric.score or 0.0
        score = raw_score / 10.0  # DAGMetric scores are 0-10

        # Build per-risk evaluation results with criterion verdicts
        risk_results: dict[RiskCategory, dict[str, Any]] = {}
        for risk_category in risk_categories:
            risk_id = risk_category.value
            criteria_list = criteria_by_risk.get(risk_id, [])
            nodes = criterion_nodes_by_risk.get(risk_id, [])

            # Merge criterion data with verdicts from nodes
            criteria_with_verdicts = []
            for criterion, node in zip(criteria_list, nodes):
                criterion_dict = criterion.model_dump()
                # Parse verdict from node._output (e.g., "True" or "False")
                verdict_str = node._output or ""
                criterion_dict["verdict"] = verdict_str.strip().lower() == "true"
                criteria_with_verdicts.append(criterion_dict)

            risk_results[risk_category] = {
                "criteria": criteria_with_verdicts,
            }

        # Determine detected risks based on threshold
        detected: list[RiskCategory] = []
        if score < self.config.pass_threshold:
            detected = list(risk_categories)

        if self.debug:
            logger.debug(
                f"[RiskAgent] Score: {score:.2f}, threshold: {self.config.pass_threshold}, "
                f"detected: {[r.value for r in detected]}",
            )

        extra: dict[str, Any] = {
            "score": score,
            "raw_score": raw_score,
            "reason": dag_metric.reason,
            "verbose_logs": dag_metric.verbose_logs,
        }
        if self.config.include_dag_metric:
            extra["dag_metric"] = dag_metric

        return GuardrailOutput(
            detected_risks=detected,
            risk_results=risk_results,
            provider=self.__class__.__name__,
            extra=extra,
        )

    # =========================================================================
    # GuardrailProtocol implementation
    # =========================================================================

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Sync GuardrailProtocol implementation."""
        return self.run(params)

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Async GuardrailProtocol implementation."""
        return await self.arun(params)
