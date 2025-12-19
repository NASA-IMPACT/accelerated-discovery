"""RiskAgent guardrail provider implementing GuardrailProtocol.

This module provides RiskAgent, which generates evaluation criteria for risks
and builds DAG metrics for hierarchical evaluation. It directly uses
GuardrailInput/GuardrailOutput for unified interface with other guardrail providers.
"""

import asyncio
import re
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

from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent
from akd.agents._base import BaseAgentConfig
from akd.configs.prompts import RISK_REPORT_SYSTEM_PROMPT, RISK_SYSTEM_PROMPT
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


class RiskReportInputSchema(InputSchema):
    """Input schema for the Risk Report Agent (from feature branch)."""

    failed_criteria: dict[RiskCategory, list[str]] = Field(
        ...,
        description="Failed criteria per risk category.",
    )
    risky_content: str = Field(
        ...,
        description="Risky generated content that was evaluated.",
    )


class RiskReportOutputSchema(OutputSchema):
    """Output schema for risk report generation (from feature branch)."""

    risk_report: str = Field(
        ...,
        description="A report documenting detected risks in generated content.",
    )


class _RiskReportAgentConfig(BaseAgentConfig):
    """Configuration for the internal RiskReportAgent (from feature branch)."""

    system_prompt: str = RISK_REPORT_SYSTEM_PROMPT


class _RiskReportAgent(
    LiteLLMInstructorBaseAgent[RiskReportInputSchema, RiskReportOutputSchema],
):
    """Internal agent that generates risk reports for failed criteria.

    From feature/risks-in-decorator branch (Tigran's RiskReportAgent).
    """

    input_schema = RiskReportInputSchema
    output_schema = RiskReportOutputSchema
    config_schema = _RiskReportAgentConfig

    async def _arun(
        self,
        params: RiskReportInputSchema,
        **kwargs,
    ) -> RiskReportOutputSchema:
        """Generate a risk report from failed criteria."""
        messages = [self._default_system_message()]

        # Build risk definitions from RiskCategory metadata
        failed_criteria_str = {rc.value: criteria for rc, criteria in params.failed_criteria.items()}
        relevant_risk_definitions = {rc.value: rc.metadata.description for rc in params.failed_criteria.keys()}

        user_prompt = f"""
risky_content: {params.risky_content}

failed_criteria: {failed_criteria_str}

relevant_risk_definitions: {relevant_risk_definitions}
"""
        messages.append({"role": "user", "content": user_prompt})

        response: RiskReportOutputSchema = await self.get_response_async(
            response_model=RiskReportOutputSchema,
            messages=messages,
        )  # type: ignore[assignment]

        return response


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
    risk_weights: dict[RiskCategory, float] | None = Field(
        default=None,
        description="Per-risk weight overrides. Defaults to 1.0 for unspecified risks.",
    )
    include_dag_metric: bool = Field(
        default=False,
        description="Include full DAGMetric object in output extra (for advanced inspection).",
    )
    dag_verbose: bool = Field(
        default=True,
        description="Enable verbose mode for DAGMetric (detailed step logs).",
    )
    risk_report_config: _RiskReportAgentConfig = Field(
        default_factory=_RiskReportAgentConfig,
        description="Configuration for the internal risk report agent.",
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
        self._risk_report_agent = _RiskReportAgent(
            config=self.config.risk_report_config,
            debug=self.debug,
        )
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
        verdicts: list[VerdictNode],
    ) -> str:
        # Extract thresholds from verdict strings (all verdicts we create are strings)
        thresholds = sorted(
            [
                float(str(v.verdict).split("≥")[1].strip())
                for v in verdicts
                if isinstance(v.verdict, str) and "≥" in v.verdict
            ],
            reverse=True,
        )

        # Pick top three + next lower for the example
        top3 = thresholds[:3]
        example_lower = thresholds[3] if len(thresholds) > 3 else thresholds[-1]

        # Choose example ratio slightly below lowest of top3
        example_ratio = round(example_lower + (top3[-1] - example_lower) / 2 - 0.01, 2)
        top3_str = ", ".join(f"{t:.2f}" for t in top3)

        # Get verdict strings for display
        verdict_strs = [str(v.verdict) for v in verdicts if isinstance(v.verdict, str)]

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
            "6) Determine the verdict bucket using the following logic:\n"
            "   - Each verdict node defines a single threshold of the form 'Weighted pass ratio ≥ X'.\n"
            "   - Compute weighted_ratio as above.\n"
            "   - Start from the **highest threshold** (the largest X) and move **downward**.\n"
            "   - The first threshold where weighted_ratio ≥ X is the correct one — stop there.\n"
            "   - Do NOT select any higher thresholds that the ratio does not meet.\n"
            "   - Return exactly that node's verdict text as the final verdict.\n"
            "   - The associated score must be the numeric score linked to that node.\n"
            "   - If weighted_ratio is below the lowest threshold, return the node that explicitly represents ratios below 0.05.\n"
            f"   - Example: if weighted_ratio = {example_ratio:.2f}, the highest thresholds ({top3_str}) are not satisfied, "
            f"but {example_lower:.2f} is; therefore, the correct verdict is 'Weighted pass ratio ≥ {example_lower:.2f}'.\n\n"
            "The possible verdicts are given below:\n"
            f"{chr(10).join(verdict_strs)}"
        )

    def _build_dag_from_criteria(
        self,
        criteria_by_risk: dict[RiskCategory, list[Criterion]],
    ) -> tuple[DAGMetric, dict[RiskCategory, list[TaskNode]], dict[RiskCategory, TaskNode]]:
        """Build a DAGMetric from criteria grouped by risk.

        Returns:
            Tuple of (DAGMetric, criterion_nodes_by_risk, risk_agg_nodes_by_risk) where:
            - criterion_nodes_by_risk maps RiskCategory to list of TaskNodes in same order as criteria
            - risk_agg_nodes_by_risk maps RiskCategory to its aggregation TaskNode
        """
        root_nodes: list[TaskNode] = []
        final_risk_nodes: list[TaskNode] = []
        criterion_nodes_by_risk: dict[RiskCategory, list[TaskNode]] = {}
        risk_agg_nodes_by_risk: dict[RiskCategory, TaskNode] = {}

        for risk_category, criteria in criteria_by_risk.items():
            risk_id = risk_category.value
            child_nodes = []
            criterion_nodes_by_risk[risk_category] = []

            # Create root nodes (binary True/False) with importance-aware labeling
            for i, criterion in enumerate(criteria):
                importance = getattr(criterion, "importance", CriterionImportance.MEDIUM)

                node = TaskNode(
                    output_label=f"{risk_id}_{i + 1}",
                    instructions=(f"{criterion.description}\nAnswer strictly with True or False."),
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    children=[],
                    label=f"{risk_id} | importance: {importance.value} | {criterion.description}",
                )
                child_nodes.append((node, importance.value))
                root_nodes.append(node)
                criterion_nodes_by_risk[risk_category].append(node)
                if self.debug:
                    logger.debug(
                        f"Created DAG root node for criterion: `{criterion}` for risk: {risk_id}",
                    )

            # Group nodes by importance for aggregation logic
            high_nodes = [n for n, imp in child_nodes if imp == CriterionImportance.HIGH.value]
            medium_nodes = [n for n, imp in child_nodes if imp == CriterionImportance.MEDIUM.value]
            low_nodes = [n for n, imp in child_nodes if imp == CriterionImportance.LOW.value]

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
            risk_agg_nodes_by_risk[risk_category] = risk_agg_node

        # WEIGHTED FINAL AGGREGATION (defaults to 1.0 for unspecified risks)
        weights: dict[RiskCategory, float] = {
            rc: (self.config.risk_weights or {}).get(rc, 1.0) for rc in criteria_by_risk.keys()
        }

        weights_text = "\n".join([f"- {rc.value}: {w}" for rc, w in weights.items()])
        denom = sum(weights.values())
        if denom == 0:
            logger.error("Risk weights sum to zero; cannot compute weighted ratio.")
            raise ValueError("Risk weights sum to zero; cannot compute weighted ratio.")

        all_risk_outputs = [n.output_label for n in final_risk_nodes]

        # ----- Adaptive verdict generation based on number of risks ---
        num_risks = len(criteria_by_risk)

        # For 7 risks 14 buckets is ok, so number of buckets ~ num_risks + 4 for now (setting minimum of 5 buckets)
        num_buckets = max(5, num_risks + 4)
        step = 1.0 / (num_buckets - 1)

        scores = [round(i * step * 10, 1) for i in range(0, num_buckets)]
        thresholds = [round(i / 10 - step / 2, 2) for i in scores]
        thresholds[0] = 0.0

        verdicts = [
            VerdictNode(verdict=f"Weighted pass ratio ≥ {thr}", score=sco)
            for thr, sco in zip(reversed(thresholds), reversed(scores))
        ]

        summary_instructions = self._get_summary_instructions(
            all_risk_outputs,
            weights_text,
            denom,
            verdicts,
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
            name=f"Evaluate result based on risks (weighted): {', '.join(rc.value for rc in criteria_by_risk.keys())}",
            dag=DeepAcyclicGraph(root_nodes=root_nodes),
            model=self.config.model_name,
            verbose_mode=self.config.dag_verbose,
        )
        return dag_metric, criterion_nodes_by_risk, risk_agg_nodes_by_risk

    async def _generate_criteria_for_risk(
        self,
        risk_category: RiskCategory,
        context: str | None,
        content: str,
    ) -> list[Criterion]:
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
        return response.criteria

    def _extract_verdicts_from_nodes(
        self,
        criterion_nodes_by_risk: dict[RiskCategory, list[TaskNode]],
        risk_agg_nodes_by_risk: dict[RiskCategory, TaskNode],
    ) -> tuple[dict[str, bool], dict[RiskCategory, bool]]:
        """Extract verdicts directly from DAG nodes after a_measure() (primary method).

        DeepEval populates TaskNode._output with "True" or "False" after execution.
        This is more reliable than parsing verbose logs.

        Returns:
            Tuple of (criterion_verdicts, risk_verdicts) where:
            - criterion_verdicts maps criterion_id (e.g., "consistency_1") to verdict
            - risk_verdicts maps RiskCategory to whether it passed
        """
        criterion_verdicts: dict[str, bool] = {}
        risk_verdicts: dict[RiskCategory, bool] = {}

        # Per-criterion verdicts from Level 0 TaskNodes
        for risk_category, nodes in criterion_nodes_by_risk.items():
            risk_id = risk_category.value
            for i, node in enumerate(nodes):
                if node._output:
                    verdict = node._output.strip().lower() == "true"
                    criterion_verdicts[f"{risk_id}_{i + 1}"] = verdict

        # Per-risk aggregation verdicts from Level 1 TaskNodes
        for risk_category, agg_node in risk_agg_nodes_by_risk.items():
            if agg_node._output:
                risk_verdicts[risk_category] = agg_node._output.strip().lower() == "true"

        return criterion_verdicts, risk_verdicts

    def _extract_risk_verdicts_from_verbose_logs(
        self,
        dag_metric: DAGMetric,
        risk_categories: Sequence[RiskCategory],
    ) -> dict[RiskCategory, bool]:
        """Extract per-risk pass/fail verdicts from DAG verbose logs (fallback method).

        This parses verbose logs as a fallback when direct node access fails.

        From feature/risks-in-decorator branch (Tigran's implementation).

        Returns:
            Dict mapping RiskCategory to whether it passed (True) or failed (False).
        """
        # Build mapping from risk_id string to RiskCategory enum
        risk_id_to_category = {rc.value: rc for rc in risk_categories}
        passed_risks: dict[RiskCategory, bool] = {}

        # Get verbose steps from the DAG metric
        verbose_steps = getattr(dag_metric, "_verbose_steps", []) or []
        if not verbose_steps:
            return passed_risks

        blob = "\n".join(verbose_steps)

        # Level 1 aggregation nodes pattern (from feature branch)
        level1_pattern = re.compile(
            r"Label:\s*([\w\-]+)\s+aggregation node.*?\n[\s\S]*?\n\s*([\w\-]+_importance_aware_pass):\s*(True|False)\s*$",
            re.DOTALL | re.MULTILINE,
        )

        for match in level1_pattern.finditer(blob):
            risk_id = match.group(1)
            verdict = match.group(3).strip().lower()
            if risk_id in risk_id_to_category:
                passed_risks[risk_id_to_category[risk_id]] = verdict == "true"

            if self.debug:
                logger.debug(f"[RiskAgent] Parsed verdict for {risk_id}: {verdict}")

        return passed_risks

    def _extract_criterion_verdicts_from_verbose_logs(
        self,
        dag_metric: DAGMetric,
    ) -> dict[str, bool]:
        """Extract individual criterion verdicts from verbose logs.

        Returns:
            Dict mapping criterion_id (e.g., "consistency_1") to verdict (True/False).
        """
        verdicts: dict[str, bool] = {}

        verbose_steps = getattr(dag_metric, "_verbose_steps", []) or []
        if not verbose_steps:
            return verdicts

        # Pattern for criterion verdicts at Level 0
        verdict_pattern = re.compile(
            r"\n([\w\-]+_\d+):\s*\n?\s*(True|False)",
            re.IGNORECASE,
        )

        for block in verbose_steps:
            # Only Level 0
            if "Level == 0" not in block:
                continue

            for match in verdict_pattern.finditer(block):
                criterion_id = match.group(1)
                verdict = match.group(2).strip().lower() == "true"
                verdicts[criterion_id] = verdict

        return verdicts

    def _build_risk_results(
        self,
        risk_categories: Sequence[RiskCategory],
        criteria_by_risk: dict[RiskCategory, list[Criterion]],
        criterion_verdicts: dict[str, bool],
        risk_verdicts: dict[RiskCategory, bool],
    ) -> dict[RiskCategory, dict[str, Any]]:
        """Build per-risk evaluation results with criterion verdicts.

        Returns:
            Dict mapping RiskCategory to dict with criteria and pass status.
        """
        risk_results: dict[RiskCategory, dict[str, Any]] = {}

        for risk_category in risk_categories:
            risk_id = risk_category.value
            criteria_list = criteria_by_risk.get(risk_category, [])

            # Merge criterion data with verdicts
            criteria_with_verdicts = []
            for i, criterion in enumerate(criteria_list):
                criterion_dict = criterion.model_dump(mode="json")  # Serialize enums as values
                criterion_id = f"{risk_id}_{i + 1}"
                criterion_dict["verdict"] = criterion_verdicts.get(criterion_id, False)
                criteria_with_verdicts.append(criterion_dict)

            # Check if this specific risk passed
            risk_passed = risk_verdicts.get(risk_category, True)  # Default to passed if not found

            risk_results[risk_category] = {
                "criteria": criteria_with_verdicts,
                "passed": risk_passed,
            }

        return risk_results

    async def _generate_risk_report(
        self,
        detected_risks: list[RiskCategory],
        risk_results: dict[RiskCategory, dict[str, Any]],
        content: str,
    ) -> str | None:
        """Generate risk report for failed risks.

        Extracts failed HIGH importance criteria and generates a report.

        Returns:
            Risk report string, or None if no failed criteria or error.
        """
        if not detected_risks:
            return None

        # Extract failed HIGH importance criteria from risk_results
        failed_criteria: dict[RiskCategory, list[str]] = {}
        for rc in detected_risks:
            criteria = risk_results[rc]["criteria"]
            print(criteria)
            failed = [
                c["description"]
                for c in criteria
                if not c.get("verdict", True) and c.get("importance") == CriterionImportance.HIGH.value
            ]
            if failed:
                failed_criteria[rc] = failed

        if not failed_criteria:
            return None

        try:
            report_result = await self._risk_report_agent.arun(
                RiskReportInputSchema(
                    failed_criteria=failed_criteria,
                    risky_content=content,
                ),
            )
            return report_result.risk_report
        except Exception as e:
            logger.error(f"[RiskAgent] Error generating risk report: {e}")
            return None

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

        # Generate criteria for all risk categories in parallel
        results = await asyncio.gather(
            *[self._generate_criteria_for_risk(rc, params.context, params.content) for rc in risk_categories],
        )
        criteria_by_risk: dict[RiskCategory, list[Criterion]] = dict(zip(risk_categories, results))

        # Build DAG metric from criteria
        dag_metric, criterion_nodes_by_risk, risk_agg_nodes_by_risk = self._build_dag_from_criteria(
            criteria_by_risk,
        )
        logger.info("DAG metric created.")

        # Evaluate the DAG metric
        test_case = LLMTestCase(
            input=params.context or "",
            actual_output=params.content,
        )
        await dag_metric.a_measure(test_case)

        # Get score (normalized to 0-1 from 0-10)
        raw_score = dag_metric.score or 0.0
        score = raw_score / 10.0  # DAGMetric scores are 0-10

        # Primary: Extract verdicts directly from DAG nodes (populated by DeepEval after a_measure)
        criterion_verdicts, risk_verdicts = self._extract_verdicts_from_nodes(
            criterion_nodes_by_risk,
            risk_agg_nodes_by_risk,
        )

        # Fallback: Parse verbose logs if direct node extraction failed
        if not risk_verdicts:
            logger.warning("Direct node verdict extraction failed, falling back to verbose logs")
            risk_verdicts = self._extract_risk_verdicts_from_verbose_logs(dag_metric, risk_categories)
        if not criterion_verdicts:
            criterion_verdicts = self._extract_criterion_verdicts_from_verbose_logs(dag_metric)

        # Build per-risk evaluation results
        risk_results = self._build_risk_results(
            risk_categories,
            criteria_by_risk,
            criterion_verdicts,
            risk_verdicts,
        )
        detected_risks = [rc for rc, data in risk_results.items() if not data.get("passed", True)]

        if self.debug:
            logger.debug(
                f"[RiskAgent] Score: {score:.2f}, threshold: {self.config.pass_threshold}, "
                f"detected_risks: {[r.value for r in detected_risks]}",
            )

        # Generate risk report for failed risks
        risk_report = await self._generate_risk_report(detected_risks, risk_results, params.content)

        extra: dict[str, Any] = {
            "score": score,
            "raw_score": raw_score,
            "reason": dag_metric.reason,
            "verbose_logs": dag_metric.verbose_logs,
            "risk_report": risk_report,
        }
        if self.config.include_dag_metric:
            extra["dag_metric"] = dag_metric

        return GuardrailOutput(
            detected_risks=detected_risks,
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
