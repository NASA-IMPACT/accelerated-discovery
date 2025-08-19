from collections import deque
from enum import Enum
from typing import Iterable, List, Optional, Set, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from loguru import logger
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from akd.structures import SearchResultItem

from .._base import BaseTool, BaseToolConfig, InputSchema, OutputSchema
from .preset_evaluators import (
    accuracy,
    completeness,
    faithfulness,
    guardian,
    structure,
    timeliness,
    usefulness,
)


class EvalMetricDefinition(Enum):
    """
    Enumeration preset of evaluation metrics.
    """

    ACCURACY = accuracy
    COMPLETENESS = completeness
    FAITHFULNESS = faithfulness
    TIMELINESS = timeliness
    USEFULNESS = usefulness
    GUARDIAN = guardian
    STRUCTURE = structure


# from deepeval import Evaluator, SomeMetric
class LLMEvaluatorInputSchema(InputSchema):
    """Input schema for the LLMEvaluator"""

    input: Optional[str] = Field(
        None,
        description="User input or prompt sent to the LLM.",
    )
    output: Optional[str] = Field(None, description="LLM's generated output.")
    context: Optional[List[str]] = Field(
        None,
        description="Optional context used for generation.",
    )
    retrieval_context: Optional[List[str]] = Field(
        None,
        description="Optional retrieved content.",
    )
    reference: Optional[str] = Field(
        None,
        description="Optional reference output for comparison (e.g., gold label).",
    )
    search_results: Optional[List[SearchResultItem]] = Field(
        None,
        description="If provided, fields from it will be used instead of input/output/context.",
    )
    metrics: List[Optional[EvalMetricDefinition]] = Field(
        None,
        description="Preset metric to use.",
    )

    @model_validator(mode="after")
    def check_mutually_exclusive_fields(self) -> Self:
        if self.search_results:
            if self.input or self.output or self.retrieval_context:
                raise ValueError(
                    "If 'search_results' is provided, 'retrieval_context', 'input' and 'output' must not be provided.",
                )

        if not self.search_results:
            if not (self.input or self.retrieval_context or self.output):
                raise ValueError(
                    "If 'search_results' is not provided, 'input', 'output' or 'retrieval_context' must be provided.",
                )

        return self


class SingleEvaluationOutputSchema(OutputSchema):
    """Output schema for a single metric evaluation"""

    score: float = Field(..., description="The numeric score assigned by this metric.")
    reason: str = Field(
        ...,
        description="Textual explanation or justification for the score.",
    )
    metric: str = Field(..., description="Name of the metric used.")


class TestCaseEvaluationResult(OutputSchema):
    """Holds metric evaluations for a single LLMTestCase."""

    test_case_index: int = Field(
        ...,
        description="Index of the test case within the input.",
    )
    test_case_input: str = Field(..., description="Input used for this test case.")
    actual_output: str = Field(..., description="LLM output being evaluated.")
    retrieval_context: List[str] = Field(
        ...,
        description="Retrieved context for the test case.",
    )
    metric_evaluations: List[SingleEvaluationOutputSchema] = Field(
        ...,
        description="List of metric evaluations for this test case.",
    )


class LLMEvaluatorOutputSchema(OutputSchema):
    """Output schema for the overall LLMEvaluator result"""

    score: float = Field(
        ...,
        description="The average score across all metrics and all test cases.",
    )
    test_case_results: List[TestCaseEvaluationResult] = Field(
        ...,
        description="Evaluation results for each test case.",
    )
    success: bool = Field(
        ...,
        description="True if score >= threshold, False otherwise.",
    )


class LLMEvaluatorConfig(BaseToolConfig):
    """
    Configuration class for LLMEvaluator.
    This class can be extended to add evaluator-specific configurations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_metrics: List[BaseMetric] = Field(
        default_factory=lambda: [
            EvalMetricDefinition.USEFULNESS,
            EvalMetricDefinition.FAITHFULNESS,
            EvalMetricDefinition.COMPLETENESS,
            EvalMetricDefinition.ACCURACY,
            EvalMetricDefinition.TIMELINESS,
            EvalMetricDefinition.GUARDIAN,
            EvalMetricDefinition.STRUCTURE,
        ],
        description="List of metrics to use for evaluation. Can be extended with custom metrics.",
    )
    custom_metrics: List[BaseMetric] = Field(
        default_factory=list,
        description="List of instantiated custom metrics to use for evaluation (GEval or DAGMetric)",
    )
    threshold: float = Field(
        default=0.5,
        description="Default threshold for evaluation success",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )


class LLMEvaluator(BaseTool):
    input_schema = LLMEvaluatorInputSchema
    output_schema = LLMEvaluatorOutputSchema
    config_schema = LLMEvaluatorConfig
    field_mapping = {
        "input": "input",
        "output": "actual_output",
        "context": "context",
        "retrieval_context": "retrieval_context",
        "reference": "expected_output",
    }

    """
    Modular evaluator for LLM outputs using configurable criteria and deepeval metrics.
    """

    def __init__(
        self,
        config: LLMEvaluatorConfig | None = None,
        debug: bool = False,
    ):
        self.config = config or LLMEvaluatorConfig(debug=debug)
        super().__init__(config=self.config)

    def _post_init(self) -> None:
        self.metrics = [
            metric_def.value
            if isinstance(metric_def, EvalMetricDefinition)
            else metric_def
            for metric_def in (
                self.config.custom_metrics or self.config.default_metrics
            )
        ]

        self.threshold = self.config.threshold

    async def _arun(self, params: LLMEvaluatorInputSchema) -> LLMEvaluatorOutputSchema:
        self.metrics = params.metrics or self.metrics

        # Case 1: No search_results: simple evaluation
        if not params.search_results:
            all_scores, test_case_results = self._process_simple_inputs(params)

        # Case 2: With search_results: possibly multiple LLMTestCases in future
        else:
            all_scores, test_case_results = self._process_search_results(params)

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        return LLMEvaluatorOutputSchema(
            score=avg_score,
            test_case_results=test_case_results,
            success=avg_score >= self.threshold,
        )

    def _process_simple_inputs(
        self,
        params,
    ) -> Tuple[List[float], List[TestCaseEvaluationResult]]:
        test_case_results = []
        all_scores = []

        # Only enforce when user didn't supply search_results; the model_validator
        # already enforced base presence, but this ensures metric-specific needs.

        available_params = self._present_params_available(params)

        problems = []
        for m in self.metrics:
            required = self._collect_metric_required_params(m.value)
            if not required:
                # Could not introspect; optionally warn instead of failing hard.
                # problems.append(f"Warning: could not determine required params for metric '{getattr(m, 'name', type(m).__name__)}'; skipping strict check.")
                continue

            missing = required - available_params
            if missing:
                m_name = getattr(m, "name", type(m).__name__)
                problems.append(
                    f"Metric '{m_name}' requires {self._pretty_missing(missing)}, "
                    f"but they are missing from the provided input.",
                )

        if problems:
            raise ValueError(
                "Input does not satisfy the selected metrics:\n- "
                + "\n- ".join(problems)
                + "\n\nSupply the missing fields, remove the offending metrics, or provide a 'search_results'.",
            )

        params_dict = params.model_dump(exclude_none=True)
        filtered_params = {
            self.field_mapping[key]: value
            for key, value in params_dict.items()
            if key in self.field_mapping
        }
        # for some reason LLMTestCase requires 'input' and 'actual output', even if unused by the metrics
        filtered_params.setdefault("input", "")
        filtered_params.setdefault("actual_output", "")
        test_case = LLMTestCase(**filtered_params)
        metric_evals = []

        for m in self.metrics:
            metric = m.value
            metric.measure(test_case)
            metric_evals.append(
                SingleEvaluationOutputSchema(
                    score=metric.score,
                    reason=metric.reason,
                    metric=metric.name,
                ),
            )
            all_scores.append(metric.score)

        test_case_results.append(
            TestCaseEvaluationResult(
                test_case_index=0,
                test_case_input=test_case.input,
                actual_output=test_case.actual_output,
                retrieval_context=test_case.retrieval_context or [],
                metric_evaluations=metric_evals,
            ),
        )

        return all_scores, test_case_results

    def _process_search_results(
        self,
        params,
    ) -> Tuple[List[float], List[TestCaseEvaluationResult]]:
        test_case_results = []
        all_scores = []

        # --- Temporary warnings for unsupported metrics (TODO: design new metrics for SearchResultItem) ---
        unsupported_metrics = [
            m for m in self.metrics if m is not EvalMetricDefinition.STRUCTURE
        ]
        if unsupported_metrics:
            metric_names = [
                getattr(m, "name", type(m).__name__) for m in unsupported_metrics
            ]
            warning_msg = (
                f"Warning: The following metrics are not fully supported with search_results "
                f"and will be skipped or may behave unexpectedly: {', '.join(metric_names)}"
            )
            logger.warning(warning_msg)

        # --- Structure-based case ---
        if EvalMetricDefinition.STRUCTURE in self.metrics or any(
            ["structure" in m.name.lower() for m in self.metrics],
        ):
            # for now, the STRUCTURE metric is the only suitable one for search_results
            for sr in params.search_results:
                sections_test_case = LLMTestCase(
                    input=sr.query,
                    actual_output="",
                    retrieval_context=[sr.content],
                    expected_output=params.reference,
                )
                metric_evals = []

                # metric = EvalMetricDefinition.STRUCTURE.value
                metric = next(
                    (m.value for m in self.metrics if "structure" in m.name.lower()),
                    None,
                )
                metric.measure(sections_test_case)
                metric_evals.append(
                    SingleEvaluationOutputSchema(
                        score=metric.score,
                        reason=metric.reason,
                        metric=metric.name,
                    ),
                )
                all_scores.append(metric.score)

                test_case_results.append(
                    TestCaseEvaluationResult(
                        test_case_index=len(test_case_results),
                        test_case_input=sections_test_case.input,
                        actual_output=sections_test_case.actual_output,
                        retrieval_context=sections_test_case.retrieval_context or [],
                        metric_evaluations=metric_evals,
                    ),
                )

        return all_scores, test_case_results

    def _reverse_field_mapping(self, field_mapping: dict[str, str]) -> dict[str, str]:
        # e.g. {"input":"input","output":"actual_output"} -> {"input":"input","actual_output":"output"}
        return {v: k for k, v in field_mapping.items()}

    def _collect_metric_required_params(self, metric) -> Set["LLMTestCaseParams"]:
        """Best-effort collection of required LLMTestCaseParams for a metric."""
        required: Set[LLMTestCaseParams] = set()

        # DAGMetric path: traverse nodes and union their evaluation_params
        dag = getattr(metric, "dag", None)
        if dag is not None:
            roots = getattr(dag, "root_nodes", None)
            if roots:
                q = deque(roots)
                seen = set()
                while q:
                    node = q.popleft()
                    if id(node) in seen:
                        continue
                    seen.add(id(node))
                    eps = getattr(node, "evaluation_params", None)
                    if eps:
                        required.update(eps)
                    children = getattr(node, "children", None)
                    if children:
                        # children can be verdict nodes or nested judgement nodes
                        # only nested nodes will have their own evaluation_params
                        for ch in children:
                            # Push only if it *could* be a node with evaluation_params/children
                            if hasattr(ch, "evaluation_params") or hasattr(
                                ch,
                                "children",
                            ):
                                q.append(ch)
                return required

        # Non-DAG metrics that still declare evaluation_params
        eps = getattr(metric, "evaluation_params", None)
        if eps:
            required.update(eps)

        # If we can't introspect, return empty set (skip strict checking)
        return required

    def _present_params_available(self, params) -> Set["LLMTestCaseParams"]:
        """Return which LLMTestCaseParams are actually present (non-None) on this input."""
        available: Set[LLMTestCaseParams] = set()
        rev = self._reverse_field_mapping(self.field_mapping)
        for llm_param_name, schema_field in rev.items():
            if (
                hasattr(params, schema_field)
                and getattr(params, schema_field) is not None
            ):
                try:
                    available.add(LLMTestCaseParams(llm_param_name))
                except Exception:
                    # If field_mapping uses a key that isn't in the enum, ignore it
                    pass
        return available

    def _pretty_missing(self, missing: Iterable["LLMTestCaseParams"]) -> str:
        return ", ".join(sorted(m.value for m in missing))
