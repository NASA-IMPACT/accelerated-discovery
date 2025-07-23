from typing import Any, Dict, List, Optional

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCaseParams
from pydantic import Field, model_validator
from typing_extensions import Self

from akd.tools.search import SearchResultItem

from .._base import BaseTool, BaseToolConfig, InputSchema, OutputSchema
from .custom_deepeval_extensions import LLMTestCaseGuardian
from .preset_evaluators import (
    accuracy,
    completeness,
    faithfulness,
    timeliness,
    usefulness,
)


# from deepeval import Evaluator, SomeMetric
class LLMEvaluatorInputSchema(InputSchema):
    """
    Input schema for Evaluator tool
    """

    input: str
    output: str
    search_result: SearchResultItem
    # TODO: implement reference output checking
    reference: Optional[str] = None

    @model_validator(mode="after")
    def check_mutually_exclusive_fields(self) -> Self:
        if self.search_result:
            if self.input or self.output:
                raise ValueError(
                    "If 'search_result' is provided, 'input' and 'output' must not be provided.",
                )
        return self


class SingleEvaluationOutputSchema(OutputSchema):
    """
    Output schema single evaluation
    """

    score: float
    reason: str
    metric: str


class LLMEvaluatorOutputSchema(OutputSchema):
    """
    Output schema for Evaluator tool
    """

    score: float
    evaluations: List[SingleEvaluationOutputSchema]
    success: bool

    # Add any additional fields needed for the output schema


class LLMEvaluatorConfig(BaseToolConfig):
    """
    Configuration class for LLMEvaluator.
    This class can be extended to add evaluator-specific configurations.
    """

    default_metrics: List[BaseMetric] = Field(
        default_factory=lambda: [
            usefulness,
            faithfulness,
            completeness,
            accuracy,
            timeliness,
        ],
        description="List of metrics to use for evaluation. Can be extended with custom metrics.",
    )
    custom_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of custom metrics to use for evaluation. Each metric should be a dictionary with the necessary parameters.",
    )
    threshold: float = Field(
        default=0.5,
        description="Default threshold for evaluation success",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


class LLMEvaluator(BaseTool):
    input_schema = LLMEvaluatorInputSchema
    output_schema = LLMEvaluatorOutputSchema
    config_schema = LLMEvaluatorConfig

    """
    Modular evaluator for LLM outputs using configurable criteria and deepeval metrics.
    Loads criteria from a JSON file and exposes an evaluate method for integration.
    """

    def __init__(
        self,
        config: LLMEvaluatorConfig | None = None,
        debug: bool = False,
    ):
        """ """
        config = config or LLMEvaluatorConfig(debug=debug)
        super().__init__(config=config)
        # add stock metrics
        # add custom evaluator methods (DAG, Criteria based GEval, etc.)
        # add _run_ and _arun methods for synchronous and asynchronous evaluation

        # return a dictionary with scores and explanations

        self.metrics = (
            [
                GEval(
                    **m,
                    evaluation_params=[
                        # TODO: add option to use reference output
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                )
                for m in config.custom_metrics
            ]
            if config.custom_metrics
            else config.default_metrics
        )

        self.threshold = config.threshold

    async def _arun(self, params: LLMEvaluatorInputSchema) -> LLMEvaluatorOutputSchema:
        """ """

        if not params.search_result:
            test_case = LLMTestCaseGuardian(
                input=params.input,
                actual_output=params.output,
            )
        else:
            test_case = LLMTestCaseGuardian(search_result=params.search_result)

        for metric in self.metrics:
            metric.measure(test_case)

        avg_score = sum(metric.score for metric in self.metrics) / len(self.metrics)

        return LLMEvaluatorOutputSchema(
            score=avg_score,
            success=avg_score >= self.threshold,
            evaluations=[
                SingleEvaluationOutputSchema(
                    score=metric.score,
                    reason=metric.reason,
                    metric=metric.name,
                )
                for metric in self.metrics
            ],
        )
