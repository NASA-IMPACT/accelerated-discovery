import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import Field
from akd.tools.search import SimpleAgenticLitSearchToolConfig
from .._base import BaseTool, BaseToolConfig, InputSchema, OutputSchema
from .preset_evaluators import (
    usefulness,
    faithfulness,
    completeness,
    accuracy,
    timeliness,
)


# from deepeval import Evaluator, SomeMetric
class LLMEvaluatorInputSchema(InputSchema):
    input: str
    output: str
    # TODO: implement reference output checking
    reference: Optional[str] = None


class SingleEvaluationOutputSchema(OutputSchema):
    score: float
    reason: str
    metric: str


class LLMEvaluatorOutputSchema(OutputSchema):

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
        default=0.5, description="Default threshold for evaluation success"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )


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
        self.config = config or LLMEvaluatorConfig(debug=debug)
        super().__init__(config=self.config)

    def _post_init(self) -> None:
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
            if self.config.custom_metrics
            else self.config.default_metrics

            self.threshold = config.threshold

        )

    async def _arun(self, params: LLMEvaluatorInputSchema) -> LLMEvaluatorOutputSchema:
        """ """
        test_case = LLMTestCase(input=params.input, actual_output=params.output)

        for metric in self.metrics:
            metric.measure(test_case)

        avg_score = sum(metric.score for metric in self.metrics) / len(self.metrics)

        return LLMEvaluatorOutputSchema(
            score=avg_score,
            success=avg_score >= self.threshold,
            evaluations=[
                SingleEvaluationOutputSchema(
                    score=metric.score, reason=metric.reason, metric=metric.name
                )
                for metric in self.metrics
            ],
        )
