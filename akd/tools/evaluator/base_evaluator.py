import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from .._base import BaseTool, BaseToolConfig
from .preset_evaluators import (
    usefulness,
    faithfulness,
    completeness,
    accuracy,
    timeliness,
)


# from deepeval import Evaluator, SomeMetric
class LLMEvaluatorInput(BaseToolConfig):
    input: str
    output: str
    # TODO: implement reference output checking
    reference: Optional[str] = None


class SinlgeEvaluationOutput(BaseToolConfig):
    score: float
    reason: str
    metric: str


class LLMEvaluatorOutput(BaseToolConfig):

    score: Dict[str, Any] = {}
    evalautions: List[SinlgeEvaluationOutput]

    # Add any additional fields needed for the output schema


class LLMEvaluator(BaseTool):

    input_schema = LLMEvaluatorInput
    output_schema = LLMEvaluatorOutput

    """
    Modular evaluator for LLM outputs using configurable criteria and deepeval metrics.
    Loads criteria from a JSON file and exposes an evaluate method for integration.
    """

    def __init__(self, custom_metrics: List[Dict] = []):
        """ """

        # add stock metrics
        # add custom evaluator methods (DAG, Criteria based GEval, etc.)
        # add _run_ and _arun methods for synchronous and asynchronous evaluation

        # return a dictionary with scores and explanations
        custom_evals = [
            GEval(
                **m,
                evaluation_params=[
                    # TODO: add option to use reference output
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
            )
            for m in custom_metrics
        ]

        self.metrics = (
            custom_evals
            if custom_evals
            else [usefulness, faithfulness, completeness, accuracy, timeliness]
        )

    async def arun(self, params: LLMEvaluatorInput) -> LLMEvaluatorOutput:
        """ """
        test_case = LLMTestCase(input=params.input_query, actual_output=params.output)

        for metric in self.metrics:
            metric.measure(test_case)

        return LLMEvaluatorOutput(
            score=sum(metric.score for metric in self.metrics) / len(self.metrics),
            evaluations=[
                SinlgeEvaluationOutput(
                    score=metric.score, reason=metric.reason, metric=metric.name
                )
                for metric in self.metrics
            ],
        )
