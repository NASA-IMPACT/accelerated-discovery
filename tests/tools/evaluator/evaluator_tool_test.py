from typing import List

import pytest
from deepeval.test_case import LLMTestCaseParams

from akd.structures import SearchResultItem
from akd.tools.evaluator.base_evaluator import (
    LLMEvaluator,
    LLMEvaluatorConfig,
    LLMEvaluatorInputSchema,
)


# ------- Test helpers (fake metrics) -------
class _FakeMetricBase:
    """Minimal BaseMetric-like object for unit tests."""

    def __init__(
        self,
        name: str,
        required_params: List[LLMTestCaseParams] | None = None,
        score_value: float = 0.7,
    ):
        self.name = name
        self.score = 0.0
        self.reason = ""
        # For LLMEvaluator._collect_metric_required_params
        # emulate non-DAG metric declaring evaluation_params
        self.evaluation_params = set(required_params or [])
        self._score_value = score_value

    @property
    def value(self):
        return self

    def measure(self, test_case):
        # Simple deterministic behavior for tests
        self.score = self._score_value
        self.reason = (
            f"{self.name} measured with input={bool(getattr(test_case, 'input', None))}, "
            f"output={bool(getattr(test_case, 'actual_output', None))}, "
            f"retrieval={bool(getattr(test_case, 'retrieval_context', None))}"
        )


def _fake_metric_requires_output(name="REQS_OUTPUT", score=0.5):
    return _FakeMetricBase(
        name=name,
        required_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],  # requires "output" (mapped to actual_output)
        score_value=score,
    )


def _fake_metric_requires_retrieval(name="REQS_RETRIEVAL", score=0.6):
    return _FakeMetricBase(
        name=name,
        required_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
        score_value=score,
    )


def _fake_metric_no_requirements(name="NO_REQS", score=0.9):
    return _FakeMetricBase(
        name=name,
        required_params=[],
        score_value=score,
    )


# ------ Tests -------------------------


@pytest.mark.asyncio
async def test_with_search_results_runs_structure_and_returns_two_cases():
    """
    Provides two SearchResultItem entries and relies on the configured STRUCTURE metric.
    Asserts we get two TestCaseEvaluationResult objects back.
    """
    evaluator = LLMEvaluator(config=LLMEvaluatorConfig(debug=True))
    evaluator._post_init()  # ensure defaults are expanded to metric instances

    # Build substantial search results
    sr1 = SearchResultItem(
        query="What is long COVID?",
        content=(
            "Long COVID refers to symptoms lasting more than four weeks after infection. "
            "Common issues include fatigue, shortness of breath, cognitive difficulties "
            "(often referred to as 'brain fog'), sleep problems, and mood changes. "
            "Research suggests multi-system involvement, including immune dysregulation and "
            "autonomic dysfunction."
        ),
        title="Understanding Long COVID",
        url="https://example.com/long-covid",
    )
    sr2 = SearchResultItem(
        query="What is long COVID?",
        content=(
            "Post-acute sequelae of SARS-CoV-2 infection (PASC), commonly known as long COVID, "
            "encompasses a range of persistent or relapsing symptoms. These may include "
            "exercise intolerance, post-exertional malaise, dysautonomia (e.g., POTS), "
            "and cardiopulmonary symptoms. Management often focuses on symptom-directed care "
            "and pacing strategies while research continues."
        ),
        title="Clinician Guide to Long COVID",
        url="https://example.com/pasc-guide",
    )

    evaluator.metrics = [_fake_metric_no_requirements(name="FAKE_STRUCTURE")]

    params = LLMEvaluatorInputSchema(
        search_results=[sr1, sr2],
        reference=None,  # optional (not yet tested)
    )

    result = await evaluator._arun(params)

    # We should have two results, one per SearchResultItem
    assert len(result.test_case_results) == 2

    # FAKE_STRUCTURE should have been the metric evaluated for each (using FAKE_STRUCURE to avoid calling deepeval metric in test)
    for tcr in result.test_case_results:
        assert len(tcr.metric_evaluations) == 1
        # The code uses the metric's .name
        assert tcr.metric_evaluations[0].metric == "FAKE_STRUCTURE"

    # Average score should be between 0 and 1
    assert 0.0 <= result.score <= 1.0


def test_model_validator_raises_if_missing_both_search_and_nonsearch_fields():
    """
    Triggers the model_validator error by providing neither search_results nor input/output/retrieval_context.
    """
    with pytest.raises(ValueError) as err:
        LLMEvaluatorInputSchema()
    msg = str(err.value)
    assert "If 'search_results' is not provided" in msg


@pytest.mark.asyncio
async def test_validator_passes_but_arun_raises_when_required_metric_params_missing():
    """
    Provide only 'input' so model_validator passes, but override evaluator.metrics
    with a fake metric requiring 'output' (actual_output) to force _arun's param check to fail.
    """
    evaluator = LLMEvaluator(config=LLMEvaluatorConfig(debug=True))
    evaluator._post_init()

    # Override metrics with a fake metric that requires actual_output
    evaluator.metrics = [_fake_metric_requires_output()]

    params = LLMEvaluatorInputSchema(
        input="Explain the mechanism of action.",
        # No output, no retrieval_context -> should pass validator but fail _arun checks
    )

    with pytest.raises(ValueError) as err:
        await evaluator._arun(params)

    msg = str(err.value)
    # Our error message should list the metric name and the missing param(s)
    assert "requires" in msg and "actual_output" in msg


@pytest.mark.asyncio
async def test_non_search_success_with_fake_metric_no_requirements():
    """
    A simple run using a fake metric that requires nothing.
    Ensures _arun executes and returns a single test_case result.
    """
    evaluator = LLMEvaluator(config=LLMEvaluatorConfig(debug=False, threshold=0.5))
    evaluator._post_init()
    evaluator.metrics = [_fake_metric_no_requirements(name="BLAH", score=0.88)]

    params = LLMEvaluatorInputSchema(
        input="Summarize the article.",
        output="This is a concise summary.",
        retrieval_context=["Some supporting context."],
    )

    result = await evaluator._arun(params)

    assert result.success is True
    assert len(result.test_case_results) == 1
    tcr = result.test_case_results[0]
    assert tcr.metric_evaluations[0].metric == "BLAH"
    assert abs(result.score - 0.88) < 1e-6


@pytest.mark.asyncio
async def test_arun_missing_multiple_required_params_lists_them():
    """
    Demonstrates that multiple required params are listed in the error message.
    We require BOTH output and retrieval_context, but supply only input.
    """
    evaluator = LLMEvaluator(config=LLMEvaluatorConfig(debug=True))
    evaluator._post_init()
    evaluator.metrics = [
        _fake_metric_requires_output(name="NEEDS_OUTPUT"),
        _fake_metric_requires_retrieval(name="NEEDS_RETRIEVAL"),
    ]

    params = LLMEvaluatorInputSchema(
        input="Only input provided here.",
    )

    with pytest.raises(ValueError) as err:
        await evaluator._arun(params)

    msg = str(err.value)
    # Ensure both missing fields are referenced in the message
    assert "actual_output" in msg
    assert "retrieval_context" in msg
