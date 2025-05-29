import pytest
import asyncio
import os
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test
from scripts import run_lit_agent


@pytest.fixture(scope="module")
def query():
    return "How does the brown water effect affect hurricane landfall windspeeds in the US Gulf Coast?"


class RunLitAgentArgs:
    config: str
    query: str

    def __init__(self, config: str, query: str):
        self.config = config
        self.query = query


def get_test_cases():

    query = "How does the brown water effect affect hurricane landfall windspeeds in the US Gulf Coast?"

    async def _get_cases():
        cache_path = "tests/test_lit_agent_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                result = f.read()
                results = run_lit_agent.LitAgentOutputSchema.model_validate_json(
                    result
                ).results
        else:
            result = await run_lit_agent.main(
                RunLitAgentArgs(config="config/lit_agent.toml", query=query)
            )
            results = result.results
            with open(cache_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
        return [
            LLMTestCase(input=query, actual_output=str(r.model_dump())) for r in results
        ]

    return asyncio.get_event_loop().run_until_complete(_get_cases())


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", get_test_cases())
async def test_geval_correctness(test_case):
    metric = GEval(
        name="geval_correctness",
        criteria="Correctness - determine if the given outout is a correct and helpful scientific literature search result to the query",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    )
    print(f"TEST CASE OUTPUT: {test_case.actual_output}")
    assert_test(test_case, [metric])
