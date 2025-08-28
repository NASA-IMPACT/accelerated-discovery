from unittest.mock import AsyncMock

import pytest
from pydantic import HttpUrl

from akd.tools.fact_check import (
    FactCheckInputSchema,
    FactCheckOutputSchema,
    FactCheckTool,
    FactCheckToolConfig,
)


@pytest.fixture
def mock_api_start_response() -> dict:
    """Provides a mock JSON response for when the tool is first started."""
    return {"job_id": "mock-job-1234"}


@pytest.fixture
def mock_api_final_response() -> dict:
    """Provides a mock, successful JSON response from the fact-check API."""
    return {
        "fact_reasoner_score": {
            "factuality_score_per_atom": [{"a0": {"score": 0.9877, "support": "S"}}],
            "factuality_score": 1.0,
        },
        "supported_atoms": [{"id": "a0", "text": "The sky is blue"}],
        "not_supported_atoms": [],
        "contexts": [{"id": "c_a0_0", "title": "Why Is the Sky Blue?"}],
        "graph_id": "mock-graph-id-123",
        "logging_metadata": {},
    }


@pytest.fixture
def fact_check_tool():
    config = FactCheckToolConfig(
        base_url=HttpUrl("http://localhost:8011"),
        polling_interval_seconds=1,
    )
    tool = FactCheckTool(config=config)
    return tool


@pytest.mark.asyncio
async def test_fact_check_tool_polling_workflow(
    mocker,
    fact_check_tool,
    mock_api_start_response,
    mock_api_final_response,
):
    """
    Tests the full polling workflow by mocking both the POST and GET calls.
    """

    # Mock the POST call to /fact-check/start
    mock_post_response = mocker.Mock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = mock_api_start_response

    # Mock the GET call to /fact-check/status/{job_id}
    mock_get_response = mocker.Mock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "status": "completed",
        "result": mock_api_final_response,
    }

    # Patch both 'post' and 'get' methods of the httpx.AsyncClient
    mocker.patch(
        "akd.tools.fact_check.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_post_response,
    )
    mocker.patch(
        "akd.tools.fact_check.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_get_response,
    )

    # Prepare the input for the tool
    input_data = FactCheckInputSchema(
        question="What colour is the sky?",
        answer="The sky is blue.",
    )

    # This will call the mocked versions of post and then get.
    output = await fact_check_tool.arun(input_data)

    # Assert:
    assert isinstance(output, FactCheckOutputSchema)
    assert output.fact_reasoner_score["factuality_score"] == 1.0
    assert output.graph_id == "mock-graph-id-123"
    assert len(output.supported_atoms) == 1


def test_fact_check_tool_initialization(fact_check_tool):
    """Tests that the tool initializes with the correct config."""
    # This test remains the same and is a good check.
    assert fact_check_tool.config.base_url == HttpUrl("http://localhost:8011")
