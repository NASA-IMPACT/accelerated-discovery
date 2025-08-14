from unittest.mock import AsyncMock

import pytest

from akd.tools.fact_check import (
    FactCheckInputSchema,
    FactCheckOutputSchema,
    FactCheckTool,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_api_response() -> dict:
    """Provides a mock, successful JSON response from the fact-check API."""
    return {
        "fact_reasoner_score": {
            "factuality_score_per_atom": [{"a0": {"score": 0.987795, "support": "S"}}],
            "factuality_score": 1.0,
        },
        "supported_atoms": [{"id": "a0", "text": "The sky is blue"}],
        "not_supported_atoms": [],
        "contexts": [{"id": "c_a0_0", "title": "Why Is the Sky Blue?"}],
        "graph_id": "mock-graph-id-123",
    }


async def test_fact_check_tool_with_mock(mocker, mock_api_response):
    """ """

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response

    mocker.patch(
        new_callable=AsyncMock,
        return_value=mock_response,
    )

    tool = FactCheckTool()
    input_data = FactCheckInputSchema(
        question="What colour is the sky?",
        answer="The sky is blue.",
    )

    # This will call the mocked version of the tool.
    output = await tool.arun(input_data)

    assert isinstance(output, FactCheckOutputSchema)

    assert output.fact_reasoner_score["factuality_score"] == 1.0
    assert len(output.fact_reasoner_score["factuality_score_per_atom"]) == 1
    assert output.graph_id == "mock-graph-id-123"
    assert len(output.supported_atoms) == 1
