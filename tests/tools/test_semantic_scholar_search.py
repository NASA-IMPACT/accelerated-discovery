from unittest.mock import AsyncMock

import pytest
from pydantic import HttpUrl

from akd.structures import SearchResultItem
from akd.tools.search.semantic_scholar_search import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolInputSchema,
    SemanticScholarSearchToolOutputSchema,
)


@pytest.fixture
def mock_arun_output() -> SemanticScholarSearchToolOutputSchema:
    """Provides a realistic, successful output object from the arun method."""
    return SemanticScholarSearchToolOutputSchema(
        results=[
            SearchResultItem(
                title="A Mock Paper on Transformer Architectures",
                url=HttpUrl("https://www.semanticscholar.org/paper/a1b2c3d4e5f6"),
                content="This paper discusses recent advances in transformer models for NLP.",
                query="Recent advances in transformer architectures",
            ),
        ],
        category="science",
    )


def test_from_params_constructor():
    """
    Tests that the from_params classmethod correctly initializes the tool
    and its configuration.
    """
    search_tool = SemanticScholarSearchTool.from_params(max_results=5, debug=True)
    assert search_tool.config.max_results == 5
    # Test a default value from the config
    assert search_tool.config.base_url == HttpUrl("https://api.semanticscholar.org")


@pytest.mark.asyncio
async def test_arun_with_direct_mock(mocker, mock_arun_output):
    """
    Tests the main `arun` method by directly mocking the internal `_arun` method.
    """
    # Patch the internal _arun method to return our mock output directly.
    mocker.patch(
        "akd.tools.search.semantic_scholar_search.SemanticScholarSearchTool._arun",
        new_callable=AsyncMock,
        return_value=mock_arun_output,
    )

    # Initialize the tool and input data.
    tool = SemanticScholarSearchTool()
    input_schema = SemanticScholarSearchToolInputSchema(
        queries=["Recent advances in transformer architectures"],
        max_results=3,
    )

    # This will now call the mocked version of _arun.
    output = await tool.arun(input_schema)

    assert isinstance(output, SemanticScholarSearchToolOutputSchema)
    assert len(output.results) > 0, "No results found"
    assert output.results[0].title == "A Mock Paper on Transformer Architectures"
