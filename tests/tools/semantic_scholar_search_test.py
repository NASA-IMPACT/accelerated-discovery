import asyncio

import pytest

from akd.tools.search import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolConfig,
    SemanticScholarSearchToolInputSchema,
    SemanticScholarSearchToolOutputSchema,
)

pytest_plugins = ("pytest_asyncio",)

def test_from_params_constructor():
    """
    Tests that the from_params classmethod correctly initializes the tool
    and its configuration.
    """
    search_tool = SemanticScholarSearchTool.from_params(max_results=5, debug=True)
    assert search_tool.config.max_results == 5
    # Test a default value
    assert search_tool.config.external_id == "DOI"


@pytest.mark.asyncio
async def test_fetch_paper_by_external_id():  # Renamed for clarity
    """
    Tests that fetch_paper_by_external_id can successfully retrieve
    and parse a specific paper using its ARXIV ID.
    """
    config = SemanticScholarSearchToolConfig()
    search_tool = SemanticScholarSearchTool(config=config, debug=True)

    known_arxiv_id = "1706.03762"
    input_schema = SemanticScholarSearchToolInputSchema(queries=[known_arxiv_id])

    results = await search_tool.fetch_paper_by_external_id(
        input_schema,
        external_id="ARXIV",
    )

    assert isinstance(results, list)
    assert len(results) == 1, (
        "Expected to find exactly one paper for the given ArXiv ID."
    )

    paper = results[0]
    # Check that the title and ArXiv ID match the paper we requested.
    assert paper.external_ids["ArXiv"] == known_arxiv_id


@pytest.mark.asyncio
async def test_arun():
    """
    Tests the main `arun` method to ensure the full process works.
    """
    config = SemanticScholarSearchToolConfig()
    search_tool = SemanticScholarSearchTool(config=config, debug=True)

    queries = ["Enhanced dependency parsing approaches"]
    input_schema = SemanticScholarSearchToolInputSchema(queries=queries, max_results=3)

    output = await search_tool.arun(input_schema)

    # Assertions to check the final, processed output
    assert isinstance(output, SemanticScholarSearchToolOutputSchema)
    assert len(output.results) > 0, "No results found"

    first_result = output.results[0]
    assert first_result.url, "No url included"
    assert first_result.title, "No title included"
    assert first_result.content, "No content included"


async def main():
    """Runs all the defined tests."""
    test_from_params_constructor()
    await test_fetch_paper_by_external_id()
    await test_arun()


if __name__ == "__main__":
    asyncio.run(main())
