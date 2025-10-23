"""
End-to-end parameterized tests for search tools at the public arun() level.

Simple tests for science category with single queries only.
Tests the public-facing interface that users actually call.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd.structures import SearchResultItem
from akd.tools.search import SearchToolOutputSchema
from akd.tools.search.searxng_search import SearxNGSearchToolInputSchema
from akd.tools.search.serper import SerperSearchToolInputSchema

# ============================================================================
# Test Scenarios - Science Category Only, Single Query
# ============================================================================

# Different search queries for science category
SCIENCE_QUERIES = [
    pytest.param("quantum computing", id="quantum-computing"),
    pytest.param("machine learning", id="machine-learning"),
    pytest.param("neural networks", id="neural-networks"),
]

# Different max_results values
MAX_RESULTS_VALUES = [
    pytest.param(1, id="1-result"),
    pytest.param(5, id="5-results"),
    pytest.param(10, id="10-results"),
]


# ============================================================================
# Serper E2E Tests (Public arun() Interface) - Science Category
# ============================================================================


class TestSerperSearchToolE2E:
    """Test Serper search tool via public arun() interface - science category only."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", SCIENCE_QUERIES)
    async def test_arun_science_category(
        self,
        mock_serper_tool,
        mock_serper_scholar_response,
        query,
    ):
        """Test arun() with different science queries."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_scholar_response)
            mock_client.post.return_value = mock_response

            # Create input - single query, science category
            input_params = SerperSearchToolInputSchema(
                queries=[query],
                category="science",
                max_results=5,
            )

            # Call public arun() interface
            result = await mock_serper_tool.arun(input_params)

            # Verify output structure
            assert isinstance(result, SearchToolOutputSchema)
            assert hasattr(result, "results")
            assert hasattr(result, "extra")
            assert isinstance(result.results, list)

            # Verify all results are valid SearchResultItem
            for item in result.results:
                assert isinstance(item, SearchResultItem)
                assert item.engine == "serper"
                assert item.url is not None
                assert item.title is not None
                assert item.query == query

            # Verify metadata
            assert isinstance(result.extra, dict)
            assert "credits" in result.extra

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_results", MAX_RESULTS_VALUES)
    async def test_arun_max_results(
        self,
        mock_serper_tool,
        mock_serper_scholar_response,
        max_results,
    ):
        """Test that arun() respects max_results parameter."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_scholar_response)
            mock_client.post.return_value = mock_response

            input_params = SerperSearchToolInputSchema(
                queries=["quantum computing"],
                category="science",
                max_results=max_results,
            )

            result = await mock_serper_tool.arun(input_params)

            assert isinstance(result, SearchToolOutputSchema)
            assert len(result.results) <= max_results

    @pytest.mark.asyncio
    async def test_arun_empty_results(
        self,
        mock_serper_tool,
        mock_serper_empty_response,
    ):
        """Test arun() handles empty results gracefully."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_empty_response)
            mock_client.post.return_value = mock_response

            input_params = SerperSearchToolInputSchema(
                queries=["nonexistent xyz123 query"],
                category="science",
                max_results=10,
            )

            result = await mock_serper_tool.arun(input_params)

            assert isinstance(result, SearchToolOutputSchema)
            assert len(result.results) == 0
            assert isinstance(result.extra, dict)


# ============================================================================
# SearxNG E2E Tests (Public arun() Interface) - Science Category
# ============================================================================


class TestSearxNGSearchToolE2E:
    """Test SearxNG search tool via public arun() interface - science category only."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", SCIENCE_QUERIES)
    async def test_arun_science_category(
        self,
        sample_searxng_config,
        mock_searxng_response,
        query,
    ):
        """Test arun() with different science queries."""
        from akd.tools.search.searxng_search import SearxNGSearchTool

        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock the internal _fetch_search_results_paginated method
        async def mock_fetch_results(client, query_str, cat, target_results):
            return [
                SearchResultItem(
                    url=result["url"],
                    title=result["title"],
                    content=result["content"],
                    query=query_str,
                    score=result.get("score", 0.5),
                    engine=result.get("engine", "google"),
                )
                for result in mock_searxng_response["results"][:5]
                if result.get("score", 0) >= tool.config.score_cutoff
            ]

        with patch.object(tool, "_fetch_search_results_paginated", mock_fetch_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=[query],
                category="science",
                max_results=5,
            )

            # Call public arun() interface
            result = await tool.arun(input_params)

            # Verify output structure
            assert isinstance(result, SearchToolOutputSchema)
            assert isinstance(result.results, list)

            # Verify all results are valid
            for item in result.results:
                assert isinstance(item, SearchResultItem)
                assert item.url is not None
                assert item.title is not None
                assert item.query == query

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_results", MAX_RESULTS_VALUES)
    async def test_arun_max_results(
        self,
        sample_searxng_config,
        mock_searxng_response,
        max_results,
    ):
        """Test that arun() respects max_results parameter."""
        from akd.tools.search.searxng_search import SearxNGSearchTool

        tool = SearxNGSearchTool(config=sample_searxng_config)

        async def mock_fetch_results(client, query_str, cat, target_results):
            return [
                SearchResultItem(
                    url=result["url"],
                    title=result["title"],
                    content=result["content"],
                    query=query_str,
                    score=result.get("score", 0.5),
                )
                for result in mock_searxng_response["results"][:target_results]
            ]

        with patch.object(tool, "_fetch_search_results_paginated", mock_fetch_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["quantum computing"],
                category="science",
                max_results=max_results,
            )

            result = await tool.arun(input_params)

            assert isinstance(result, SearchToolOutputSchema)
            assert len(result.results) <= max_results

    @pytest.mark.asyncio
    async def test_arun_empty_results(
        self,
        sample_searxng_config,
    ):
        """Test arun() handles empty results gracefully."""
        from akd.tools.search.searxng_search import SearxNGSearchTool

        tool = SearxNGSearchTool(config=sample_searxng_config)

        async def mock_fetch_results(client, query_str, cat, target_results):
            return []  # Empty results

        with patch.object(tool, "_fetch_search_results_paginated", mock_fetch_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["nonexistent xyz123 query"],
                category="science",
                max_results=10,
            )

            result = await tool.arun(input_params)

            assert isinstance(result, SearchToolOutputSchema)
            assert len(result.results) == 0
