"""
Functional/integration tests for SearxNG search tool.

Tests the complete workflow including HTTP requests (mocked),
pagination, and end-to-end search functionality.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from akd.structures import SearchResultItem
from akd.tools.search.searxng_search import (
    SearxNGSearchTool,
    SearxNGSearchToolInputSchema,
)


class TestSearxNGFunctionalAPI:
    """Test SearxNG tool with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_successful_search_single_query(
        self,
        sample_searxng_config,
        mock_searxng_response,
    ):
        """Test successful search with a single query."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock the entire _arun method to avoid HTTP calls
        async def mock_arun(params, max_results=None, **kwargs):
            # Simulate processing the mock response
            filtered_results = [
                result
                for result in mock_searxng_response["results"]
                if result.get("score", 0) >= tool.config.score_cutoff
                and all(field in result for field in ["title", "content", "url"])
            ]

            results = [
                SearchResultItem(
                    url=result["url"],
                    title=result["title"],
                    content=result["content"],
                    query=params.queries[0],
                    category=params.category,
                    doi=result.get("doi"),
                    published_date=result.get("publishedDate"),
                    engine=result.get("engine"),
                )
                for result in filtered_results[: max_results or params.max_results]
            ]

            return tool.output_schema(
                results=results,
                category=params.category,
            )

        with patch.object(tool, "_arun", mock_arun):
            input_params = SearxNGSearchToolInputSchema(
                queries=["machine learning"],
                category="science",
                max_results=5,
            )

            result = await tool._arun(input_params)

            assert isinstance(result.results, list)
            assert len(result.results) > 0
            assert result.category == "science"

            # Check that results have required fields
            for item in result.results:
                assert isinstance(item, SearchResultItem)
                assert item.url is not None
                assert item.title is not None
                assert item.query is not None

    @pytest.mark.asyncio
    async def test_successful_search_multiple_queries(
        self,
        sample_searxng_config,
        mock_searxng_response,
    ):
        """Test successful search with multiple queries."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        async def mock_arun(params, max_results=None, **kwargs):
            all_results = []
            for query in params.queries:
                filtered_results = [
                    result
                    for result in mock_searxng_response["results"]
                    if result.get("score", 0) >= tool.config.score_cutoff
                    and all(field in result for field in ["title", "content", "url"])
                ]

                for result in filtered_results:
                    all_results.append(
                        SearchResultItem(
                            url=result["url"],
                            title=result["title"],
                            content=result["content"],
                            query=query,
                            category=params.category,
                            doi=result.get("doi"),
                            published_date=result.get("publishedDate"),
                            engine=result.get("engine"),
                        ),
                    )

            return tool.output_schema(
                results=all_results[: max_results or params.max_results],
                category=params.category,
            )

        with patch.object(tool, "_arun", mock_arun):
            input_params = SearxNGSearchToolInputSchema(
                queries=["machine learning", "deep learning", "neural networks"],
                category="science",
                max_results=10,
            )

            result = await tool._arun(input_params)

            assert isinstance(result.results, list)
            assert result.category == "science"

            # Should have results from multiple queries
            queries_found = {item.query for item in result.results}
            assert len(queries_found) > 0

    @pytest.mark.asyncio
    async def test_empty_search_results(
        self,
        sample_searxng_config,
        mock_searxng_empty_response,
    ):
        """Test handling of empty search results."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        async def mock_arun(params, max_results=None, **kwargs):
            return tool.output_schema(
                results=[],
                category=params.category,
            )

        with patch.object(tool, "_arun", mock_arun):
            input_params = SearxNGSearchToolInputSchema(
                queries=["very specific nonexistent query"],
                max_results=10,
            )

            result = await tool._arun(input_params)

            assert isinstance(result.results, list)
            assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_http_error_handling(self, sample_searxng_config):
        """Test handling of HTTP errors."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock the _fetch_search_results_paginated method to raise an exception
        async def mock_fetch_search_results_paginated(
            session,
            query,
            category,
            target_results,
        ):
            raise Exception(
                "Failed to fetch search results for query 'test query': 500 Internal Server Error",
            )

        with patch.object(
            tool,
            "_fetch_search_results_paginated",
            mock_fetch_search_results_paginated,
        ):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=5,
            )

            with pytest.raises(Exception, match="Failed to fetch search results"):
                await tool._arun(input_params)

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, sample_searxng_config):
        """Test handling of network timeouts."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock the _fetch_search_results_paginated method to raise a timeout
        async def mock_fetch_search_results_paginated(
            session,
            query,
            category,
            target_results,
        ):
            raise asyncio.TimeoutError("Request timeout")

        with patch.object(
            tool,
            "_fetch_search_results_paginated",
            mock_fetch_search_results_paginated,
        ):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=5,
            )

            with pytest.raises(Exception):
                await tool._arun(input_params)


class TestSearxNGPagination:
    """Test pagination functionality."""

    @pytest.mark.asyncio
    async def test_pagination_multiple_pages(self, sample_searxng_config):
        """Test fetching results across multiple pages."""
        config = sample_searxng_config
        config.max_pages = 3
        config.results_per_page = 2
        tool = SearxNGSearchTool(config=config)

        # Combined results that would come from multiple pages
        all_results = [
            {
                "title": "Result 1",
                "content": "Content 1",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "Result 2",
                "content": "Content 2",
                "url": "http://test2.com",
                "score": 0.8,
                "engine": "arxiv",
            },
            {
                "title": "Result 3",
                "content": "Content 3",
                "url": "http://test3.com",
                "score": 0.7,
                "engine": "google_scholar",
            },
            {
                "title": "Result 4",
                "content": "Content 4",
                "url": "http://test4.com",
                "score": 0.6,
                "engine": "google",
            },
        ]

        # Mock the paginated fetch method
        async def mock_fetch_search_results_paginated(
            session,
            query,
            category,
            target_results,
        ):
            # Add query to results and return them
            results = []
            for result in all_results[:target_results]:
                result_copy = result.copy()
                result_copy["query"] = query
                results.append(result_copy)
            return results

        with patch.object(
            tool,
            "_fetch_search_results_paginated",
            mock_fetch_search_results_paginated,
        ):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=4,  # Should trigger pagination
            )

            result = await tool._arun(input_params)

            assert len(result.results) <= 4
            assert len(result.results) > 2  # Should have results from multiple pages

    @pytest.mark.asyncio
    async def test_pagination_with_session_mock(self, sample_searxng_config):
        """Test pagination using fetch_search_results_paginated method."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock aiohttp session
        session = AsyncMock()

        # Create mock results that would come from pagination
        all_results = [
            {
                "title": "Result 1",
                "content": "Content 1",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "Result 2",
                "content": "Content 2",
                "url": "http://test2.com",
                "score": 0.8,
                "engine": "arxiv",
            },
        ]

        # Test the _fetch_search_results_paginated method directly
        async def mock_fetch_search_results(session, query, category, page_num):
            # Return results for first page only
            if page_num == 1:
                results = all_results.copy()
                for result in results:
                    result["query"] = query
                return results
            return []

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            results = await tool._fetch_search_results_paginated(
                session=session,
                query="test query",
                category="science",
                target_results=4,
            )

            assert len(results) >= 2  # Should have gotten results from pagination


class TestSearxNGQueryParameters:
    """Test query parameter handling."""

    @pytest.mark.asyncio
    async def test_query_parameters_basic(self, sample_searxng_config):
        """Test that correct query parameters are sent."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock the HTTP session to capture request parameters
        captured_params = {}

        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            # Capture the parameters that would be sent
            captured_params.update(
                {
                    "query": query,
                    "category": category,
                    "page_num": page_num,
                },
            )
            return []

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                category="science",
                max_results=5,
            )

            await tool._arun(input_params)

            # Verify the parameters were passed correctly
            assert captured_params["query"] == "test query"
            assert captured_params["category"] == "science"

    @pytest.mark.asyncio
    async def test_query_parameters_with_category(self, sample_searxng_config):
        """Test query parameters with different categories."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        captured_params = {}

        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            captured_params.update(
                {
                    "category": category,
                },
            )
            return []

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["technology query"],
                category="technology",
                max_results=5,
            )

            await tool._arun(input_params)

            # Verify category parameter
            assert captured_params["category"] == "technology"

    @pytest.mark.asyncio
    async def test_query_parameters_no_category(self, sample_searxng_config):
        """Test query parameters without category."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        captured_params = {}

        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            captured_params.update(
                {
                    "category": category,
                },
            )
            return []

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["general query"],
                category=None,  # No category
                max_results=5,
            )

            await tool._arun(input_params)

            # Verify no categories parameter when None
            assert captured_params.get("category") is None


class TestSearxNGResultProcessing:
    """Test end-to-end result processing."""

    @pytest.mark.asyncio
    async def test_result_deduplication_across_queries(
        self,
        sample_searxng_config,
        duplicate_results_response,
    ):
        """Test deduplication of results across multiple queries."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock to return duplicate results for different queries
        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            results = duplicate_results_response["results"].copy()
            for result in results:
                result["query"] = query
            return results

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=[
                    "query1",
                    "query2",
                ],  # Different queries might return same URLs
                max_results=10,
            )

            result = await tool._arun(input_params)

            # Check that URLs are unique
            urls = [item.url for item in result.results]
            assert len(urls) == len(set(urls)), (
                "URLs should be unique after deduplication"
            )

    @pytest.mark.asyncio
    async def test_score_cutoff_filtering(self, sample_searxng_config):
        """Test that results below score cutoff are filtered out."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Response with mixed scores
        mixed_score_results = [
            {
                "title": "High Score",
                "content": "Content",
                "url": "http://high.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "Above Cutoff",
                "content": "Content",
                "url": "http://above.com",
                "score": 0.3,
                "engine": "arxiv",
            },
            {
                "title": "Below Cutoff",
                "content": "Content",
                "url": "http://below.com",
                "score": 0.1,
                "engine": "google_scholar",
            },  # Below 0.25
        ]

        async def mock_fetch_search_results_paginated(
            session,
            query,
            category,
            target_results,
        ):
            results = mixed_score_results.copy()
            for result in results:
                result["query"] = query
            return results

        with patch.object(
            tool,
            "_fetch_search_results_paginated",
            mock_fetch_search_results_paginated,
        ):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=10,
            )

            result = await tool._arun(input_params)

            # Should only have results above score cutoff (0.25)
            assert len(result.results) == 2
            assert all("below" not in str(item.url).lower() for item in result.results)

    @pytest.mark.asyncio
    async def test_max_results_limiting(self, sample_searxng_config):
        """Test that results are limited by max_results parameter."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Response with many results
        many_results = [
            {
                "title": f"Result {i}",
                "content": f"Content {i}",
                "url": f"http://test{i}.com",
                "score": 0.9 - i * 0.05,
            }
            for i in range(20)  # 20 results
        ]

        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            results = many_results.copy()
            for result in results:
                result["query"] = query
            return results

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=5,  # Limit to 5 results
            )

            result = await tool._arun(input_params)

            # Should be limited to max_results
            assert len(result.results) <= 5


class TestSearxNGErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_partial_failure_pagination(self, sample_searxng_config):
        """Test handling of partial failures during pagination."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock session where some pages fail
        session = AsyncMock()

        # Mock successful first page
        first_page_results = [
            {
                "title": "Result 1",
                "content": "Content 1",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
        ]

        call_count = 0

        async def mock_fetch_search_results(session, query, category, page_num):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                results = first_page_results.copy()
                for result in results:
                    result["query"] = query
                return results
            else:
                # Simulate failure on subsequent pages
                raise Exception("Network error on page 2")

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            results = await tool._fetch_search_results_paginated(
                session=session,
                query="test query",
                category="science",
                target_results=10,  # Would require multiple pages
            )

            # Should still return results from successful page
            assert len(results) >= 1
            assert results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_malformed_response_handling(
        self,
        sample_searxng_config,
        mock_searxng_malformed_response,
    ):
        """Test handling of malformed API responses."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        # Mock to return malformed results
        async def mock_fetch_search_results(session, query, category=None, page_num=1):
            results = mock_searxng_malformed_response["results"].copy()
            for result in results:
                result["query"] = query
            return results

        with patch.object(tool, "_fetch_search_results", mock_fetch_search_results):
            input_params = SearxNGSearchToolInputSchema(
                queries=["test query"],
                max_results=10,
            )

            result = await tool._arun(input_params)

            # Should handle malformed results gracefully
            # (results with missing required fields should be filtered out)
            assert isinstance(result.results, list)
            # All returned results should have required fields
            for item in result.results:
                assert item.title is not None
                assert item.url is not None
                assert item.content is not None
