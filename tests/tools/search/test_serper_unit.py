"""
Unit tests for Serper search tool.

Tests individual components and methods of the SerperSearchTool
without making actual HTTP requests to avoid credit usage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr
from pydantic.networks import HttpUrl

from akd.structures import SearchResultItem
from akd.tools.search.serper import (
    SerperSearchTool,
    SerperSearchToolConfig,
    SerperSearchToolInputSchema,
    SerperSearchToolOutputSchema,
)


class TestSerperSearchToolConfig:
    """Test Serper tool configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SerperSearchToolConfig(
            api_key=SecretStr("test_key"),
        )

        assert config.max_results == 10
        assert config.num_per_page == 20
        assert config.score_cutoff == 0.0
        assert config.category == "scholar"
        assert config.gl == "us"
        assert config.hl == "en"
        assert config.autocorrect is True
        assert config.max_pages == 1
        assert config.pre_authenticate is False
        assert config.debug is False

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = SerperSearchToolConfig(
            api_key=SecretStr("custom_api_key"),
            base_url=HttpUrl("https://custom.serper.dev"),
            category="search",
            max_results=20,
            num_per_page=10,
            score_cutoff=0.5,
            gl="uk",
            hl="es",
            autocorrect=False,
            max_pages=10,
            pre_authenticate=True,
            debug=True,
        )

        assert config.api_key.get_secret_value() == "custom_api_key"
        assert str(config.base_url) == "https://custom.serper.dev/"
        assert config.category == "search"
        assert config.max_results == 20
        assert config.num_per_page == 10
        assert config.score_cutoff == 0.5
        assert config.gl == "uk"
        assert config.hl == "es"
        assert config.autocorrect is False
        assert config.max_pages == 10
        assert config.pre_authenticate is True
        assert config.debug is True

    def test_config_validation_max_pages(self):
        """Test max_pages validation."""
        # Valid values
        SerperSearchToolConfig(api_key=SecretStr("test"), max_pages=1)
        SerperSearchToolConfig(api_key=SecretStr("test"), max_pages=10)

        # Invalid values
        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), max_pages=0)

        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), max_pages=11)

    def test_config_validation_num_per_page(self):
        """Test num_per_page validation."""
        # Valid values
        SerperSearchToolConfig(api_key=SecretStr("test"), num_per_page=1)
        SerperSearchToolConfig(api_key=SecretStr("test"), num_per_page=100)

        # Invalid values
        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), num_per_page=0)

        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), num_per_page=101)

    def test_config_validation_score_cutoff(self):
        """Test score_cutoff validation."""
        # Valid values
        SerperSearchToolConfig(api_key=SecretStr("test"), score_cutoff=0.0)
        SerperSearchToolConfig(api_key=SecretStr("test"), score_cutoff=0.5)
        SerperSearchToolConfig(api_key=SecretStr("test"), score_cutoff=1.0)

        # Invalid values
        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), score_cutoff=-0.1)

        with pytest.raises(ValueError):
            SerperSearchToolConfig(api_key=SecretStr("test"), score_cutoff=1.1)


class TestSerperSearchToolSchemas:
    """Test Serper tool input/output schemas."""

    def test_input_schema_validation(self):
        """Test input schema validation."""
        # Valid input
        valid_input = SerperSearchToolInputSchema(
            queries=["quantum computing"],
            category="science",
            max_results=5,
        )
        assert valid_input.queries == ["quantum computing"]
        assert valid_input.category == "science"
        assert valid_input.max_results == 5

    def test_input_schema_defaults(self):
        """Test input schema default values."""
        input_schema = SerperSearchToolInputSchema(queries=["test"])

        assert input_schema.category == "science"
        assert input_schema.max_results is None

    def test_input_schema_multiple_queries(self):
        """Test input schema with multiple queries."""
        input_schema = SerperSearchToolInputSchema(
            queries=["query1", "query2", "query3"],
            category="general",
        )

        assert len(input_schema.queries) == 3
        assert input_schema.category == "general"

    def test_output_schema_validation(self):
        """Test output schema validation."""
        sample_results = [
            SearchResultItem(
                url="https://example.com/1",
                title="Test Result 1",
                content="Test content 1",
                query="test query",
                category="science",
                engine="serper",
            ),
            SearchResultItem(
                url="https://example.com/2",
                title="Test Result 2",
                content="Test content 2",
                query="test query",
                category="science",
                engine="serper",
            ),
        ]

        output = SerperSearchToolOutputSchema(
            results=sample_results,
            category="science",
            extra={"credits": 1, "relatedSearches": ["test1", "test2"]},
        )

        assert len(output.results) == 2
        assert output.category == "science"
        assert output.extra["credits"] == 1
        assert all(isinstance(item, SearchResultItem) for item in output.results)


class TestSerperSearchTool:
    """Test Serper search tool initialization and utility methods."""

    def test_tool_initialization_default(self):
        """Test tool initialization with default config."""
        config = SerperSearchToolConfig(api_key=SecretStr("test_key"))
        tool = SerperSearchTool(config=config)

        assert tool.config.max_results == 10
        assert tool.config.category == "scholar"
        assert tool.config.num_per_page == 20

    def test_tool_initialization_custom_config(self, sample_serper_config):
        """Test tool initialization with custom config."""
        tool = SerperSearchTool(config=sample_serper_config)

        assert tool.config.max_results == 10
        assert tool.config.category == "scholar"
        assert tool.config.num_per_page == 20

    def test_tool_requires_api_key(self):
        """Test that tool requires API key."""
        config = SerperSearchToolConfig(api_key=SecretStr(""))

        with pytest.raises(ValueError, match="SERPER_API_KEY"):
            SerperSearchTool(config=config)

    def test_headers_property(self, mock_serper_tool):
        """Test headers property returns correct format."""
        headers = mock_serper_tool.headers

        assert "X-API-KEY" in headers
        assert headers["X-API-KEY"] == "test_api_key_12345"
        assert headers["Content-Type"] == "application/json"

    def test_get_search_endpoint_science_category(self, mock_serper_tool):
        """Test endpoint mapping for science category."""
        endpoint = mock_serper_tool._get_search_endpoint("science")
        assert endpoint.endswith("scholar")

    def test_get_search_endpoint_general_category(self, mock_serper_tool):
        """Test endpoint mapping for general category."""
        endpoint = mock_serper_tool._get_search_endpoint("general")
        assert endpoint.endswith("search")

    def test_get_search_endpoint_technology_category(self, mock_serper_tool):
        """Test endpoint mapping for technology category."""
        endpoint = mock_serper_tool._get_search_endpoint("technology")
        assert endpoint.endswith("search")

    def test_get_search_endpoint_default_category(self, mock_serper_tool):
        """Test endpoint uses default category when none specified."""
        endpoint = mock_serper_tool._get_search_endpoint(None)
        # Default is scholar
        assert endpoint.endswith("scholar")

    @patch("httpx.post")
    def test_validate_connection_success(self, mock_post):
        """Test connection validation succeeds with valid API key."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = SerperSearchToolConfig(
            api_key=SecretStr("valid_key"),
            pre_authenticate=True,
        )

        # Should not raise
        tool = SerperSearchTool(config=config)
        assert tool is not None

    @patch("httpx.post")
    def test_validate_connection_unauthorized(self, mock_post):
        """Test connection validation fails with invalid API key."""
        # Mock unauthorized response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        config = SerperSearchToolConfig(
            api_key=SecretStr("invalid_key"),
            pre_authenticate=True,
        )

        with pytest.raises(RuntimeError, match="Invalid or unauthorized"):
            SerperSearchTool(config=config)

    @patch("httpx.post")
    def test_validate_connection_network_error(self, mock_post):
        """Test connection validation handles network errors."""
        # Mock network error
        mock_post.side_effect = httpx.RequestError("Network error")

        config = SerperSearchToolConfig(
            api_key=SecretStr("test_key"),
            pre_authenticate=True,
        )

        with pytest.raises(RuntimeError, match="Cannot connect to Serper API"):
            SerperSearchTool(config=config)


class TestCategoryMapping:
    """Test category to endpoint mapping."""

    def test_category_map_completeness(self):
        """Test that all input categories are mapped."""
        tool = SerperSearchTool(
            config=SerperSearchToolConfig(api_key=SecretStr("test")),
        )

        # All categories from SearchToolInputSchema should be mapped
        assert "science" in tool._category_map
        assert "general" in tool._category_map
        assert "technology" in tool._category_map

    def test_category_map_values(self):
        """Test category mappings are correct."""
        tool = SerperSearchTool(
            config=SerperSearchToolConfig(api_key=SecretStr("test")),
        )

        assert tool._category_map["science"] == "scholar"
        assert tool._category_map["general"] == "search"
        assert tool._category_map["technology"] == "search"

    def test_max_results_map_completeness(self):
        """Test that all Serper endpoints have max results defined."""
        tool = SerperSearchTool(
            config=SerperSearchToolConfig(api_key=SecretStr("test")),
        )

        assert "search" in tool._max_results_map
        assert "scholar" in tool._max_results_map
        assert "news" in tool._max_results_map
        assert "images" in tool._max_results_map
        assert "places" in tool._max_results_map

    def test_result_key_map_completeness(self):
        """Test that all Serper endpoints have result keys defined."""
        tool = SerperSearchTool(
            config=SerperSearchToolConfig(api_key=SecretStr("test")),
        )

        assert "search" in tool._result_key_map
        assert "scholar" in tool._result_key_map
        assert "news" in tool._result_key_map
        assert "images" in tool._result_key_map
        assert "places" in tool._result_key_map


class TestScoreCalculation:
    """Test score calculation methods."""

    def test_calculate_score_first_position(self, mock_serper_tool):
        """Test score for first position result."""
        result = {"title": "Test", "snippet": "Test snippet"}
        score = mock_serper_tool._calculate_score(result, position=0)

        assert score > 0.9  # First result should have high score

    def test_calculate_score_later_position(self, mock_serper_tool):
        """Test score decreases with position."""
        result = {"title": "Test"}

        score_first = mock_serper_tool._calculate_score(result, position=0)
        score_tenth = mock_serper_tool._calculate_score(result, position=9)

        assert score_first > score_tenth

    def test_calculate_score_with_metadata_bonus(self, mock_serper_tool):
        """Test score bonus for rich metadata."""
        result_basic = {"title": "Test"}
        result_rich = {
            "title": "Test",
            "snippet": "Has snippet",
            "sitelinks": ["link1"],
            "date": "2023-01-01",
        }

        # Use later position to avoid hitting max score of 1.0
        score_basic = mock_serper_tool._calculate_score(result_basic, position=5)
        score_rich = mock_serper_tool._calculate_score(result_rich, position=5)

        assert score_rich > score_basic

    def test_calculate_score_max_value(self, mock_serper_tool):
        """Test score is capped at 1.0."""
        result = {
            "title": "Test",
            "snippet": "Test",
            "sitelinks": ["link"],
            "date": "2023",
        }

        score = mock_serper_tool._calculate_score(result, position=0)
        assert score <= 1.0


class TestResultProcessing:
    """Test result processing methods."""

    def test_process_results_score_filtering(self, sample_search_result_items):
        """Test that results are filtered by score cutoff."""
        config = SerperSearchToolConfig(
            api_key=SecretStr("test"),
            score_cutoff=0.5,
        )
        tool = SerperSearchTool(config=config)

        processed = tool._process_results(sample_search_result_items)

        # Should filter out low score results (< 0.5)
        assert all((r.score or 0) >= 0.5 for r in processed)

    def test_process_results_deduplication(self, mock_serper_tool, sample_search_result_items):
        """Test URL deduplication."""
        processed = mock_serper_tool._process_results(sample_search_result_items)

        # Should keep only unique URLs
        urls = [str(r.url) for r in processed]
        assert len(urls) == len(set(urls)), "Should have no duplicate URLs"

    def test_process_results_sorting_by_score(self, mock_serper_tool, sample_search_result_items):
        """Test results are sorted by score in descending order."""
        processed = mock_serper_tool._process_results(sample_search_result_items)

        scores = [r.score or 0 for r in processed]
        assert scores == sorted(scores, reverse=True)

    def test_process_results_adds_missing_scores(self, mock_serper_tool):
        """Test that results without scores are still processed."""
        mock_results = [
            SearchResultItem(
                title="No Score",
                url="http://test.com",
                content="test",
                query="test",
                score=None,  # No score
            ),
        ]

        processed = mock_serper_tool._process_results(mock_results)

        assert len(processed) == 1
        # Score can be None, that's okay
        assert processed[0].score is None or processed[0].score >= 0


class TestMetadataMerging:
    """Test metadata merging across pages."""

    def test_merge_metadata_empty_list(self, mock_serper_tool):
        """Test merging empty metadata list."""
        merged = mock_serper_tool._merge_metadata([])
        assert merged == {}

    def test_merge_metadata_single_page(self, mock_serper_tool):
        """Test merging metadata from single page."""
        metadata = [{"credits": 1, "searchParameters": {"q": "test"}}]

        merged = mock_serper_tool._merge_metadata(metadata)

        assert merged["credits"] == 1
        assert merged["searchParameters"]["q"] == "test"

    def test_merge_metadata_sum_credits(self, mock_serper_tool):
        """Test credits are summed across pages."""
        metadata = [
            {"credits": 1, "searchParameters": {"q": "test"}},
            {"credits": 1, "searchParameters": {"q": "test"}},
            {"credits": 1, "searchParameters": {"q": "test"}},
        ]

        merged = mock_serper_tool._merge_metadata(metadata)

        assert merged["credits"] == 3

    def test_merge_metadata_extend_lists(self, mock_serper_tool):
        """Test list fields are extended/aggregated."""
        metadata = [
            {"relatedSearches": ["a", "b"], "credits": 1},
            {"relatedSearches": ["c", "d"], "credits": 1},
        ]

        merged = mock_serper_tool._merge_metadata(metadata)

        assert merged["relatedSearches"] == ["a", "b", "c", "d"]
        assert merged["credits"] == 2

    def test_merge_metadata_keep_first_non_list(self, mock_serper_tool):
        """Test non-list fields keep first occurrence."""
        metadata = [
            {"searchParameters": {"q": "first"}, "credits": 1},
            {"searchParameters": {"q": "second"}, "credits": 1},
        ]

        merged = mock_serper_tool._merge_metadata(metadata)

        # Should keep first occurrence of searchParameters
        assert merged["searchParameters"]["q"] == "first"
        assert merged["credits"] == 2


class TestPagination:
    """Test pagination functionality."""

    @pytest.mark.asyncio
    async def test_fetch_single_page(
        self,
        mock_serper_tool,
        mock_serper_scholar_response,
    ):
        """Test fetching single page of results."""
        async with httpx.AsyncClient() as client:
            with patch.object(client, "post") as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.json = MagicMock(return_value=mock_serper_scholar_response)
                mock_post.return_value = mock_response

                results, metadata = await mock_serper_tool._fetch_serper_results(
                    client,
                    "test query",
                    "science",
                    page=1,
                )

                assert len(results) == 3
                assert metadata["credits"] == 1
                assert all(r.query == "test query" for r in results)

    @pytest.mark.asyncio
    async def test_fetch_paginated_results_multiple_pages(
        self,
        mock_serper_tool,
        mock_serper_paginated_response_page1,
        mock_serper_paginated_response_page2,
    ):
        """Test fetching multiple pages of results."""
        async with httpx.AsyncClient() as client:
            with patch.object(client, "post") as mock_post:
                # Mock responses for page 1 and page 2
                mock_response_1 = AsyncMock()
                mock_response_1.status_code = 200
                mock_response_1.json = MagicMock(return_value=mock_serper_paginated_response_page1)

                mock_response_2 = AsyncMock()
                mock_response_2.status_code = 200
                mock_response_2.json = MagicMock(return_value=mock_serper_paginated_response_page2)

                mock_post.side_effect = [mock_response_1, mock_response_2]

                results, metadata = await mock_serper_tool._fetch_serper_results_paginated(
                    client,
                    "deep learning",
                    "science",
                    target_results=15,
                )

                # Should fetch 2 pages to get 15+ results
                assert len(results) >= 15
                assert metadata["credits"] == 2  # Credits summed from both pages

    @pytest.mark.asyncio
    async def test_fetch_paginated_respects_max_pages(self, mock_serper_tool):
        """Test pagination respects max_pages limit."""
        async with httpx.AsyncClient() as client:
            with patch.object(client, "post") as mock_post:
                # Create mock response with 10 results per page
                def create_response(page_num):
                    response = AsyncMock()
                    response.status_code = 200
                    response.json = MagicMock(
                        return_value={
                            "organic": [
                                {
                                    "title": f"Result {i}",
                                    "link": f"http://test{i}.com",
                                    "snippet": "test",
                                    "position": i,
                                }
                                for i in range((page_num - 1) * 10 + 1, page_num * 10 + 1)
                            ],
                            "credits": 1,
                        },
                    )
                    return response

                mock_post.side_effect = [create_response(i) for i in range(1, 11)]

                results, _ = await mock_serper_tool._fetch_serper_results_paginated(
                    client,
                    "test",
                    "science",
                    target_results=100,
                )

                # Should stop at max_pages (5) even though target is higher
                assert mock_post.call_count <= mock_serper_tool.max_pages

    @pytest.mark.asyncio
    async def test_fetch_paginated_stops_on_empty_results(self, mock_serper_tool):
        """Test pagination stops when no more results available."""
        async with httpx.AsyncClient() as client:
            with patch.object(client, "post") as mock_post:
                # First page has results, second page is empty
                response_1 = AsyncMock()
                response_1.status_code = 200
                response_1.json = MagicMock(
                    return_value={
                        "organic": [
                            {"title": "Result 1", "link": "http://test1.com", "snippet": "test"},
                        ],
                        "credits": 1,
                    },
                )

                response_2 = AsyncMock()
                response_2.status_code = 200
                response_2.json = MagicMock(return_value={"organic": [], "credits": 1})

                mock_post.side_effect = [response_1, response_2]

                results, _ = await mock_serper_tool._fetch_serper_results_paginated(
                    client,
                    "rare query",
                    "science",
                    target_results=20,
                )

                # Should stop after page 2 (empty results)
                assert mock_post.call_count == 2
                assert len(results) == 1


class TestEndToEndBlackbox:
    """End-to-end blackbox tests for science and general categories."""

    @pytest.mark.asyncio
    async def test_science_category_full_flow(
        self,
        mock_serper_tool,
        mock_serper_scholar_response,
    ):
        """Test full flow for science category (scholar endpoint)."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_scholar_response)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["quantum computing"],
                category="science",
                max_results=3,
            )

            result = await mock_serper_tool._arun(input_schema)

            # Verify output schema
            assert isinstance(result, SerperSearchToolOutputSchema)
            assert len(result.results) <= 3
            assert result.category == "science"

            # Verify search results
            for item in result.results:
                assert isinstance(item, SearchResultItem)
                assert item.engine == "serper"
                assert item.url is not None
                assert item.title is not None

            # Verify metadata
            assert "credits" in result.extra
            assert result.extra["credits"] >= 1

    @pytest.mark.asyncio
    async def test_general_category_full_flow(
        self,
        mock_serper_tool,
        mock_serper_general_response,
    ):
        """Test full flow for general category (search endpoint)."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_general_response)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["machine learning frameworks"],
                category="general",
                max_results=3,
            )

            result = await mock_serper_tool._arun(input_schema)

            # Verify output schema
            assert isinstance(result, SerperSearchToolOutputSchema)
            assert len(result.results) <= 3
            assert result.category == "general"

            # Verify search results
            for item in result.results:
                assert isinstance(item, SearchResultItem)
                assert item.engine == "serper"
                assert item.url is not None

            # Verify metadata includes general search specific fields
            assert "credits" in result.extra
            if "topStories" in mock_serper_general_response:
                assert "topStories" in result.extra

    @pytest.mark.asyncio
    async def test_multiple_queries(
        self,
        mock_serper_tool,
        mock_serper_scholar_response,
    ):
        """Test handling multiple queries in single request."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_scholar_response)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["query1", "query2", "query3"],
                category="science",
                max_results=10,
            )

            result = await mock_serper_tool._arun(input_schema)

            # Should make at least 3 API calls (one per query, possibly more for pagination)
            assert mock_client.post.call_count >= 3
            assert isinstance(result, SerperSearchToolOutputSchema)
            assert len(result.results) <= 10

    @pytest.mark.asyncio
    async def test_empty_results_handling(
        self,
        mock_serper_tool,
        mock_serper_empty_response,
    ):
        """Test handling of empty search results."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_serper_empty_response)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["extremely rare query"],
                category="science",
                max_results=10,
            )

            result = await mock_serper_tool._arun(input_schema)

            assert isinstance(result, SerperSearchToolOutputSchema)
            assert len(result.results) == 0
            # Credits may or may not be present depending on if results were found
            assert result.extra is not None

    @pytest.mark.asyncio
    async def test_http_error_handling(self, mock_serper_tool):
        """Test handling of HTTP errors - tool handles errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.reason_phrase = "Internal Server Error"
            mock_response.text = "Server error"
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["test query"],
                category="science",
                max_results=5,
            )

            # The tool handles HTTP errors gracefully and returns empty results
            # rather than raising exceptions
            result = await mock_serper_tool._arun(input_schema)

            assert isinstance(result, SerperSearchToolOutputSchema)
            # Should return empty results when API errors occur
            assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_pdf_url_extraction(self, mock_serper_tool):
        """Test extraction of PDF URLs from scholar results."""
        mock_response_data = {
            "organic": [
                {
                    "title": "Paper with PDF",
                    "link": "https://arxiv.org/abs/2101.12345",
                    "snippet": "Test paper",
                    "pdfUrl": "https://arxiv.org/pdf/2101.12345.pdf",
                    "position": 1,
                },
            ],
            "credits": 1,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["test"],
                category="science",
                max_results=5,
            )

            result = await mock_serper_tool._arun(input_schema)

            assert len(result.results) == 1
            # pdf_url is an AnyUrl type, convert to string for comparison
            assert str(result.results[0].pdf_url) == "https://arxiv.org/pdf/2101.12345.pdf"

    @pytest.mark.asyncio
    async def test_published_date_extraction(self, mock_serper_tool):
        """Test extraction of published dates from results."""
        mock_response_data = {
            "organic": [
                {
                    "title": "Recent Paper",
                    "link": "https://example.com/paper",
                    "snippet": "Test paper",
                    "date": "2023-05-15",
                    "position": 1,
                },
            ],
            "credits": 1,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_client.post.return_value = mock_response

            input_schema = SerperSearchToolInputSchema(
                queries=["test"],
                category="science",
                max_results=5,
            )

            result = await mock_serper_tool._arun(input_schema)

            assert len(result.results) == 1
            assert result.results[0].published_date == "2023-05-15"
