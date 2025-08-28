"""
Unit tests for SearxNG search tool.

Tests individual components and methods of the SearxNGSearchTool
without making actual HTTP requests.
"""

import pytest

from akd.structures import SearchResultItem
from akd.tools.search.searxng_search import (
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
    SearxNGSearchToolInputSchema,
    SearxNGSearchToolOutputSchema,
)


class TestSearxNGSearchToolConfig:
    """Test SearxNG tool configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SearxNGSearchToolConfig()

        assert config.max_results == 10
        assert config.max_pages == 25
        assert config.results_per_page == 10
        assert config.score_cutoff == 0.25
        assert config.strict is False
        assert config.debug is False
        assert config.engines == ["google", "arxiv", "google_scholar"]

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = SearxNGSearchToolConfig(
            base_url="http://custom.searxng.com",
            max_results=20,
            engines=["duckduckgo", "bing"],
            max_pages=10,
            results_per_page=5,
            score_cutoff=0.5,
            strict=True,
            debug=True,
        )

        # HttpUrl automatically adds trailing slash
        assert str(config.base_url) == "http://custom.searxng.com/"
        assert config.max_results == 20
        assert config.engines == ["duckduckgo", "bing"]
        assert config.max_pages == 10
        assert config.results_per_page == 5
        assert config.score_cutoff == 0.5
        assert config.strict is True
        assert config.debug is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Test max_pages validation
        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(max_pages=0)

        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(max_pages=101)

        # Test results_per_page validation
        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(results_per_page=0)

        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(results_per_page=101)

        # Test score_cutoff validation
        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(score_cutoff=-0.1)

        with pytest.raises(ValueError):
            SearxNGSearchToolConfig(score_cutoff=1.1)

    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        # Test manual instantiation with values (since env vars are resolved at import)
        config = SearxNGSearchToolConfig(
            base_url="http://env.searxng.com",
            max_results=15,
            engines=["duckduckgo", "startpage"],
            max_pages=30,
            results_per_page=8,
            score_cutoff=0.3,
        )

        # HttpUrl automatically adds trailing slash
        assert str(config.base_url) == "http://env.searxng.com/"
        assert config.max_results == 15
        assert config.engines == ["duckduckgo", "startpage"]
        assert config.max_pages == 30
        assert config.results_per_page == 8
        assert config.score_cutoff == 0.3


class TestSearxNGSearchToolSchemas:
    """Test SearxNG tool input/output schemas."""

    def test_input_schema_validation(self):
        """Test input schema validation."""
        # Valid input
        valid_input = SearxNGSearchToolInputSchema(
            queries=["test query"],
            category="science",
            max_results=5,
        )
        assert valid_input.queries == ["test query"]
        assert valid_input.category == "science"
        assert valid_input.max_results == 5

    def test_input_schema_defaults(self):
        """Test input schema default values."""
        input_schema = SearxNGSearchToolInputSchema(queries=["test"])

        assert input_schema.category == "science"
        assert input_schema.max_results == 10

    def test_output_schema_validation(self, sample_search_result_items):
        """Test output schema validation."""
        output = SearxNGSearchToolOutputSchema(
            results=sample_search_result_items,
            category="science",
        )

        assert len(output.results) == 2
        assert output.category == "science"
        assert all(isinstance(item, SearchResultItem) for item in output.results)


class TestSearxNGSearchTool:
    """Test SearxNG search tool initialization and utility methods."""

    def test_tool_initialization_default(self):
        """Test tool initialization with default config."""
        tool = SearxNGSearchTool()

        assert tool.config.max_results == 10
        assert tool.config.base_url == "http://localhost:8080"
        assert tool.config.engines == ["google", "arxiv", "google_scholar"]

    def test_tool_initialization_custom_config(self, sample_searxng_config):
        """Test tool initialization with custom config."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        assert tool.config.max_results == 10
        assert tool.config.strict is True
        assert tool.config.engines == ["google", "arxiv", "google_scholar"]

    def test_from_params_class_method(self):
        """Test tool creation using from_params class method."""
        tool = SearxNGSearchTool.from_params(
            base_url="http://test.searxng.com",
            max_results=20,
            engines=["duckduckgo"],
            strict=False,
            debug=True,
        )

        # HttpUrl automatically adds trailing slash
        assert str(tool.config.base_url) == "http://test.searxng.com/"
        assert tool.config.max_results == 20
        assert tool.config.engines == ["duckduckgo"]
        assert tool.config.strict is False
        assert tool.config.debug is True

    def test_normalize_engine_name(self):
        """Test engine name normalization."""
        assert (
            SearxNGSearchTool.normalize_engine_name("Google Scholar")
            == "google_scholar"
        )
        assert SearxNGSearchTool.normalize_engine_name("DuckDuckGo") == "duckduckgo"
        assert SearxNGSearchTool.normalize_engine_name("google") == "google"
        assert SearxNGSearchTool.normalize_engine_name("ARXIV") == "arxiv"

    def test_engine_names_match(self):
        """Test engine name matching."""
        assert SearxNGSearchTool.engine_names_match("google", "Google")
        assert SearxNGSearchTool.engine_names_match("Google Scholar", "google_scholar")
        assert SearxNGSearchTool.engine_names_match("duckduckgo", "DuckDuckGo")
        assert not SearxNGSearchTool.engine_names_match("google", "bing")


class TestResultProcessing:
    """Test result processing methods."""

    @pytest.mark.asyncio
    async def test_process_results_score_filtering(self, mock_searxng_tool):
        """Test that results are filtered by score cutoff."""
        # Mock results with different scores and engines that match the config
        mock_results = [
            {
                "title": "High Score",
                "content": "test",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "Medium Score",
                "content": "test",
                "url": "http://test2.com",
                "score": 0.3,
                "engine": "arxiv",
            },
            {
                "title": "Low Score",
                "content": "test",
                "url": "http://test3.com",
                "score": 0.1,
                "engine": "google",
            },  # Below cutoff
        ]

        processed = await mock_searxng_tool._process_results(mock_results)

        # Should filter out low score result (0.1 < 0.25 cutoff)
        assert len(processed) == 2
        assert all(
            result.get("score", 0) >= mock_searxng_tool.config.score_cutoff
            for result in processed
        )

    @pytest.mark.asyncio
    async def test_process_results_strict_engine_filtering(self):
        """Test strict engine filtering."""
        config = SearxNGSearchToolConfig(
            engines=["google", "arxiv"],
            strict=True,
        )
        tool = SearxNGSearchTool(config=config)

        mock_results = [
            {
                "title": "Google Result",
                "content": "test",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "ArXiv Result",
                "content": "test",
                "url": "http://test2.com",
                "score": 0.8,
                "engine": "arxiv",
            },
            {
                "title": "Bing Result",
                "content": "test",
                "url": "http://test3.com",
                "score": 0.7,
                "engine": "bing",
            },  # Should be filtered
        ]

        processed = await tool._process_results(mock_results)

        # Should filter out bing result
        assert len(processed) == 2
        assert all(result.get("engine") in ["google", "arxiv"] for result in processed)

    @pytest.mark.asyncio
    async def test_process_results_deduplication(self, mock_searxng_tool):
        """Test URL deduplication."""
        mock_results = [
            {
                "title": "First",
                "content": "test",
                "url": "http://duplicate.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "title": "Second",
                "content": "test",
                "url": "http://unique.com",
                "score": 0.8,
                "engine": "arxiv",
            },
            {
                "title": "Third",
                "content": "test",
                "url": "http://duplicate.com",
                "score": 0.7,
                "engine": "google_scholar",
            },  # Duplicate URL
        ]

        processed = await mock_searxng_tool._process_results(mock_results)

        # Should remove duplicate URL, keeping the higher scored one
        assert len(processed) == 2
        urls = [result["url"] for result in processed]
        assert "http://duplicate.com" in urls
        assert "http://unique.com" in urls
        assert urls.count("http://duplicate.com") == 1

    @pytest.mark.asyncio
    async def test_process_results_sorting_by_score(self, mock_searxng_tool):
        """Test results are sorted by score in descending order."""
        mock_results = [
            {
                "title": "Medium",
                "content": "test",
                "url": "http://test1.com",
                "score": 0.5,
                "engine": "google",
            },
            {
                "title": "High",
                "content": "test",
                "url": "http://test2.com",
                "score": 0.9,
                "engine": "arxiv",
            },
            {
                "title": "Low",
                "content": "test",
                "url": "http://test3.com",
                "score": 0.3,
                "engine": "google_scholar",
            },
        ]

        processed = await mock_searxng_tool._process_results(mock_results)

        # Should be sorted by score descending
        scores = [result.get("score", 0) for result in processed]
        assert scores == sorted(scores, reverse=True)
        assert processed[0]["title"] == "High"
        assert processed[-1]["title"] == "Low"

    @pytest.mark.asyncio
    async def test_process_results_missing_required_fields(self, mock_searxng_tool):
        """Test filtering of results with missing required fields."""
        mock_results = [
            {
                "title": "Valid",
                "content": "test",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
            },
            {
                "content": "test",
                "url": "http://test2.com",
                "score": 0.8,
                "engine": "arxiv",
            },  # Missing title
            {
                "title": "No Content",
                "url": "http://test3.com",
                "score": 0.7,
                "engine": "google",
            },  # Missing content
            {
                "title": "No URL",
                "content": "test",
                "score": 0.6,
                "engine": "arxiv",
            },  # Missing URL
        ]

        processed = await mock_searxng_tool._process_results(mock_results)

        # Should only include results with all required fields
        assert len(processed) == 1
        assert processed[0]["title"] == "Valid"

    @pytest.mark.asyncio
    async def test_process_results_doi_handling(self, mock_searxng_tool):
        """Test DOI field processing."""
        mock_results = [
            {
                "title": "Paper with DOI list",
                "content": "test",
                "url": "http://test1.com",
                "score": 0.9,
                "engine": "google",
                "doi": ["10.1000/test123", "10.1000/alt456"],  # DOI as list
            },
            {
                "title": "Paper with DOI string",
                "content": "test",
                "url": "http://test2.com",
                "score": 0.8,
                "engine": "arxiv",
                "doi": "10.1000/single789",  # DOI as string
            },
            {
                "title": "Paper with numeric DOI",
                "content": "test",
                "url": "http://test3.com",
                "score": 0.7,
                "engine": "google_scholar",
                "doi": 123456789,  # DOI as number
            },
        ]

        processed = await mock_searxng_tool._process_results(mock_results)

        assert len(processed) == 3
        # Results are sorted by score, so order is: 0.9, 0.8, 0.7
        # Should take first DOI from list
        assert processed[0]["doi"] == "10.1000/test123"
        # Should keep string DOI as-is
        assert processed[1]["doi"] == "10.1000/single789"
        # Should convert numeric DOI to string
        assert processed[2]["doi"] == "123456789"
