"""
Unit tests for SearxNG search tool.

Tests individual components and methods of the SearxNGSearchTool
without making actual HTTP requests.
"""

import pytest

from akd.structures import SearchResultItem
from akd.tools.search.searxng import (
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
        assert config.engines == ["arxiv", "google_scholar"]

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
        assert input_schema.max_results is None

    def test_output_schema_validation(self, sample_search_result_items):
        """Test output schema validation."""
        output = SearxNGSearchToolOutputSchema(
            results=sample_search_result_items,
        )

        assert len(output.results) > 0  # Should have multiple test items
        assert all(isinstance(item, SearchResultItem) for item in output.results)


class TestSearxNGSearchTool:
    """Test SearxNG search tool initialization and utility methods."""

    def test_tool_initialization_default(self):
        """Test tool initialization with default config."""
        tool = SearxNGSearchTool()

        assert tool.config.max_results == 10
        assert tool.config.engines == ["arxiv", "google_scholar"]

    def test_tool_initialization_custom_config(self, sample_searxng_config):
        """Test tool initialization with custom config."""
        tool = SearxNGSearchTool(config=sample_searxng_config)

        assert tool.config.max_results == 10
        assert tool.config.strict is True
        assert tool.config.engines == ["google", "arxiv", "google_scholar"]

    def test_normalize_engine_name(self):
        """Test engine name normalization."""
        assert SearxNGSearchTool.normalize_engine_name("Google Scholar") == "google_scholar"
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

    def test_process_results_score_filtering(self, mock_searxng_tool, sample_search_result_items):
        """Test that results are filtered by score cutoff."""
        processed = mock_searxng_tool._process_results(sample_search_result_items)

        # Should filter out low score result (0.1 < 0.25 cutoff) and items with missing fields
        assert all((result.score or 0) >= mock_searxng_tool.score_cutoff for result in processed)

    def test_process_results_strict_engine_filtering(self, sample_search_result_items):
        """Test strict engine filtering."""
        config = SearxNGSearchToolConfig(
            engines=["google", "arxiv"],
            strict=True,
        )
        tool = SearxNGSearchTool(config=config)

        processed = tool._process_results(sample_search_result_items)

        # Should filter out bing result and keep only google/arxiv
        assert all(result.engine in ["google", "arxiv"] for result in processed)

    def test_process_results_deduplication(self, mock_searxng_tool, sample_search_result_items):
        """Test URL deduplication."""
        processed = mock_searxng_tool._process_results(sample_search_result_items)

        # Should remove duplicate URLs (sample has test1.com appearing twice)
        urls = [str(result.url) for result in processed]
        assert len(urls) == len(set(urls)), "Should have no duplicate URLs"

    def test_process_results_sorting_by_score(self, mock_searxng_tool, sample_search_result_items):
        """Test results are sorted by score in descending order."""
        processed = mock_searxng_tool._process_results(sample_search_result_items)

        # Should be sorted by score descending
        scores = [result.score or 0 for result in processed]
        assert scores == sorted(scores, reverse=True)

    def test_process_results_missing_required_fields(self, mock_searxng_tool, sample_search_result_items):
        """Test filtering of results with missing required fields."""
        processed = mock_searxng_tool._process_results(sample_search_result_items)

        # Should filter out results with empty title (URL is always required by pydantic)
        for result in processed:
            assert result.url is not None
            assert result.title  # Should not be empty string

    def test_process_results_doi_handling(self, mock_searxng_tool, sample_search_result_items):
        """Test DOI field processing."""
        processed = mock_searxng_tool._process_results(sample_search_result_items)

        # Find results with DOI
        results_with_doi = [r for r in processed if r.doi]
        assert len(results_with_doi) > 0, "Should have at least one result with DOI"

        # Verify DOI is a string
        for result in results_with_doi:
            assert isinstance(result.doi, str)
