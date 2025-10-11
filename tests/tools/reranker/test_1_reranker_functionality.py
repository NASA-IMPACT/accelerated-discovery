"""
Blackbox tests for reranker tools.

These tests verify the external behavior of reranker tools without examining
internal implementation details. Tests are parameterized to run against all
reranker implementations.
"""

from typing import List

import pytest
from pydantic import ValidationError

from akd.structures import SearchResultItem
from akd.tools.reranker import (
    RerankerToolConfig,
    RerankerToolInputSchema,
    RerankerToolOutputSchema,
)


class TestRerankerToolConfig:
    """Test reranker tool configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = RerankerToolConfig()

        assert config.deduplication is True
        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L12-v2"
        assert config.deduplication_keys == ["url"]
        assert config.sort_key == "score"

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = RerankerToolConfig(
            deduplication=False,
            model_name="custom-model",
            deduplication_keys=["url", "title"],
            sort_key="custom_score",
        )

        assert config.deduplication is False
        assert config.model_name == "custom-model"
        assert config.deduplication_keys == ["url", "title"]
        assert config.sort_key == "custom_score"


class TestRerankerToolSchemas:
    """Test reranker tool input/output schemas."""

    def test_input_schema_valid(self, sample_search_results: List[SearchResultItem], test_query: str):
        """Test valid input schema creation."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        assert input_schema.query == test_query
        assert len(input_schema.results) == 5
        assert all(isinstance(r, SearchResultItem) for r in input_schema.results)

    def test_input_schema_missing_query(self, sample_search_results: List[SearchResultItem]):
        """Test input schema validation fails with missing query."""
        with pytest.raises(ValidationError):
            RerankerToolInputSchema(results=sample_search_results)

    def test_input_schema_missing_results(self, test_query: str):
        """Test input schema validation fails with missing results."""
        with pytest.raises(ValidationError):
            RerankerToolInputSchema(query=test_query)

    def test_output_schema_valid(self, sample_search_results: List[SearchResultItem], test_query: str):
        """Test valid output schema creation."""
        output_schema = RerankerToolOutputSchema(
            query=test_query,
            results=sample_search_results,
        )

        assert output_schema.query == test_query
        assert len(output_schema.results) == 5
        assert all(isinstance(r, SearchResultItem) for r in output_schema.results)


class TestRerankerToolCoreFunctionality:
    """Test core reranking functionality across all implementations."""

    @pytest.mark.asyncio
    async def test_reranking_correctness(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """
        Test that reranking produces correct result ordering.

        The 3rd item should rank first for query "this is a dummy query"
        because its content is "this is a dummy query content".
        """
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        assert isinstance(output, RerankerToolOutputSchema)
        assert output.query == test_query
        assert len(output.results) > 0

        # The 3rd item (index 2) should now be first after reranking
        top_result = output.results[0]
        assert str(top_result.url) == "https://example.com/article3"
        assert top_result.content == "this is a dummy query content"

    @pytest.mark.asyncio
    async def test_all_results_have_scores(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that all reranked results have scores assigned."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        for result in output.results:
            assert result.score is not None
            # Score should be a numeric type (including numpy types)
            assert hasattr(result.score, "__float__"), f"Score {result.score} is not numeric"
            # Scores should be in reasonable range (0-1 for normalized scores)
            assert float(result.score) >= 0

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that results are sorted by score in descending order."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        scores = [r.score for r in output.results]

        # Verify descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Scores not in descending order: {scores}"

    @pytest.mark.asyncio
    async def test_all_input_items_present_in_output(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that all input items are present in output (possibly deduplicated)."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        # With deduplication enabled (default), we should have at most as many results
        assert len(output.results) <= len(sample_search_results)
        assert len(output.results) > 0

        # All output URLs should be from input
        input_urls = {str(r.url) for r in sample_search_results}
        output_urls = {str(r.url) for r in output.results}
        assert output_urls.issubset(input_urls)

    @pytest.mark.asyncio
    async def test_query_preserved_in_output(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that the query is preserved in the output."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        assert output.query == test_query


class TestRerankerToolDeduplication:
    """Test deduplication functionality."""

    @pytest.mark.asyncio
    async def test_deduplication_enabled_by_default(
        self,
        reranker_tool_class,
        test_query: str,
    ):
        """Test that deduplication is enabled by default."""
        # Create results with duplicate URLs
        duplicate_results = [
            SearchResultItem(
                url="https://example.com/same",
                title="First",
                query=test_query,
                content="First content about the query",
            ),
            SearchResultItem(
                url="https://example.com/same",
                title="Second",
                query=test_query,
                content="Second content about the query",
            ),
            SearchResultItem(
                url="https://example.com/different",
                title="Third",
                query=test_query,
                content="Third content about the query",
            ),
        ]

        reranker = reranker_tool_class()
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=duplicate_results,
        )

        output = await reranker.arun(input_schema)

        # Should have deduplicated to 2 unique URLs
        assert len(output.results) == 2
        urls = [str(r.url) for r in output.results]
        assert len(set(urls)) == 2

    @pytest.mark.asyncio
    async def test_deduplication_disabled(
        self,
        reranker_tool_class,
        test_query: str,
    ):
        """Test reranking with deduplication disabled."""
        # Create results with duplicate URLs
        duplicate_results = [
            SearchResultItem(
                url="https://example.com/same",
                title="First",
                query=test_query,
                content="First content about the query",
            ),
            SearchResultItem(
                url="https://example.com/same",
                title="Second",
                query=test_query,
                content="Second content about the query",
            ),
            SearchResultItem(
                url="https://example.com/different",
                title="Third",
                query=test_query,
                content="Third content about the query",
            ),
        ]

        config = RerankerToolConfig(deduplication=False)
        reranker = reranker_tool_class(config=config)
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=duplicate_results,
        )

        output = await reranker.arun(input_schema)

        # Should have all 3 results (no deduplication)
        assert len(output.results) == 3

    @pytest.mark.asyncio
    async def test_custom_deduplication_keys(
        self,
        reranker_tool_class,
        test_query: str,
    ):
        """Test deduplication with custom keys."""
        # Create results with duplicate titles but different URLs
        results_with_duplicate_titles = [
            SearchResultItem(
                url="https://example.com/url1",
                title="Same Title",
                query=test_query,
                content="First content about the query",
            ),
            SearchResultItem(
                url="https://example.com/url2",
                title="Same Title",
                query=test_query,
                content="Second content about the query",
            ),
            SearchResultItem(
                url="https://example.com/url3",
                title="Different Title",
                query=test_query,
                content="Third content about the query",
            ),
        ]

        config = RerankerToolConfig(
            deduplication=True,
            deduplication_keys=["title"],
        )
        reranker = reranker_tool_class(config=config)
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=results_with_duplicate_titles,
        )

        output = await reranker.arun(input_schema)

        # Should deduplicate based on title to 2 unique results
        assert len(output.results) == 2
        titles = [r.title for r in output.results]
        assert len(set(titles)) == 2


class TestRerankerToolEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_results_list(
        self,
        reranker_instance,
        test_query: str,
    ):
        """Test reranking with empty results list."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=[],
        )

        output = await reranker_instance.arun(input_schema)

        assert isinstance(output, RerankerToolOutputSchema)
        assert output.query == test_query
        assert len(output.results) == 0

    @pytest.mark.asyncio
    async def test_single_result(
        self,
        reranker_instance,
        test_query: str,
    ):
        """Test reranking with a single result."""
        single_result = [
            SearchResultItem(
                url="https://example.com/single",
                title="Single Result",
                query=test_query,
                content="this is a dummy query content",
            ),
        ]

        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=single_result,
        )

        output = await reranker_instance.arun(input_schema)

        assert len(output.results) == 1
        assert str(output.results[0].url) == "https://example.com/single"
        assert output.results[0].score is not None

    @pytest.mark.asyncio
    async def test_all_identical_content(
        self,
        reranker_instance,
        test_query: str,
    ):
        """Test reranking when all results have identical content."""
        identical_results = [
            SearchResultItem(
                url=f"https://example.com/item{i}",
                title=f"Item {i}",
                query=test_query,
                content="identical content",
            )
            for i in range(5)
        ]

        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=identical_results,
        )

        output = await reranker_instance.arun(input_schema)

        assert len(output.results) == 5
        # All should have scores (possibly very similar)
        assert all(r.score is not None for r in output.results)

    @pytest.mark.asyncio
    async def test_missing_content_fields(
        self,
        reranker_instance,
        test_query: str,
    ):
        """Test reranking with results that have empty content."""
        results_with_empty_content = [
            SearchResultItem(
                url="https://example.com/empty1",
                title="Empty Content 1",
                query=test_query,
                content="",
            ),
            SearchResultItem(
                url="https://example.com/full",
                title="Full Content",
                query=test_query,
                content="this is a dummy query content",
            ),
            SearchResultItem(
                url="https://example.com/empty2",
                title="Empty Content 2",
                query=test_query,
                content="",
            ),
        ]

        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=results_with_empty_content,
        )

        output = await reranker_instance.arun(input_schema)

        assert len(output.results) == 3
        # All results should have scores even with empty content
        assert all(r.score is not None for r in output.results)
        # Result with matching content should rank higher
        assert str(output.results[0].url) == "https://example.com/full"

    @pytest.mark.asyncio
    async def test_very_long_content(
        self,
        reranker_instance,
        test_query: str,
    ):
        """Test reranking with very long content."""
        long_content = "this is a dummy query. " + "padding content. " * 1000

        results_with_long_content = [
            SearchResultItem(
                url="https://example.com/short",
                title="Short Content",
                query=test_query,
                content="unrelated content",
            ),
            SearchResultItem(
                url="https://example.com/long",
                title="Long Content",
                query=test_query,
                content=long_content,
            ),
        ]

        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=results_with_long_content,
        )

        output = await reranker_instance.arun(input_schema)

        assert len(output.results) == 2
        # Should handle long content without errors
        assert all(r.score is not None for r in output.results)


class TestRerankerToolMultipleQueries:
    """Test reranking behavior with different query types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,expected_top_url",
        [
            ("machine learning", "https://example.com/article1"),
            ("neural network", "https://example.com/article2"),
            ("this is a dummy query", "https://example.com/article3"),
            ("AI systems", "https://example.com/article4"),
            ("deep learning", "https://example.com/article5"),
        ],
    )
    async def test_different_queries_rank_correctly(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        query: str,
        expected_top_url: str,
    ):
        """Test that different queries produce appropriate rankings."""
        input_schema = RerankerToolInputSchema(
            query=query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        # The most relevant result should rank first
        top_result = output.results[0]
        assert str(top_result.url) == expected_top_url

    @pytest.mark.asyncio
    async def test_empty_query_string(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
    ):
        """Test reranking with empty query string."""
        input_schema = RerankerToolInputSchema(
            query="",
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        # Should still produce results with scores
        assert len(output.results) > 0
        assert all(r.score is not None for r in output.results)


class TestRerankerToolScoreStorage:
    """Test that scores are properly stored in results."""

    @pytest.mark.asyncio
    async def test_scores_in_main_field(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that scores are stored in the main score field."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        for result in output.results:
            assert hasattr(result, "score")
            assert result.score is not None

    @pytest.mark.asyncio
    async def test_scores_in_extra_field(
        self,
        reranker_instance,
        sample_search_results: List[SearchResultItem],
        test_query: str,
    ):
        """Test that scores are also stored in the extra field."""
        input_schema = RerankerToolInputSchema(
            query=test_query,
            results=sample_search_results,
        )

        output = await reranker_instance.arun(input_schema)

        for result in output.results:
            assert "score" in result.extra
            assert result.extra["score"] is not None
            # Score in extra should match score in main field
            assert result.extra["score"] == result.score
