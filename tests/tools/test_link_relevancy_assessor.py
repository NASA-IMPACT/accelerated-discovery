"""
Blackbox tests for LinkRelevancyAssessor tool.

These tests verify the external behavior of the LinkRelevancyAssessor without
examining internal implementation details.
"""

from typing import List
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from akd.agents.relevancy import (
    ContentDepthLabel,
    EnhancedRelevancyLabel,
    EvidenceQualityLabel,
    MethodologicalRelevanceLabel,
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyOutputSchema,
    RecencyRelevanceLabel,
    ScopeRelevanceLabel,
    TopicAlignmentLabel,
)
from akd.structures import SearchResultItem
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
    LinkRelevancyAssessorInputSchema,
    LinkRelevancyAssessorOutputSchema,
    ScoringWeights,
)


class TestLinkRelevancyAssessor:
    """Test suite for LinkRelevancyAssessor blackbox functionality."""

    @pytest.fixture
    def mock_relevancy_agent(self):
        """Create a mock MultiRubricRelevancyAgent with predictable responses."""
        agent = AsyncMock(spec=MultiRubricRelevancyAgent)

        # Default high-relevancy response
        agent.arun.return_value = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.COMPREHENSIVE,
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            overall_relevance=EnhancedRelevancyLabel.HIGHLY_RELEVANT,
            reasoning_steps=[
                "Content aligns with query",
                "High quality evidence found",
            ],
        )

        agent.reset_memory = Mock()
        return agent

    @pytest.fixture
    def sample_search_results(self) -> List[SearchResultItem]:
        """Create sample search results for testing."""
        return [
            SearchResultItem(
                query="climate change impacts",
                title="Climate Change Effects on Agriculture",
                content="Detailed analysis of climate change impacts on crop yields and agricultural systems.",
                url="https://example.com/climate-agriculture",
                extra={},
            ),
            SearchResultItem(
                query="climate change impacts",
                title="Weather Report",
                content="Today's weather will be sunny with temperatures reaching 25°C.",
                url="https://example.com/weather",
                extra={},
            ),
            SearchResultItem(
                query="climate change impacts",
                title="Climate Science Research",
                content="Comprehensive study on greenhouse gas emissions and global temperature rise.",
                url="https://example.com/climate-research",
                extra={},
            ),
        ]

    @pytest.fixture
    def basic_input(self, sample_search_results) -> LinkRelevancyAssessorInputSchema:
        """Create basic input schema for testing."""
        return LinkRelevancyAssessorInputSchema(
            search_results=sample_search_results,
            original_query="climate change impacts",
            reformulated_query=None,
            domain_context="environmental science",
        )

    @pytest.fixture
    def default_config(self) -> LinkRelevancyAssessorConfig:
        """Create default configuration for testing."""
        return LinkRelevancyAssessorConfig(
            min_relevancy_score=0.3,
            full_content_threshold=0.7,
            assessment_batch_size=5,
            enable_caching=True,
            debug=False,
        )

    @pytest.fixture
    def assessor(self, default_config, mock_relevancy_agent) -> LinkRelevancyAssessor:
        """Create LinkRelevancyAssessor instance for testing."""
        return LinkRelevancyAssessor(
            config=default_config,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

    # Core Functionality Tests

    @pytest.mark.asyncio
    async def test_basic_assessment(self, assessor, basic_input):
        """Test basic relevancy assessment functionality."""
        result = await assessor.arun(basic_input)

        # Verify output schema compliance
        assert isinstance(result, LinkRelevancyAssessorOutputSchema)
        assert isinstance(result.assessed_results, list)
        assert isinstance(result.filtered_results, list)
        assert isinstance(result.high_relevancy_results, list)
        assert isinstance(result.assessment_summary, dict)

        # Verify all original results are assessed
        assert len(result.assessed_results) == len(basic_input.search_results)

        # Verify all assessed results have scores
        for result_item in result.assessed_results:
            assert result_item.score is not None
            assert 0.0 <= result_item.score <= 1.0
            assert "relevancy_assessment" in result_item.extra

    @pytest.mark.asyncio
    async def test_empty_search_results(self, assessor):
        """Test handling of empty search results list."""
        empty_input = LinkRelevancyAssessorInputSchema(
            search_results=[],
            original_query="test query",
            reformulated_query=None,
            domain_context=None,
        )

        result = await assessor.arun(empty_input)

        assert len(result.assessed_results) == 0
        assert len(result.filtered_results) == 0
        assert len(result.high_relevancy_results) == 0
        assert result.assessment_summary["total_results"] == 0

    @pytest.mark.asyncio
    async def test_no_content_results(self, assessor, mock_relevancy_agent):
        """Test handling of search results without content."""
        no_content_results = [
            SearchResultItem(
                query="test query",
                title="Test Title",
                content="",  # No content
                url="https://example.com/no-content",
                extra={},
            ),
            SearchResultItem(
                query="test query",
                title="Another Test",
                content=None,  # None content
                url="https://example.com/none-content",
                extra={},
            ),
        ]

        input_data = LinkRelevancyAssessorInputSchema(
            search_results=no_content_results,
            original_query="test query",
            reformulated_query=None,
            domain_context=None,
        )

        result = await assessor.arun(input_data)

        # Should not call relevancy agent for no-content results
        mock_relevancy_agent.arun.assert_not_called()

        # All results should have low scores
        for result_item in result.assessed_results:
            assert result_item.score == 0.1
            assert result_item.extra.get("should_fetch_full_content") is False

    @pytest.mark.asyncio
    async def test_scoring_range_validation(self, assessor, basic_input):
        """Test that all relevancy scores are within valid range [0.0, 1.0]."""
        result = await assessor.arun(basic_input)

        for result_item in result.assessed_results:
            assert result_item.score is not None
            assert 0.0 <= result_item.score <= 1.0, (
                f"Score {result_item.score} out of range for {result_item.url}"
            )

    # Configuration Tests

    @pytest.mark.asyncio
    async def test_custom_scoring_weights(self, mock_relevancy_agent, basic_input):
        """Test with custom scoring weight configuration."""
        custom_weights = ScoringWeights(
            topic_alignment_weight=0.5,
            content_depth_weight=0.2,
            evidence_quality_weight=0.1,
            methodological_relevance_weight=0.1,
            recency_relevance_weight=0.05,
            scope_relevance_weight=0.05,
        )

        config = LinkRelevancyAssessorConfig(
            scoring_weights=custom_weights,
            min_relevancy_score=0.4,
            debug=False,
        )

        assessor = LinkRelevancyAssessor(
            config=config,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

        result = await assessor.arun(basic_input)

        # Verify configuration is applied (should affect scoring calculation)
        assert len(result.assessed_results) > 0
        for result_item in result.assessed_results:
            assert result_item.score is not None

    @pytest.mark.asyncio
    async def test_relevancy_thresholds(self, mock_relevancy_agent, basic_input):
        """Test minimum relevancy score and full content threshold configuration."""
        # Configure with high thresholds
        config = LinkRelevancyAssessorConfig(
            min_relevancy_score=0.8,
            full_content_threshold=0.9,
            debug=False,
        )

        # Mock low-relevancy responses
        mock_relevancy_agent.arun.return_value = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.NOT_ALIGNED,
            content_depth=ContentDepthLabel.SURFACE_LEVEL,
            evidence_quality=EvidenceQualityLabel.LOW_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_WEAK,
            recency_relevance=RecencyRelevanceLabel.OUTDATED,
            scope_relevance=ScopeRelevanceLabel.OUT_OF_SCOPE,
            overall_relevance=EnhancedRelevancyLabel.NOT_RELEVANT,
            reasoning_steps=["Low relevancy assessment"],
        )

        assessor = LinkRelevancyAssessor(
            config=config,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

        result = await assessor.arun(basic_input)

        # With low scores and high thresholds, should have fewer filtered/high results
        assert len(result.filtered_results) <= len(result.assessed_results)
        assert len(result.high_relevancy_results) <= len(result.filtered_results)

    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_relevancy_agent, basic_input):
        """Test different batch sizes for assessment processing."""
        # Test with batch size of 1
        config = LinkRelevancyAssessorConfig(
            assessment_batch_size=1,
            debug=False,
        )

        assessor = LinkRelevancyAssessor(
            config=config,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

        result = await assessor.arun(basic_input)

        # Should process all results regardless of batch size
        assert len(result.assessed_results) == len(basic_input.search_results)
        # Each result with content should trigger agent call
        expected_calls = len([r for r in basic_input.search_results if r.content])
        assert mock_relevancy_agent.arun.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_caching_behavior(self, mock_relevancy_agent, basic_input):
        """Test caching enabled vs disabled behavior."""
        # Test with caching enabled
        config_cached = LinkRelevancyAssessorConfig(
            enable_caching=True,
            debug=False,
        )

        assessor_cached = LinkRelevancyAssessor(
            config=config_cached,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

        # Run assessment twice
        await assessor_cached.arun(basic_input)
        mock_relevancy_agent.arun.reset_mock()

        result2 = await assessor_cached.arun(basic_input)

        # Should use cache for second run (no new agent calls)
        assert mock_relevancy_agent.arun.call_count == 0
        assert result2.assessment_summary["cache_hits"] > 0

    # Edge Cases

    @pytest.mark.asyncio
    async def test_reformulated_query_handling(
        self,
        assessor,
        mock_relevancy_agent,
        sample_search_results,
    ):
        """Test handling of original vs reformulated queries."""
        input_with_reformulated = LinkRelevancyAssessorInputSchema(
            search_results=sample_search_results,
            original_query="climate change impacts",
            reformulated_query="effects of global warming on environment",
            domain_context="environmental science",
        )

        result = await assessor.arun(input_with_reformulated)

        # Should assess against both queries
        expected_calls = (
            len([r for r in sample_search_results if r.content]) * 2
        )  # original + reformulated
        assert mock_relevancy_agent.arun.call_count == expected_calls

        # Results should have query alignment details
        for result_item in result.assessed_results:
            if result_item.content:
                assert "query_alignment_details" in result_item.extra

    @pytest.mark.asyncio
    async def test_large_content_handling(self, assessor, mock_relevancy_agent):
        """Test handling of very large content strings."""
        large_content_result = SearchResultItem(
            query="test query",
            title="Large Content Test",
            content="A" * 10000,  # Very large content
            url="https://example.com/large-content",
            extra={},
        )

        input_data = LinkRelevancyAssessorInputSchema(
            search_results=[large_content_result],
            original_query="test query",
            reformulated_query=None,
            domain_context=None,
        )

        result = await assessor.arun(input_data)

        # Should handle large content without errors
        assert len(result.assessed_results) == 1
        assert result.assessed_results[0].score is not None
        assert mock_relevancy_agent.arun.called

    @pytest.mark.asyncio
    async def test_special_characters_handling(self, assessor, mock_relevancy_agent):
        """Test handling of content with special characters."""
        special_char_result = SearchResultItem(
            query="test query",
            title="Special Characters: áéíóú ñ ¿¡ €£¥",
            content="Content with special characters: 🌍🔬📊 and unicode: αβγδε",
            url="https://example.com/special-chars",
            extra={},
        )

        input_data = LinkRelevancyAssessorInputSchema(
            search_results=[special_char_result],
            original_query="test query with special chars: αβγ",
            reformulated_query=None,
            domain_context=None,
        )

        result = await assessor.arun(input_data)

        # Should handle special characters without errors
        assert len(result.assessed_results) == 1
        assert result.assessed_results[0].score is not None

    # Output Validation Tests

    @pytest.mark.asyncio
    async def test_output_schema_compliance(self, assessor, basic_input):
        """Test comprehensive output schema validation."""
        result = await assessor.arun(basic_input)

        # Test main output schema
        assert isinstance(result, LinkRelevancyAssessorOutputSchema)

        # Test assessed_results
        assert isinstance(result.assessed_results, list)
        for item in result.assessed_results:
            assert isinstance(item, SearchResultItem)
            assert item.score is not None
            assert isinstance(item.score, (int, float))
            assert 0.0 <= item.score <= 1.0

        # Test filtered_results
        assert isinstance(result.filtered_results, list)
        assert len(result.filtered_results) <= len(result.assessed_results)

        # Test high_relevancy_results
        assert isinstance(result.high_relevancy_results, list)
        assert len(result.high_relevancy_results) <= len(result.assessed_results)

        # Test assessment_summary
        assert isinstance(result.assessment_summary, dict)
        required_summary_keys = [
            "total_results",
            "assessed_results",
            "avg_relevancy_score",
            "min_relevancy_score",
            "max_relevancy_score",
            "high_relevancy_count",
            "filtered_count",
            "cache_hits",
        ]
        for key in required_summary_keys:
            assert key in result.assessment_summary

    @pytest.mark.asyncio
    async def test_result_filtering_accuracy(self, mock_relevancy_agent, basic_input):
        """Test accuracy of result filtering based on thresholds."""
        # Configure specific thresholds
        config = LinkRelevancyAssessorConfig(
            min_relevancy_score=0.5,
            full_content_threshold=0.8,
            debug=False,
        )

        # Mock mixed relevancy responses
        def mock_assessment_side_effect(input_schema):
            if "agriculture" in input_schema.content.lower():
                # High relevancy
                return MultiRubricRelevancyOutputSchema(
                    topic_alignment=TopicAlignmentLabel.ALIGNED,
                    content_depth=ContentDepthLabel.COMPREHENSIVE,
                    evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
                    methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
                    recency_relevance=RecencyRelevanceLabel.CURRENT,
                    scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
                    overall_relevance=EnhancedRelevancyLabel.HIGHLY_RELEVANT,
                    reasoning_steps=["High relevancy"],
                )
            else:
                # Low relevancy
                return MultiRubricRelevancyOutputSchema(
                    topic_alignment=TopicAlignmentLabel.NOT_ALIGNED,
                    content_depth=ContentDepthLabel.SURFACE_LEVEL,
                    evidence_quality=EvidenceQualityLabel.LOW_QUALITY_EVIDENCE,
                    methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_WEAK,
                    recency_relevance=RecencyRelevanceLabel.OUTDATED,
                    scope_relevance=ScopeRelevanceLabel.OUT_OF_SCOPE,
                    overall_relevance=EnhancedRelevancyLabel.NOT_RELEVANT,
                    reasoning_steps=["Low relevancy"],
                )

        mock_relevancy_agent.arun.side_effect = mock_assessment_side_effect

        assessor = LinkRelevancyAssessor(
            config=config,
            relevancy_agent=mock_relevancy_agent,
            debug=False,
        )

        result = await assessor.arun(basic_input)

        # Verify filtering logic
        for item in result.filtered_results:
            assert item.score >= config.min_relevancy_score

        for item in result.high_relevancy_results:
            assert item.extra.get("should_fetch_full_content") is True

    @pytest.mark.asyncio
    async def test_assessment_summary_accuracy(self, assessor, basic_input):
        """Test accuracy of assessment summary statistics."""
        result = await assessor.arun(basic_input)

        summary = result.assessment_summary
        assessed_results = result.assessed_results

        # Verify summary statistics
        assert summary["total_results"] == len(basic_input.search_results)
        assert summary["assessed_results"] == len(assessed_results)

        # Verify score calculations
        scores = [r.score for r in assessed_results if r.score is not None]
        if scores:
            expected_avg = sum(scores) / len(scores)
            assert abs(summary["avg_relevancy_score"] - expected_avg) < 0.001
            assert summary["min_relevancy_score"] == min(scores)
            assert summary["max_relevancy_score"] == max(scores)

        # Verify counts
        actual_filtered_count = len(
            [
                r
                for r in assessed_results
                if r.score is not None
                and r.score >= assessor.config.min_relevancy_score
            ],
        )
        assert summary["filtered_count"] == actual_filtered_count

        actual_high_relevancy_count = len(
            [
                r
                for r in assessed_results
                if r.extra.get("should_fetch_full_content", False)
            ],
        )
        assert summary["high_relevancy_count"] == actual_high_relevancy_count

    # Configuration Validation Tests

    def test_invalid_scoring_weights(self):
        """Test validation of scoring weights that don't sum to 1.0."""
        with pytest.raises(ValidationError):
            ScoringWeights(
                topic_alignment_weight=0.5,
                content_depth_weight=0.5,  # Total = 1.0, others = 0, should fail
                evidence_quality_weight=0.2,  # This makes total > 1.0
                methodological_relevance_weight=0.0,
                recency_relevance_weight=0.0,
                scope_relevance_weight=0.0,
            )

    def test_valid_scoring_weights(self):
        """Test validation of valid scoring weights."""
        # Should not raise exception
        weights = ScoringWeights(
            topic_alignment_weight=0.4,
            content_depth_weight=0.2,
            evidence_quality_weight=0.2,
            methodological_relevance_weight=0.1,
            recency_relevance_weight=0.05,
            scope_relevance_weight=0.05,
        )
        assert weights is not None

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        config = LinkRelevancyAssessorConfig(
            min_relevancy_score=0.5,
            full_content_threshold=0.7,
            assessment_batch_size=3,
        )
        assert config.min_relevancy_score == 0.5
        assert config.full_content_threshold == 0.7
        assert config.assessment_batch_size == 3

        # Test invalid ranges
        with pytest.raises(ValidationError):
            LinkRelevancyAssessorConfig(min_relevancy_score=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            LinkRelevancyAssessorConfig(min_relevancy_score=-0.1)  # < 0.0

        with pytest.raises(ValidationError):
            LinkRelevancyAssessorConfig(assessment_batch_size=0)  # < 1
