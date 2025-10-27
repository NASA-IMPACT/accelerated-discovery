"""Tests for reciprocal_rank_fusion utility function."""

import pytest

from akd.structures import SearchResultItem
from akd.utils import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Test suite for RRF (Reciprocal Rank Fusion) implementation."""

    def test_single_query_maintains_order(self):
        """Test that single query results maintain order with normalized scores."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content="Content A"),
            SearchResultItem(url="https://example.com/b", title="B", query="test", content="Content B"),
            SearchResultItem(url="https://example.com/c", title="C", query="test", content="Content C"),
        ]

        fused = reciprocal_rank_fusion(results)

        assert len(fused) == 3
        # URLs should maintain order
        assert str(fused[0].url) == "https://example.com/a"
        assert str(fused[1].url) == "https://example.com/b"
        assert str(fused[2].url) == "https://example.com/c"
        # Top item gets 1.0, bottom gets 0.1
        assert fused[0].score == pytest.approx(1.0)
        assert fused[2].score == pytest.approx(0.1)

    def test_common_items_get_boosted(self):
        """Test that items appearing in multiple queries get higher scores."""
        results_1 = [
            SearchResultItem(url="https://example.com/a", title="A", query="q1", content=""),
            SearchResultItem(url="https://example.com/b", title="B", query="q1", content=""),
        ]

        results_2 = [
            SearchResultItem(url="https://example.com/b", title="B", query="q2", content=""),
            SearchResultItem(url="https://example.com/c", title="C", query="q2", content=""),
        ]

        fused = reciprocal_rank_fusion(results_1, results_2)

        # B appears in both queries, should be top
        assert str(fused[0].url) == "https://example.com/b"
        assert fused[0].score == pytest.approx(1.0)  # Highest normalized score

    def test_rrf_score_in_extra(self):
        """Test that raw RRF scores are preserved in extra field."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content=""),
        ]

        fused = reciprocal_rank_fusion(results)

        assert "rrf_score" in fused[0].extra
        # Single item at rank 1: RRF = 1/(60+1)
        assert fused[0].extra["rrf_score"] == pytest.approx(1.0 / 61)

    def test_empty_results(self):
        """Test handling of empty query results."""
        fused = reciprocal_rank_fusion()
        assert fused == []

    def test_normalization_disabled(self):
        """Test RRF without normalization returns raw scores."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content=""),
            SearchResultItem(url="https://example.com/b", title="B", query="test", content=""),
        ]

        fused = reciprocal_rank_fusion(results, normalize=False)

        # Without normalization, scores are raw RRF
        assert fused[0].score == pytest.approx(1.0 / 61)  # rank 1
        assert fused[1].score == pytest.approx(1.0 / 62)  # rank 2

    def test_normalization_range(self):
        """Test that normalized scores are in [0.1, 1.0] range."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content=""),
            SearchResultItem(url="https://example.com/b", title="B", query="test", content=""),
            SearchResultItem(url="https://example.com/c", title="C", query="test", content=""),
        ]

        fused = reciprocal_rank_fusion(results, normalize=True)

        scores = [item.score for item in fused]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) == pytest.approx(0.1)

    def test_custom_k_parameter(self):
        """Test that custom k parameter affects RRF scoring."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content=""),
        ]

        fused_k60 = reciprocal_rank_fusion(results, k=60, normalize=False)
        fused_k10 = reciprocal_rank_fusion(results, k=10, normalize=False)

        # Lower k means higher scores
        assert fused_k10[0].score > fused_k60[0].score

    def test_custom_deduplication_key(self):
        """Test using custom deduplication key like DOI."""
        results_1 = [
            SearchResultItem(
                url="https://example.com/a",
                title="A",
                query="q1",
                content="",
                doi="10.1234/a",
            ),
        ]

        results_2 = [
            SearchResultItem(
                url="https://different.com/a",
                title="A",
                query="q2",
                content="",
                doi="10.1234/a",
            ),
        ]

        # Using URL key - should get 2 items (different URLs)
        fused_url = reciprocal_rank_fusion(results_1, results_2, keys="url")
        assert len(fused_url) == 2

        # Using DOI key - should get 1 item (same DOI, deduplicated)
        fused_doi = reciprocal_rank_fusion(results_1, results_2, keys="doi")
        assert len(fused_doi) == 1

    def test_original_items_not_mutated(self):
        """Test that original SearchResultItems are not mutated."""
        original = SearchResultItem(url="https://example.com/a", title="A", query="test", content="")

        fused = reciprocal_rank_fusion([original])

        # Original should not have score set
        assert original.score is None
        # Result should have score
        assert fused[0].score is not None

    def test_single_item_normalized_to_one(self):
        """Test that single item gets score 1.0 when normalized."""
        results = [
            SearchResultItem(url="https://example.com/a", title="A", query="test", content=""),
        ]

        fused = reciprocal_rank_fusion(results, normalize=True)

        # Single item gets 1.0 (no range to normalize)
        assert fused[0].score == pytest.approx(1.0)
