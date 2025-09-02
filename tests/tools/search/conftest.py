"""
Fixtures and test data for SearxNG search tool tests.
"""

import os
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from akd.structures import SearchResultItem
from akd.tools.search.searxng_search import (
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
    SearxNGSearchToolInputSchema,
)


@pytest.fixture
def sample_searxng_config():
    """Sample SearxNG configuration for testing."""
    return SearxNGSearchToolConfig(
        base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
        max_results=10,
        engines=["google", "arxiv", "google_scholar"],
        max_pages=5,
        results_per_page=10,
        score_cutoff=0.25,
        strict=True,
        debug=False,
    )


@pytest.fixture
def sample_search_input():
    """Sample search input schema."""
    return SearxNGSearchToolInputSchema(
        queries=["machine learning", "deep learning"],
        category="science",
        max_results=5,
    )


@pytest.fixture
def mock_searxng_response() -> Dict[str, Any]:
    """Mock SearxNG API response."""
    return {
        "results": [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence...",
                "url": "https://example.com/ml-intro",
                "engine": "google",
                "score": 0.95,
                "publishedDate": "2023-01-15",
                "doi": "10.1000/test123",
                "category": "science",
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning is a machine learning technique...",
                "url": "https://example.com/dl-fundamentals",
                "engine": "arxiv",
                "score": 0.90,
                "publishedDate": "2023-02-10",
                "category": "science",
            },
            {
                "title": "Neural Networks Explained",
                "content": "Neural networks are computing systems inspired by...",
                "url": "https://example.com/neural-networks",
                "engine": "google_scholar",
                "score": 0.85,
                "publishedDate": "2023-03-05",
            },
            {
                "title": "Low Score Article",
                "content": "This article has a low relevance score...",
                "url": "https://example.com/low-score",
                "engine": "google",
                "score": 0.15,  # Below default cutoff
            },
        ],
    }


@pytest.fixture
def mock_searxng_empty_response() -> dict[str, Any]:
    """Mock empty SearxNG API response."""
    return {"results": []}


@pytest.fixture
def mock_searxng_malformed_response() -> dict[str, Any]:
    """Mock malformed SearxNG API response for error testing."""
    return {
        "results": [
            {
                "title": "Article with Missing Fields",
                # Missing required fields like 'url', 'content'
                "engine": "google",
                "score": 0.8,
            },
            {
                # Missing title
                "content": "Content without title",
                "url": "https://example.com/no-title",
                "engine": "arxiv",
                "score": 0.7,
            },
        ],
    }


@pytest.fixture
def sample_search_result_items() -> list[SearchResultItem]:
    """Sample SearchResultItem objects for testing."""
    return [
        SearchResultItem(
            url="https://example.com/ml-intro",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence...",
            query="machine learning",
            category="science",
            doi="10.1000/test123",
            published_date="2023-01-15",
            engine="google",
        ),
        SearchResultItem(
            url="https://example.com/dl-fundamentals",
            title="Deep Learning Fundamentals",
            content="Deep learning is a machine learning technique...",
            query="deep learning",
            category="science",
            published_date="2023-02-10",
            engine="arxiv",
        ),
    ]


@pytest.fixture
def mock_searxng_tool(sample_searxng_config):
    """Mock SearxNG tool instance."""
    tool = SearxNGSearchTool(config=sample_searxng_config)
    return tool


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for testing."""
    session = AsyncMock()

    # Mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.reason = "OK"
    mock_response.url = "http://localhost:8080/search"

    # Context manager support
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    return session


@pytest.fixture
def mock_failed_aiohttp_session():
    """Mock aiohttp ClientSession that returns HTTP errors."""
    session = AsyncMock()

    # Mock failed response
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"
    mock_response.url = "http://localhost:8080/search"

    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    return session


@pytest.fixture
def mock_network_error_session():
    """Mock aiohttp ClientSession that raises network errors."""
    session = AsyncMock()
    session.get.side_effect = Exception("Network error")
    return session


@pytest.fixture
def duplicate_results_response() -> Dict[str, Any]:
    """Mock response with duplicate URLs for deduplication testing."""
    return {
        "results": [
            {
                "title": "Original Article",
                "content": "This is the original article...",
                "url": "https://example.com/article",
                "engine": "google",
                "score": 0.95,
            },
            {
                "title": "Duplicate Article",
                "content": "This is a duplicate of the same article...",
                "url": "https://example.com/article",  # Same URL
                "engine": "arxiv",
                "score": 0.85,
            },
            {
                "title": "Another Article",
                "content": "This is a different article...",
                "url": "https://example.com/different",
                "engine": "google_scholar",
                "score": 0.80,
            },
        ],
    }
