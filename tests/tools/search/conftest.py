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
    """
    Comprehensive SearchResultItem test data covering various scenarios:
    - Different scores (high, medium, low)
    - Different engines (google, arxiv, bing, google_scholar)
    - Duplicates (same URL)
    - Missing fields (empty title/content, None URL)
    - DOI handling
    """
    return [
        # High score, google engine
        SearchResultItem(
            title="High Score",
            content="test content",
            url="http://test1.com",
            score=0.9,
            engine="google",
            query="test",
            doi="10.1000/test123",
        ),
        # Medium score, arxiv engine
        SearchResultItem(
            title="Medium Score",
            content="test content",
            url="http://test2.com",
            score=0.3,
            engine="arxiv",
            query="test",
        ),
        # Low score, google engine (below default 0.25 cutoff)
        SearchResultItem(
            title="Low Score",
            content="test content",
            url="http://test3.com",
            score=0.1,
            engine="google",
            query="test",
        ),
        # Bing engine (for engine filtering tests)
        SearchResultItem(
            title="Bing Result",
            content="test content",
            url="http://test4.com",
            score=0.7,
            engine="bing",
            query="test",
        ),
        # Duplicate URL (same as test1.com)
        SearchResultItem(
            title="Duplicate URL",
            content="test content",
            url="http://test1.com",
            score=0.6,
            engine="google_scholar",
            query="test",
        ),
        # Missing title (empty string)
        SearchResultItem(
            title="",
            content="test content",
            url="http://test5.com",
            score=0.8,
            engine="arxiv",
            query="test",
        ),
        # Missing content (empty string)
        SearchResultItem(
            title="No Content",
            content="",
            url="http://test6.com",
            score=0.7,
            engine="google",
            query="test",
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


# ==================== Serper Tool Fixtures ====================


@pytest.fixture
def sample_serper_config():
    """Sample Serper configuration for testing."""
    from pydantic import SecretStr
    from pydantic.networks import HttpUrl

    from akd.tools.search.serper import SerperSearchToolConfig

    return SerperSearchToolConfig(
        api_key=SecretStr("test_api_key_12345"),
        base_url=HttpUrl("https://google.serper.dev"),
        category="scholar",
        max_results=10,
        num_per_page=20,
        score_cutoff=0.0,
        gl="us",
        hl="en",
        autocorrect=True,
        max_pages=5,
        result_multiplier=1.0,
        pre_authenticate=False,
        debug=False,
    )


@pytest.fixture
def sample_serper_search_input():
    """Sample Serper search input schema."""
    from akd.tools.search.serper import SerperSearchToolInputSchema

    return SerperSearchToolInputSchema(
        queries=["quantum computing applications", "neural networks"],
        category="science",
        max_results=5,
    )


@pytest.fixture
def mock_serper_scholar_response() -> Dict[str, Any]:
    """Mock Serper API response for scholar endpoint."""
    return {
        "searchParameters": {
            "q": "quantum computing",
            "gl": "us",
            "hl": "en",
            "num": 10,
            "page": 1,
            "type": "search",
            "engine": "google_scholar",
        },
        "organic": [
            {
                "title": "Quantum Computing: Progress and Prospects",
                "link": "https://arxiv.org/abs/2101.12345",
                "snippet": "We review the current state of quantum computing technology...",
                "date": "2021-05-15",
                "citation": "Nature Physics, 2021",
                "citedBy": 450,
                "pdfUrl": "https://arxiv.org/pdf/2101.12345.pdf",
                "position": 1,
            },
            {
                "title": "Advances in Quantum Error Correction",
                "link": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.456789",
                "snippet": "Quantum error correction is essential for scalable quantum computing...",
                "date": "2022-03-20",
                "citation": "Physical Review Letters, 2022",
                "citedBy": 320,
                "position": 2,
            },
            {
                "title": "Quantum Algorithms for Machine Learning",
                "link": "https://example.com/quantum-ml",
                "snippet": "This paper explores quantum algorithms for machine learning tasks...",
                "date": "2023-01-10",
                "citation": "arXiv preprint, 2023",
                "citedBy": 125,
                "position": 3,
            },
        ],
        "relatedSearches": [
            "quantum computing applications",
            "quantum supremacy",
            "quantum algorithms",
        ],
        "credits": 1,
    }


@pytest.fixture
def mock_serper_general_response() -> Dict[str, Any]:
    """Mock Serper API response for general search endpoint."""
    return {
        "searchParameters": {
            "q": "machine learning frameworks",
            "gl": "us",
            "hl": "en",
            "num": 10,
            "page": 1,
            "type": "search",
            "engine": "google",
        },
        "organic": [
            {
                "title": "TensorFlow: An open source machine learning framework",
                "link": "https://www.tensorflow.org/",
                "snippet": "TensorFlow is an end-to-end open source platform for machine learning...",
                "date": "2023-11-15",
                "position": 1,
                "sitelinks": [
                    {"title": "Get Started", "link": "https://www.tensorflow.org/learn"},
                    {"title": "Tutorials", "link": "https://www.tensorflow.org/tutorials"},
                ],
            },
            {
                "title": "PyTorch - Deep Learning Framework",
                "link": "https://pytorch.org/",
                "snippet": "PyTorch is a Python package that provides two high-level features...",
                "date": "2023-10-22",
                "position": 2,
            },
            {
                "title": "Scikit-learn: Machine Learning in Python",
                "link": "https://scikit-learn.org/",
                "snippet": "Simple and efficient tools for predictive data analysis...",
                "position": 3,
            },
        ],
        "topStories": [
            {
                "title": "New AI Framework Released",
                "link": "https://news.example.com/ai-framework",
                "source": "Tech News",
                "date": "2 hours ago",
            },
        ],
        "relatedSearches": [
            "best machine learning frameworks 2024",
            "deep learning frameworks comparison",
            "AI frameworks",
        ],
        "credits": 1,
    }


@pytest.fixture
def mock_serper_empty_response() -> Dict[str, Any]:
    """Mock empty Serper API response."""
    return {
        "searchParameters": {
            "q": "extremely rare query with no results",
            "gl": "us",
            "hl": "en",
            "num": 10,
            "page": 1,
        },
        "organic": [],
        "credits": 1,
    }


@pytest.fixture
def mock_serper_paginated_response_page1() -> Dict[str, Any]:
    """Mock Serper API response for page 1 of paginated results."""
    return {
        "searchParameters": {
            "q": "deep learning",
            "gl": "us",
            "hl": "en",
            "num": 10,
            "page": 1,
        },
        "organic": [
            {
                "title": f"Deep Learning Paper {i}",
                "link": f"https://example.com/paper-{i}",
                "snippet": f"This is paper {i} about deep learning...",
                "date": "2023-01-15",
                "position": i,
            }
            for i in range(1, 11)
        ],
        "relatedSearches": ["neural networks", "machine learning"],
        "credits": 1,
    }


@pytest.fixture
def mock_serper_paginated_response_page2() -> Dict[str, Any]:
    """Mock Serper API response for page 2 of paginated results."""
    return {
        "searchParameters": {
            "q": "deep learning",
            "gl": "us",
            "hl": "en",
            "num": 10,
            "page": 2,
        },
        "organic": [
            {
                "title": f"Deep Learning Paper {i}",
                "link": f"https://example.com/paper-{i}",
                "snippet": f"This is paper {i} about deep learning...",
                "date": "2023-01-20",
                "position": i,
            }
            for i in range(11, 21)
        ],
        "relatedSearches": ["convolutional networks", "recurrent networks"],
        "credits": 1,
    }


@pytest.fixture
def mock_serper_tool(sample_serper_config):
    """Mock Serper tool instance."""
    from akd.tools.search.serper import SerperSearchTool

    tool = SerperSearchTool(config=sample_serper_config)
    return tool


@pytest.fixture
def mock_serper_duplicate_results() -> Dict[str, Any]:
    """Mock Serper response with duplicate URLs for deduplication testing."""
    return {
        "searchParameters": {
            "q": "test query",
            "num": 10,
            "page": 1,
        },
        "organic": [
            {
                "title": "Original Article",
                "link": "https://example.com/duplicate",
                "snippet": "First occurrence of this URL",
                "position": 1,
            },
            {
                "title": "Unique Article",
                "link": "https://example.com/unique",
                "snippet": "This is a unique article",
                "position": 2,
            },
            {
                "title": "Duplicate Article",
                "link": "https://example.com/duplicate",  # Duplicate URL
                "snippet": "Second occurrence of same URL",
                "position": 3,
            },
        ],
        "credits": 1,
    }
