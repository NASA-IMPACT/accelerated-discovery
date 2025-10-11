"""
Shared fixtures for reranker tool tests.

This module provides common fixtures used across reranker test modules.
"""

from typing import List

import pytest

from akd.structures import SearchResultItem
from akd.tools.reranker import CrossEncoderRerankerTool


@pytest.fixture
def sample_search_results() -> List[SearchResultItem]:
    """
    Fixture providing sample search results for testing.

    The 3rd item has content "this is a dummy query content" which should
    rank highest when reranked with query "this is a dummy query".

    Returns:
        List of 5 SearchResultItem objects for testing reranking.
    """
    return [
        SearchResultItem(
            url="https://example.com/article1",
            title="First Research Article on Machine Learning",
            query="machine learning applications",
            content="This article discusses various applications of machine learning in scientific research.",
            category="research",
        ),
        SearchResultItem(
            url="https://example.com/article2",
            title="Second Study on Neural Networks",
            query="neural network architectures",
            content="A comprehensive study on different neural network architectures and their performance.",
            category="research",
            pdf_url="https://example.com/article2.pdf",
        ),
        SearchResultItem(
            url="https://example.com/article3",
            title="Third Paper on Data Science",
            query="data science methods",
            content="this is a dummy query content",
            category="research",
        ),
        SearchResultItem(
            url="https://example.com/article4",
            title="Fourth Analysis of AI Systems",
            query="artificial intelligence systems",
            content="An in-depth analysis of modern AI systems and their capabilities.",
            category="research",
            pdf_url="https://example.com/article4.pdf",
        ),
        SearchResultItem(
            url="https://example.com/article5",
            title="Fifth Review of Deep Learning",
            query="deep learning techniques",
            content="A comprehensive review of deep learning techniques used in various domains.",
            category="review",
        ),
    ]


@pytest.fixture
def expected_top_result() -> SearchResultItem:
    """
    Fixture for the expected top-ranked result after reranking.

    Returns:
        The 3rd SearchResultItem which should rank first for query "this is a dummy query".
    """
    return SearchResultItem(
        url="https://example.com/article3",
        title="Third Paper on Data Science",
        query="data science methods",
        content="this is a dummy query content",
        category="research",
    )


@pytest.fixture(
    params=[
        CrossEncoderRerankerTool,
        # Add new reranker implementations here to automatically include them in all tests
    ],
    ids=lambda cls: cls.__name__,
)
def reranker_tool_class(request):
    """
    Parameterized fixture providing all reranker tool implementations.

    This fixture enables the same test suite to run against all reranker
    implementations. When a new reranker is added, simply add it to the
    params list above.

    Args:
        request: pytest fixture request object.

    Returns:
        A reranker tool class.
    """
    return request.param


@pytest.fixture
def reranker_instance(reranker_tool_class):
    """
    Fixture providing a default-configured instance of a reranker tool.

    Args:
        reranker_tool_class: Parameterized reranker class from reranker_tool_class fixture.

    Returns:
        Configured instance of a reranker tool.
    """
    return reranker_tool_class()


@pytest.fixture
def test_query() -> str:
    """
    Fixture providing the test query string.

    Returns:
        Query string designed to match the 3rd search result.
    """
    return "this is a dummy query"
