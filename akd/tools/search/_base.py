from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig


class QueryFocusStrategy(str, Enum):
    """Query adaptation strategies based on rubric analysis."""

    REFINE_TOPIC_SPECIFICITY = "refine_topic_specificity"
    SEARCH_COMPREHENSIVE_REVIEWS = "search_comprehensive_reviews"
    TARGET_PEER_REVIEWED_SOURCES = "target_peer_reviewed_sources"
    SEARCH_METHODOLOGICAL_PAPERS = "search_methodological_papers"
    ADD_RECENT_YEAR_FILTERS = "add_recent_year_filters"
    ADJUST_QUERY_SCOPE = "adjust_query_scope"


class SearchToolConfig(BaseToolConfig):
    """
    Base configuration for search tools.
    This can be extended by specific search tool configurations.
    """

    max_results: int = Field(
        10,
        description="Maximum number of search results to return.",
    )


class SearchToolInputSchema(InputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content.
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "science", "technology"]] = Field(
        "science",
        description="Category of the search queries.",
    )
    max_results: int = Field(
        10,
        description="Maximum number of search results to return.",
    )


class SearchToolOutputSchema(OutputSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content."""

    results: List[SearchResultItem] = Field(
        ...,
        description="List of search result items",
    )
    category: Optional[str] = Field(
        None,
        description="The category of the search results",
    )


class SearchTool(BaseTool[SearchToolInputSchema, SearchToolOutputSchema]):
    """
    Tool for performing searches on SearxNG based on the provided queries and category.

    Attributes:
        input_schema (SearchToolInputSchema): The schema for the input data.
        output_schema (SearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        base_url (str): The base URL for the SearxNG instance to use.
    """

    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema
    config_schema = SearchToolConfig


class AgenticSearchTool(SearchTool):
    """
    Type for agentic search tools that use multi-rubric analysis
    and does agentic decision-making.
    """
