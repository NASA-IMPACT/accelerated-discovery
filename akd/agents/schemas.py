"""
Shared schemas for agents to avoid circular imports.

This module contains input/output schemas that are used across multiple modules,
separated from the agent implementations to prevent circular dependencies.
"""

from typing import List, Literal, Optional
from pydantic import Field
from akd._base import InputSchema, OutputSchema


class QueryAgentInputSchema(InputSchema):
    """This is the input schema for the QueryAgent."""

    query: str = Field(
        ...,
        description="A detailed query/instruction or request to "
        "generate search engine queries for.",
    )
    num_queries: int = Field(
        default=3,
        description="The number of search queries to generate.",
    )


class QueryAgentOutputSchema(OutputSchema):
    """
    Schema for output queries  for information, news,
    references, and other content.

    Returns a list of search results with a short
    description or content snippet and URLs for further exploration
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "science"]] = Field(
        "science",
        description="Category of the search queries.",
    )


class FollowUpQueryAgentInputSchema(InputSchema):
    """This is the input schema for the FollowUpQueryAgent."""

    original_queries: List[str] = Field(
        ...,
        description="The original search queries that were used to retrieve content.",
    )

    content: str = Field(
        ...,
        description="The text content obtained from the original queries "
        "that will be used to generate follow-up queries.",
    )

    num_queries: int = Field(
        default=3,
        description="The number of follow-up search queries to generate.",
    )

    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas or aspects to focus on for follow-up queries. "
        "If not provided, the agent will identify gaps and interesting areas automatically.",
    )


class FollowUpQueryAgentOutputSchema(OutputSchema):
    """
    Schema for output follow-up queries based on original queries and content.
    Returns a list of refined search queries that dig deeper into the topic
    or explore related areas not covered in the original content.
    """

    followup_queries: List[str] = Field(
        ...,
        description="List of follow-up search queries based on the original content.",
    )

    category: Optional[Literal["general", "science", "research", "clarification"]] = (
        Field(
            default="general",
            description="Category of the follow-up search queries.",
        )
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why these follow-up queries were generated "
        "and what gaps or areas they aim to address.",
    )

    original_query_gaps: Optional[List[str]] = Field(
        default=None,
        description="Identified gaps or areas that weren't fully covered "
        "in the original content.",
    )