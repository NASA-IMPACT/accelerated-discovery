from typing import List, Literal, Optional

from atomic_agents.agents.base_agent import BaseIOSchema
from pydantic import Field

from ._base import BaseAgent


class QueryAgentInputSchema(BaseIOSchema):
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


class QueryAgentOutputSchema(BaseIOSchema):
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


class QueryAgent(BaseAgent):
    """
    Agent that generates search engine queries based on a given query.
    """

    input_schema = QueryAgentInputSchema
    output_schema = QueryAgentOutputSchema
