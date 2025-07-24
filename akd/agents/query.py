from ._base import InstructorBaseAgent
from .schemas import (
    QueryAgentInputSchema,
    QueryAgentOutputSchema,
    FollowUpQueryAgentInputSchema,
    FollowUpQueryAgentOutputSchema,
)


class QueryAgent(InstructorBaseAgent[QueryAgentInputSchema, QueryAgentOutputSchema]):
    """
    Agent that generates search engine queries based on a given query.
    """

    input_schema = QueryAgentInputSchema
    output_schema = QueryAgentOutputSchema


# --- follow up agent


class FollowUpQueryAgent(
    InstructorBaseAgent[FollowUpQueryAgentInputSchema, FollowUpQueryAgentOutputSchema],
):
    """
    Agent that generates follow-up search engine queries based on original queries and content.
    """

    input_schema = FollowUpQueryAgentInputSchema
    output_schema = FollowUpQueryAgentOutputSchema
