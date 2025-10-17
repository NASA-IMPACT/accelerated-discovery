from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import LiteLLMInstructorBaseAgent
from akd.structures import SearchResultItem


class AnswerAgentInputSchema(InputSchema):
    """Input schema for the Answer agent to answer query based on search results."""

    query: str = Field(..., description="The query to answer")
    search_results: list[SearchResultItem] = Field(..., description="The search results to use for answering the query")
    additional_context: str | None = Field(
        None,
        description="Any additional context to consider while answering the query",
    )


class AnswerAgentOutputSchema(OutputSchema):
    """Output schema for the Answer agent containing the generated answer."""

    answer: str = Field(..., description="Short and concise answer generated based on the search results for the given")
    reasoning_traces: list[str] = Field(..., description="Concise reasoning traces used to generate the answer")


class AnswerAgent(
    LiteLLMInstructorBaseAgent[AnswerAgentInputSchema, AnswerAgentOutputSchema],
):
    """
    Agent that answers a query based on provided search results.
    The answer is entirely generated based on search results.
    """

    input_schema = AnswerAgentInputSchema
    output_schema = AnswerAgentOutputSchema
