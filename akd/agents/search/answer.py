from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import LiteLLMInstructorBaseAgent
from akd.structures import SearchResultItem


class QuestionAnsweringAgentInputSchema(InputSchema):
    """Input schema for the Question Answering agent to answer query based on search results."""

    query: str = Field(..., description="The query to answer")
    search_results: list[SearchResultItem] = Field(..., description="The search results to use for answering the query")
    additional_context: str | None = Field(
        None,
        description="Any additional context to consider while answering the query",
    )


class QuestionAnsweringAgentOutputSchema(OutputSchema):
    """Output schema for the Question Answering agent containing the generated answer."""

    answer: str = Field(
        ...,
        title="answer",  # Explicit title to prevent LLM from capitalizing
        description="Short and concise answer generated based on the search results for the given query",
    )
    reasoning_traces: list[str] = Field(
        ...,
        title="reasoning_traces",  # Explicit title to prevent LLM from capitalizing
        description="Concise reasoning traces used to generate the answer",
    )


class QuestionAnsweringAgent(
    LiteLLMInstructorBaseAgent[QuestionAnsweringAgentInputSchema, QuestionAnsweringAgentOutputSchema],
):
    """
    Question Answering agent that generates answers based on provided search results.
    The answer is entirely generated based on the search results provided.
    """

    input_schema = QuestionAnsweringAgentInputSchema
    output_schema = QuestionAnsweringAgentOutputSchema
