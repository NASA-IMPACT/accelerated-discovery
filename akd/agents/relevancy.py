from typing import List

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.structures import RelevancyLabel


class RelevancyAgentInputSchema(InputSchema):
    """Input schema for relevancy agent"""

    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )
    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )


class RelevancyAgentOutputSchema(OutputSchema):
    """Output schema for relevancy agent"""

    label: RelevancyLabel = Field(
        ...,
        description=(
            "The label indicating the relevance between the query and the content."
        ),
    )
    reasoning_steps: List[str] = Field(
        ...,
        description=(
            "Very concise/step-by-step reasoning steps leading to the relevance check."
        ),
    )


class RelevancyAgent(
    InstructorBaseAgent[RelevancyAgentInputSchema, RelevancyAgentOutputSchema],
):
    input_schema = RelevancyAgentInputSchema
    output_schema = RelevancyAgentOutputSchema
