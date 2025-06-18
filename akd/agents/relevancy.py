from typing import List, Optional

from pydantic import Field, BaseModel

from akd.configs.project import CONFIG
from akd.structures import RelevancyLabel

from ._base import BaseAgent


class RelevancyAgentInputSchema(BaseModel):
    """Input schema for relevancy agent"""

    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )
    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )


class RelevancyAgentOutputSchema(BaseModel):
    """Output schema for relevancy agent"""

    label: RelevancyLabel = Field(
        ...,
        description=(
            "The label indicating the relevance between " "the query and the content."
        ),
    )
    reasoning_steps: List[str] = Field(
        ...,
        description=(
            "Very concise/step-by-step reasoning steps leading to the "
            "relevance check."
        ),
    )


class RelevancyAgent(BaseAgent):
    input_schema = RelevancyAgentInputSchema
    output_schema = RelevancyAgentOutputSchema
