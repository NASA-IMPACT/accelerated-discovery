from enum import Enum
from typing import List, Optional

import instructor
import openai
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from loguru import logger
from pydantic import Field

from ..config import CONFIG
from ..structures import RelevancyLabel


class RelevancyAgentInputSchema(BaseIOSchema):
    """Input schema for relevancy agent"""

    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )
    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )


class RelevancyAgentOutputSchema(BaseIOSchema):
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

    def __init__(self, config: Optional[BaseAgentConfig] = None):
        config = config or BaseAgentConfig(
            client=instructor.from_openai(
                openai.OpenAI(api_key=CONFIG.model_config_.api_keys.openai),
            ),
            model=CONFIG.model_config_.model_name,
            temperature=0.0,
        )
        super().__init__(config)
