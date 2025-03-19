from enum import Enum

from atomic_agents.agents.base_agent import BaseAgent, BaseIOSchema
from pydantic import Field


class Intent(str, Enum):
    GENERAL = "General"
    ESTIMATION = "Estimation"
    # DATA_DISCOVERY = "Data Discovery"


class IntentInputSchema(BaseIOSchema):
    """Input schema for determining intent of the query"""

    query: str = Field(..., description="The user's latest query/message/question")


class IntentOutputSchema(BaseIOSchema):
    """Output schema represents the intent of the query"""

    intent: Intent = Field(..., description="The user's intent")


class IntentAgent(BaseAgent):
    """Intent Detector Agent"""

    input_schema = IntentInputSchema
    output_schema = IntentOutputSchema
