from enum import Enum

from pydantic import BaseModel, Field

from ._base import BaseAgent


class Intent(str, Enum):
    GENERAL = "General"
    ESTIMATION = "Estimation"
    # DATA_DISCOVERY = "Data Discovery"


class IntentInputSchema(BaseModel):
    """Input schema for determining intent of the query"""

    query: str = Field(..., description="The user's latest query/message/question")


class IntentOutputSchema(BaseModel):
    """Output schema represents the intent of the query"""

    intent: Intent = Field(..., description="The user's intent")


class IntentAgent(BaseAgent):
    """Intent Detector Agent"""

    input_schema = IntentInputSchema
    output_schema = IntentOutputSchema
