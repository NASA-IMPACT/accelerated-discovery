from enum import Enum

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent


class Intent(str, Enum):
    GENERAL = "General"
    ESTIMATION = "Estimation"
    # DATA_DISCOVERY = "Data Discovery"


class IntentInputSchema(InputSchema):
    """Input schema for determining intent of the query"""

    query: str = Field(..., description="The user's latest query/message/question")


class IntentOutputSchema(OutputSchema):
    """Output schema represents the intent of the query"""

    intent: Intent = Field(..., description="The user's intent")


class IntentAgent(InstructorBaseAgent[IntentInputSchema, IntentOutputSchema]):
    """Intent Detector Agent"""

    input_schema = IntentInputSchema
    output_schema = IntentOutputSchema
