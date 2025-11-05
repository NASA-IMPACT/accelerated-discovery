from enum import Enum

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent


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


class IntentAgent(LiteLLMInstructorBaseAgent[IntentInputSchema, IntentOutputSchema]):
    """Intent Detector Agent"""

    input_schema = IntentInputSchema
    output_schema = IntentOutputSchema


class ScienceDivision(str, Enum):
    EARTH_SCIENCE = "Earth Science Division"
    PLANETARY_SCIENCE = "Planetary Science Division"
    ASTROPHYSICS = "Astrophysics Division"
    HELIOPHYSICS = "Heliophysics Division"
    BIOLOGICAL_PHYSICAL_SCIENCES = "Biological and Physical Sciences Division"
    UNKNOWN = "Unknown"


class DivisionInputSchema(InputSchema):
    """Input schema for determining NASA Science division of the query"""

    query: str = Field(..., description="The user's latest query/message/question")


class DivisionOutputSchema(OutputSchema):
    """Output schema represents the NASA Science division of the query"""

    division: ScienceDivision = Field(
        ...,
        description="The NASA Science division that the query belongs to. If the query does not belong to any of the divisions, return UNKNOWN.",
    )
    reasoning: str = Field(..., description="The reasoning for the division classification")


class DivisionAgent(LiteLLMInstructorBaseAgent[DivisionInputSchema, DivisionOutputSchema]):
    """Division Classifier Agent for NASA Science"""

    input_schema = DivisionInputSchema
    output_schema = DivisionOutputSchema
