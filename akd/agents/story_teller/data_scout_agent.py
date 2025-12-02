from pydantic import Field
from data_types import SearchedSTACData
from typing import List

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

class RelevantDataFilterAgentInputSchema(InputSchema):
  """ Description """
  scraped_text: str = Field(
    ...,
    description="The text from earth data literatures.")
  searched_stac_data: SearchedSTACData = Field(
    ...,
    description="""
      The list of STAC collections where each collection contains items.
      The items has description, spatial and temporal resolution.
    """)  

class ComplexRelevantDataFilterAgentOutputSchema(OutputSchema):
  """ Description """
  relevant_stac_data: SearchedSTACData = Field(
    ..., 
    description="""
      The list of relevant STAC collections where each collection contains items.
      The items in the collection are relevant to the scraped literature text.
      - If there is no relevancy, reject the collection items.
      - If there are no relevant items in the collection, reject the collection.
    """)

class RelevantDataFilterAgentOutputSchema(OutputSchema):
    """Simplified output schema."""
    relevant_collection_ids: List[str] = Field(
        description="List of relevant STAC collection IDs"
    )
    relevant_item_ids: List[str] = Field(
        description="List of relevant STAC item IDs"
    )
    reasoning: str = Field(
        description="Explanation of why these items are relevant"
    )

class RelevantDataFilterAgentConfig(BaseAgentConfig):
  pass

class RelevantDataFilterAgent(LiteLLMInstructorBaseAgent[RelevantDataFilterAgentInputSchema, RelevantDataFilterAgentOutputSchema]):
  """
  This Agent only selects the relevant STAC datasets.
  The relevancy is based off the checks on the items, 
  i.e. if the item description, item temporal resolution, item spatial resolution
  are relevant based off the scraped literature text
  """
  input_schema = RelevantDataFilterAgentInputSchema
  output_schema = RelevantDataFilterAgentOutputSchema
  config = RelevantDataFilterAgentConfig


