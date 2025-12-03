from pydantic import Field
from data_types import SearchedSTACData
from typing import List
from copy import deepcopy

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import STACCollection

class DataRelationshipBuilderAgentInputSchema(InputSchema):
  """ Description """
  stac_data: SearchedSTACData = Field(
    ...,
    description="""
      The list of relevant STAC collections where each collection contains items.
      The items has description, spatial and temporal resolution.
    """)  

class DataRelationshipBuilderAgentOutputSchema(OutputSchema):
  """ Description """
  relevant_stac_data: SearchedSTACData = Field(
    ..., 
    description="""
      The list of relevant STAC collections where each collection contains items.
      The items in the collection are relevant to the scraped literature text.
      - If there is no relevancy, reject the collection items.
      - If there are no relevant items in the collection, reject the collection.
    """)

class DataRelationshipBuilderAgentConfig(BaseAgentConfig):
  pass

class DataRelationshipBuilderAgent(LiteLLMInstructorBaseAgent[DataRelationshipBuilderAgentInputSchema, DataRelationshipBuilderAgentOutputSchema]):
  """
  This Agent only selects the relevant STAC datasets.
  The relevancy is based off the checks on the items, 
  i.e. if the item description, item temporal resolution, item spatial resolution
  are relevant based off the scraped literature text
  """
  input_schema = DataRelationshipBuilderAgentInputSchema
  output_schema = DataRelationshipBuilderAgentOutputSchema
  config = DataRelationshipBuilderAgentConfig
