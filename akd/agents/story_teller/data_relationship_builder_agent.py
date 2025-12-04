from pydantic import Field
from data_types import CollectionItem
from typing import List
from copy import deepcopy

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

class DataRelationshipBuilderAgentInputSchema(InputSchema):
  """ The relevant collection items."""
  stac_data: List[CollectionItem] = Field(
    ...,
    description="""
      The list of relevant collections items.
    """)

class DataRelationshipBuilderAgentOutputSchema(OutputSchema):
  """ Description """
  stac_data: List[CollectionItem] = Field(
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
  config_schema = DataRelationshipBuilderAgentConfig
  
  def __init__(
    self,
    config: DataRelationshipBuilderAgentConfig | None = None,
    debug: bool = False
  ):
    super().__init__(config=config or DataRelationshipBuilderAgentConfig(), debug=debug)
  
  def arun(self, input: DataRelationshipBuilderAgentInputSchema) -> DataRelationshipBuilderAgentOutputSchema:
    """
      Note: for now, it seems that the relevant data filter needs no further breakdown.
      However, the relevant data could be further clustered into multiple categories.
      1. comparable.
      2. sequential: for scrolly telling chapters
      3. standalone: for map display. However, this standalone's output is basically the whole
      This empty relay agent is to complete the pipeline. This implementation could be later changed for experimentations.  
    """
    return DataRelationshipBuilderAgentOutputSchema(stac_data=input.stac_data)
