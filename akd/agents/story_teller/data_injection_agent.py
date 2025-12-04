from pydantic import Field
from typing import List

from akd.agents._base import BaseAgent, BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import CollectionItem

class DataInjectionAgentInputSchema(InputSchema):
  """
    The valid collection-items that can be added to the script. 
  """
  script: str = Field(
    ...,
    description="""
      The script for the story.
    """
  )
  collection_items: List[CollectionItem] = Field(
    ...,
    description="""
      The relevant list of collection items.
      The relevancy is based off the literature context.
      - The relevant dataset will have either location or time factor somewhere mentioned in the literature.
      - The relevant dataset will have the description matching in the literature.
    """
  )

class DataInjectionAgentOutputSchema(OutputSchema):
  """
    The script with dataset tags.
  """
  script_with_data: str = Field()

class DataInjectionAgentConfig(BaseAgentConfig):
  pass

class DataInjectionAgent(LiteLLMInstructorBaseAgent[DataInjectionAgentInputSchema, DataInjectionAgentOutputSchema]):
  """
    You are a science matter expert specializing in STAC.
    Your goal is to read the script for the story, go through the available collection_items
    and figure out the datasets that can be used with in the story.
    The different catagories of data that can be added are:
      1. simple map block: It expects collection_id, item_id, and datetime
      2. compare map block: It expects two valid datetime to compare against. The collection_id and, item_id should be same.
      3. chapters map block: It expects a list of valid collection_id, item_id. The chapters are used to link similar datasets and showcase them together.
    From the list of available collection items descriptions, figure out the relevant collection items that adds up to the script.
    And in the script, add the expectations inside a xml.
    for example:
      <SimpleMapBlock>
        <CollectionId>value</CollectionId>
        <ItemId>value</ItemId>
        <Datetime>value</Datetime>
      </SimpleMapBlock>   
  """
  input_schema = DataInjectionAgentInputSchema
  output_schema = DataInjectionAgentOutputSchema
  config_schema = DataInjectionAgentInputSchema
