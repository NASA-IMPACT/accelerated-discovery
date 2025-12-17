from pydantic import Field
from typing import List
from data_types import CollectionItem

from akd.agents._base import BaseAgentConfig, BaseAgent
from akd._base import InputSchema, OutputSchema

from relevant_data_filter_agent import RelevantDataFilterAgent, RelevantDataFilterAgentInputSchema, RelevantDataFilterAgentOutputSchema, RelevantDataFilterAgentConfig
from data_relationship_builder_agent import DataRelationshipBuilderAgent, DataRelationshipBuilderAgentInputSchema, DataRelationshipBuilderAgentOutputSchema, DataRelationshipBuilderAgentConfig

class DataScoutAgentInputSchema(InputSchema):
  """
  The input schema contains of the whole relevant literature context, and the collection items.
  The collection items represent and closely resembles the STAC collection items.
  """
  literature: str = Field(..., description="The literature text which is extracted from the relevant literature sources.")
  collection_items: List[CollectionItem] = Field(
    ..., 
    description="""
      The collection items resembles the list of STAC collection item.
      The collection item is a subset of STAC collection item but with additional information about the collectio.
      It basically contains the collection description, item description, titles, spatial and temporal information. 
    """
  )

class DataScoutAgentOutputSchema(OutputSchema):
  """
  The relevant collection are the filtered collections based off the literature context.
  """
  relevant_collection: List[CollectionItem] = Field(
    ...,
    description="""
      The relevant list of collection items.
      The relevancy is based off the literature context.
      - The relevant dataset will have either location or time factor somewhere mentioned in the literature.
      - The relevant dataset will have the description matching in the literature.
    """
  )

class DataScoutAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default="""
      You are a Agent who Scouts through the data (collection item) and finds out relevant data (evidences).
      The context is the literature text.
      Your job is to find the dataset which has a match to the events, locations and time described in the literature.
      It's okay if there are no relevant data. Its actually better than having non-relevant data as data evidence.
    """
  )

class DataScoutAgent(BaseAgent):
  """
  The Data scout agent filters the relevant data and then produces evidences/relevant_collection.
  Note: This agent is just a orcheastrator, the configuration is meaningless here. It doesnot use any LLM calls.
  So the system calls, doc string for the io schema are not used.
  """
  input_schema = DataScoutAgentInputSchema
  output_schema = DataScoutAgentOutputSchema
  config_schema = DataScoutAgentConfig
  
  def __init__(
    self,
    config: DataScoutAgentConfig | None = None,
    relevant_data_filter_agent: RelevantDataFilterAgent | None = None,
    data_relationship_builder_agent: DataRelationshipBuilderAgent | None = None,
    debug: bool = False
  ):
    super().__init__(config=config or DataScoutAgentConfig(), debug=debug)
    # TODO: Check a better way of passing api keys to the subagents
    self.relevant_data_filter_agent = relevant_data_filter_agent or RelevantDataFilterAgent(RelevantDataFilterAgentConfig(api_key=config.api_key))
    self.data_relationship_builder_agent = data_relationship_builder_agent or DataRelationshipBuilderAgent(DataRelationshipBuilderAgentConfig(api_key=config.api_key))
  
  async def _get_relevant_data(self, data: RelevantDataFilterAgentInputSchema) -> RelevantDataFilterAgentOutputSchema:
    relevant_stac_agent_result: RelevantDataFilterAgentOutputSchema = await self.relevant_data_filter_agent.arun(data)
    return relevant_stac_agent_result

  async def _get_data_evidence(self, data: DataRelationshipBuilderAgentInputSchema) -> DataRelationshipBuilderAgentOutputSchema:
    output: DataRelationshipBuilderAgentOutputSchema = self.data_relationship_builder_agent.arun(data)
    return output

  async def get_response_async(self, params: DataScoutAgentInputSchema) -> DataScoutAgentOutputSchema:
    relevant_agent_inputs = RelevantDataFilterAgentInputSchema(
      literature_context=params.literature,
      stac_data=params.collection_items
    )
    relevant_data: RelevantDataFilterAgentOutputSchema = await self._get_relevant_data(relevant_agent_inputs)
    
    data_relationship_builder_agent_input = DataRelationshipBuilderAgentInputSchema(
      stac_data=relevant_data.stac_data
    )
    data_evidence: DataRelationshipBuilderAgentOutputSchema = await self._get_data_evidence(data_relationship_builder_agent_input)

    return DataScoutAgentOutputSchema(
      relevant_collection=data_evidence.stac_data
    )

  async def _arun(self, params: DataScoutAgentInputSchema) -> DataScoutAgentOutputSchema:
    result: DataScoutAgentOutputSchema = await self.get_response_async(params)
    return result
