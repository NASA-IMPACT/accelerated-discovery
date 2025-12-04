from pydantic import Field
from data_types import SearchedSTACData
from typing import List
from copy import deepcopy

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import CollectionItem

class RelevantDataFilterAgentInputSchema(InputSchema):
  """ Description """
  literature_context: str = Field(
    ...,
    description="The text from earth data literatures.")
  stac_data: List[CollectionItem] = Field(
    ...,
    description="""
      The list of STAC collections where each collection contains items.
      The items has description, spatial and temporal resolution.
    """)

class RelevantDataFilterAgentOutputSchema(OutputSchema):
  """The relevant output data."""
  stac_data: List[CollectionItem] = Field(
    ...,
    description="""
       This represents the list of relevant collections items.
       These collection items are relevant to the scraped literature text.
       Relevancy is dependent on:
        - spatial or location relevancy. i.e. If the collection item's location is available in literature text, it's relevant.
        - temporal or time based relevancy. i.e. If the collection item's time is available in literature text, it's relevant.
       Note:
       - If there is no relevancy, reject the collection item.
       - Think in steps with reasoning.
    """
  )

class RelevantDataFilterAgentConfig(BaseAgentConfig):
  pass

class RelevantDataFilterAgent(LiteLLMInstructorBaseAgent[RelevantDataFilterAgentInputSchema, RelevantDataFilterAgentOutputSchema]):
  """
  You are a Agent who filters Relevant Data.
  The context is the literature text. Say, you are given a list of STAC Items, find a match to the events described in the literature.
  Think in steps when finding the match.
  - If there is a match, the STAC item is relevant.
  - If there is no match, reject the STAC item.
  It's okay if there are no relevant data. Actually, its better than having non relevant data.
  """
  input_schema = RelevantDataFilterAgentInputSchema
  output_schema = RelevantDataFilterAgentOutputSchema
  config_schema = RelevantDataFilterAgentConfig

async def get_relevant_data(api_key, literature_context: str, stac_data: List[CollectionItem]) -> List[CollectionItem]:
  inputs = RelevantDataFilterAgentInputSchema(
    literature_context=literature_context,
    stac_data=stac_data
  )
  config = RelevantDataFilterAgentConfig(
      model_name = "gpt-4o-mini",
      api_key=api_key,
      temperature = 0.3
  )
  relevant_stac_agent: RelevantDataFilterAgent = RelevantDataFilterAgent(config=config, debug=True)
  relevant_stac_agent_result: RelevantDataFilterAgentOutputSchema = await relevant_stac_agent.arun(inputs)
  # TODO: break down the implementation to check each stac_data with respect to literature.

  return relevant_stac_agent_result.stac_data

