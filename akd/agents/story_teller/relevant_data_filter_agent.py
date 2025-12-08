from pydantic import Field
from data_types import SearchedSTACData
from typing import List
from copy import deepcopy

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema
from akd.configs.storyteller_prompts import RELEVANT_DATA_FILTER_AGENT_SYSTEM_PROMPT

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
      The list of STAC collections where each collection contains items.
      The items has description, spatial and temporal resolution.
    """
  )

class RelevantDataFilterAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default=RELEVANT_DATA_FILTER_AGENT_SYSTEM_PROMPT
  )
  model_name: str = Field(default="gpt-4o")
  input_hints: bool = Field(default=True)
  enable_trimming: bool = Field(default=False)
  temperature: float = Field(default=1.0)

class RelevantDataFilterAgent(LiteLLMInstructorBaseAgent[RelevantDataFilterAgentInputSchema, RelevantDataFilterAgentOutputSchema]):
  """
  You are a Agent who filters Relevant Data using the context of the literature text.
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
      api_key=api_key,      
  )
  relevant_stac_agent: RelevantDataFilterAgent = RelevantDataFilterAgent(config=config, debug=True)
  relevant_stac_agent_result: RelevantDataFilterAgentOutputSchema = await relevant_stac_agent.arun(inputs)
  # TODO: break down the implementation to check each stac_data with respect to literature??
  # How much better will the performance be?

  return relevant_stac_agent_result.stac_data

