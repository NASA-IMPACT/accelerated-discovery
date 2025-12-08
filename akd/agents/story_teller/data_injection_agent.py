from pydantic import Field
from typing import List

from akd.agents._base import BaseAgent, BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema
from akd.configs.storyteller_prompts import DATA_INJECTION_AGENT_SYSTEM_PROMPT

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
    The script with relevant data.
  """
  script_with_data: str = Field()

class DataInjectionAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default=DATA_INJECTION_AGENT_SYSTEM_PROMPT
  )
  input_hints: bool = Field(default=True)
  enable_trimming: bool = Field(default=False)
  temperature: float = Field(default=1.0)

class DataInjectionAgent(LiteLLMInstructorBaseAgent[DataInjectionAgentInputSchema, DataInjectionAgentOutputSchema]):
  """
    You are a data injection agent. Your job is to inject the relevant data into the script.
  """
  input_schema = DataInjectionAgentInputSchema
  output_schema = DataInjectionAgentOutputSchema
  config_schema = DataInjectionAgentConfig
