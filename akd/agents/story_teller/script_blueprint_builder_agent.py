from pydantic import Field
from typing import List

from akd.agents._base import BaseAgent, BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import CollectionItem

from akd.configs.storyteller_prompts import SCRIPT_BLUEPRINT_BUILDER_SYSTEM_PROMPT

class ScriptBlueprintBuilderAgentInputSchema(InputSchema):
  """
  This is the input to the Script Builder Agent.
  The literature text and STAC based collection_items have correlation.
  """
  literature_text: str = Field(
    ...,
    description="""
      The relevant scientific literature text.
    """
  )
  collection_items: List[CollectionItem] = Field(
    ...,
    description="""
      This is the list of relevant STAC based collection-items.
      The description, location and date information are factual.
      It is best to base the backbone of the story based off these collection_items.
    """
  )

class ScriptBlueprintBuilderAgentOutputSchema(OutputSchema):
  """
  The output of Script Builder Agent. 
  """
  script_blueprint: str = Field(
    ...,
    description="""
      A blueprint for the script which will be built into a story.
    """
  )

class ScriptBlueprintBuilderAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default=SCRIPT_BLUEPRINT_BUILDER_SYSTEM_PROMPT
  )
  input_hints: bool = Field(default=True)
  enable_trimming: bool = Field(default=False)
  model_name: str = Field(default="gpt-4o")
  temperature: float = Field(default=1.0)

class ScriptBlueprintBuilderAgent(LiteLLMInstructorBaseAgent[ScriptBlueprintBuilderAgentInputSchema, ScriptBlueprintBuilderAgentOutputSchema]):
  """
  This script blueprint builder agent is used to generate a script blueprint for a story.
  The inputs are literature text and STAC based collection_items.
  Always base the script blueprint off factual collection-items datasets whereever possible.
 """
  input_schema = ScriptBlueprintBuilderAgentInputSchema
  output_schema = ScriptBlueprintBuilderAgentOutputSchema
  config_schema = ScriptBlueprintBuilderAgentConfig
