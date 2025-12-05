from pydantic import Field
from typing import List

from akd.agents._base import BaseAgent, BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import CollectionItem

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
      A blueprint containing the Characters, the inciting incident (from the data), and the resolution.
    """
  )

class ScriptBlueprintBuilderAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default="""
      You are a Narrative Architect.
      Find the primary event being discussed and decide the Angle of the story.
      Your goal is to find the 'Drama' inside technical or factual documents.
    """
  )

class ScriptBlueprintBuilderAgent(LiteLLMInstructorBaseAgent[ScriptBlueprintBuilderAgentInputSchema, ScriptBlueprintBuilderAgentOutputSchema]):
  """
    Always base things off factual collection-items datasets whereever possible.
    If the input is about co2 emission literature: The 'Story' is the showcasing the risk, the current scenario and providing the proposal for mitigation.
  """
  input_schema = ScriptBlueprintBuilderAgentInputSchema
  output_schema = ScriptBlueprintBuilderAgentOutputSchema
  config_schema = ScriptBlueprintBuilderAgentConfig
