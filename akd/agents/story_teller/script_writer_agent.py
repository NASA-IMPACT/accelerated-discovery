from pydantic import Field
from typing import List

from akd.agents._base import BaseAgentConfig, BaseAgent
from akd._base import InputSchema, OutputSchema

from script_blueprint_builder_agent import ScriptBlueprintBuilderAgent, ScriptBlueprintBuilderAgentConfig, ScriptBlueprintBuilderAgentInputSchema, ScriptBlueprintBuilderAgentOutputSchema
from script_builder_agent import ScriptBuilderAgent, ScriptBuilderAgentConfig, ScriptBuilderAgentInputSchema, ScriptBuilderAgentOutputSchema

from data_types import CollectionItem

class ScriptWriterAgentInputSchema(InputSchema):
  """
  This is the input to the Script Writer Agent.
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

class ScriptWriterAgentOutputSchema(OutputSchema):
  """
    The generated story
  """
  script: str = Field(
    ...,
    description="""
      The story markdown that is generated from the scientific literature text and STAC-based collection-items context. 
    """
  )

class ScriptWriterAgentConfig(BaseAgentConfig):
  pass

class ScriptWriterAgent(BaseAgent):
  input_schema = ScriptWriterAgentInputSchema
  output_schema = ScriptWriterAgentOutputSchema
  config_schema = ScriptWriterAgentConfig

  def __init__(
    self,
    config: ScriptWriterAgentConfig | None = None,
    debug: bool = False,
    script_blueprint_builder_agent: ScriptBlueprintBuilderAgent | None = None,
    script_builder_agent: ScriptBuilderAgent | None = None
  ):
    super().__init__(config=config or ScriptWriterAgentConfig(), debug=debug)
    self.script_blueprint_builder_agent = script_blueprint_builder_agent or ScriptBlueprintBuilderAgent(ScriptBlueprintBuilderAgentConfig(api_key=config.api_key))
    self.script_builder_agent = script_builder_agent or ScriptBuilderAgent(ScriptBuilderAgentConfig(api_key=config.api_key))
  
  async def get_response_async(self, params: ScriptWriterAgentInputSchema) -> ScriptWriterAgentOutputSchema:
    script_blueprint_input: ScriptBlueprintBuilderAgentInputSchema = ScriptBlueprintBuilderAgentInputSchema(
      literature_text=params.literature_text,
      collection_items=params.collection_items
    )
    script_blueprint_output: ScriptBlueprintBuilderAgentOutputSchema = await self.script_blueprint_builder_agent.arun(script_blueprint_input)

    script_builder_input = ScriptBuilderAgentInputSchema(
      narrative_blueprint=script_blueprint_output.script_blueprint
    )
    script_builder_output: ScriptBuilderAgentOutputSchema = await self.script_builder_agent.arun(script_builder_input)
    return ScriptWriterAgentOutputSchema(script=script_builder_output.story_draft)
  
  async def _arun(self, params: ScriptWriterAgentInputSchema) -> ScriptWriterAgentOutputSchema:
    result: ScriptWriterAgentOutputSchema = await self.get_response_async(params)
    # TODO: use the critique-evaluator-agent and if not satisfied, rerun the _get_relevant_data once again with feedback
    return result
