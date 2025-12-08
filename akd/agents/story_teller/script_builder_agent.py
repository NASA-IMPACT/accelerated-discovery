from pydantic import Field

from akd.agents._base import BaseAgent, LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema
from akd.configs.storyteller_prompts import SCRIPT_BUILDER_SYSTEM_PROMPT

class ScriptBuilderAgentInputSchema(InputSchema):
  """
    The script blueprint to base the story off.  
  """
  narrative_blueprint: str = Field(
    ...,
    description="""
      A story blueprint containing the flow and major parts of the story.
    """
  )

class ScriptBuilderAgentOutputSchema(OutputSchema):
  """
    The elaborated story draft.
  """
  story_draft: str = Field(
    ...,
    description="""
      The story that is generated from the blueprint.
    """
  )

class ScriptBuilderAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default=SCRIPT_BUILDER_SYSTEM_PROMPT
  )
  input_hints: bool = Field(default=True)
  enable_trimming: bool = Field(default=False)
  temperature: float = Field(default=1.0)

class ScriptBuilderAgent(LiteLLMInstructorBaseAgent[ScriptBuilderAgentInputSchema, ScriptBuilderAgentOutputSchema]):
  """
    You are a script builder agent.
    Your goal is to generate a script from the script blueprint.
  """
  input_schema = ScriptBuilderAgentInputSchema
  output_schema = ScriptBuilderAgentOutputSchema
  config_schema = ScriptBuilderAgentConfig
