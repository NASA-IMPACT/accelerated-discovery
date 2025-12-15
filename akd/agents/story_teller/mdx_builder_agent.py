from pydantic import Field
from typing import List, Tuple, Optional

from akd.agents._base import LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema
from akd.configs.storyteller_prompts import MDX_BUILDER_SYSTEM_PROMPT
from helper import MDXValidator

class MDXBuilderAgentInputSchema(InputSchema):
  """
    The inputs for the MDX replacer agent 
  """
  story_script: str = Field(
    ...,
    description="""
      The script for a story to be poseted in the webpage.
      The dataset is available in XML.
      There are usage of HTML tags and XML tags in the story script.
    """
  )
  feedback: Optional[List[Tuple[str, int]]] = Field(
    default=None,
    description="""
      The feedback for the MDX Builder agent. This is the output of a MDX validator.
      The MDX validator output is a list of tuple containing:
        - list[tuple[str, int]]: List of unpaired tags with their positions. It's Empty if valid.
          Each tuple contains (tag_string, position_index) where position is the
          enumerated index of the tag in the sequence.
      
      If available, use this unpaired tags list to generate the set of MDX components.
    """
  )
  previous_generated_mdx: Optional[str] = Field(
    default=None,
    description="""
      The previous MDX output from the MDX Builder agent.
      If available, use this as the base for the output along with the feedback.
    """
  )

class MDXBuilderAgentOutputSchema(OutputSchema):
  """
  The MDX output from the MDX Builder agent.
  """
  story_mdx: str = Field(
    ...,
    description="""
      The story in mdx frontmatter format with appropriate veda MDX components.
    """
  )

class MDXBuilderAgentConfig(BaseAgentConfig):
  system_prompt: str = Field(
    default=MDX_BUILDER_SYSTEM_PROMPT
  )
  input_hints: bool = Field(default=True)
  enable_trimming: bool = Field(default=False)
  model_name: str = Field(default="gpt-4o")
  temperature: float = Field(default=1.0)

class MDXBuilderAgent(LiteLLMInstructorBaseAgent[MDXBuilderAgentInputSchema, MDXBuilderAgentOutputSchema]):
  """
    You are a mdx builder agent expert in frontmatter mdx.
    Your goal is to replace the html tags, xml related to data in story script and then replace that 
    with the appropriate veda MDX components. 
  """
  input_schema = MDXBuilderAgentInputSchema
  output_schema = MDXBuilderAgentOutputSchema
  config_schema = MDXBuilderAgentConfig

  async def _arun(self, input_data: MDXBuilderAgentInputSchema) -> MDXBuilderAgentOutputSchema:
    agent_output: MDXBuilderAgentOutputSchema = await super()._arun(input_data)

    mdx_validator: MDXValidator = MDXValidator()
    valid, unpaired_tags = mdx_validator.validate_mdx(agent_output.story_mdx)
    
    while not valid:
      input_data.previous_generated_mdx = agent_output.story_mdx
      input_data.feedback = unpaired_tags
      agent_output: MDXBuilderAgentOutputSchema = await super()._arun(input_data)
      mdx_validator: MDXValidator = MDXValidator()
      valid, unpaired_tags = mdx_validator.validate_mdx(agent_output.story_mdx)

    return agent_output
