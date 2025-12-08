from pydantic import Field

from akd.agents._base import LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema
from akd.configs.storyteller_prompts import MDX_BUILDER_SYSTEM_PROMPT

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
