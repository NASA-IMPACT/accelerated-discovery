from pydantic import Field
from typing import List

from akd.agents._base import BaseAgentConfig, BaseAgent
from akd._base import InputSchema, OutputSchema

from data_scout_agent import DataScoutAgent, DataScoutAgentConfig, DataScoutAgentInputSchema, DataScoutAgentOutputSchema
from script_writer_agent import ScriptWriterAgent, ScriptWriterAgentConfig, ScriptWriterAgentInputSchema, ScriptWriterAgentOutputSchema
from data_injection_agent import DataInjectionAgent, DataInjectionAgentConfig, DataInjectionAgentInputSchema, DataInjectionAgentOutputSchema
from mdx_builder_agent import MDXBuilderAgent, MDXBuilderAgentConfig, MDXBuilderAgentInputSchema, MDXBuilderAgentOutputSchema

from helper import scrape_text_from_urls, get_collection_items
from typing import List
from data_types import CollectionItem

class StoryTellerAgentInputSchema(InputSchema):
  """
    This is the input to the Story Teller Agent.
    The story teller agent gets the urls of the relevant scientific literature and relevant STAC collection ids.
  """
  urls: List[str] = Field(
    ...,
    description="""
      The urls of the relevant scientific literature.
    """
  )
  collection_ids: List[str] = Field(
    ...,
    description="""
      The collection ids of the relevant STAC collection items.
    """
  )

class StoryTellerAgentOutputSchema(OutputSchema):
  """
    This is the output of the Story Teller Agent.
    The story teller agent returns the story in mdx format.
  """
  story_mdx: str = Field(
    ...,
    description="""
      The story in mdx format.
    """
  )
  
class StoryTellerAgentConfig(BaseAgentConfig):
  pass

class StoryTellerAgent(BaseAgent):
  input_schema = StoryTellerAgentInputSchema
  output_schema = StoryTellerAgentOutputSchema
  config_schema = StoryTellerAgentConfig

  def __init__(
    self,
    config: StoryTellerAgentConfig | None = None,
    debug: bool = False,
    data_scout_agent: DataScoutAgent | None = None,
    script_writer_agent: ScriptWriterAgent | None = None,
    data_injection_agent: DataInjectionAgent | None = None,
    mdx_builder_agent: MDXBuilderAgent | None = None
  ):
    super().__init__(config=config or StoryTellerAgentConfig(), debug=debug)
    self.data_scout_agent = data_scout_agent or DataScoutAgent(DataScoutAgentConfig(api_key=config.api_key))
    self.script_writer_agent = script_writer_agent or ScriptWriterAgent(ScriptWriterAgentConfig(api_key=config.api_key))
    self.data_injection_agent = data_injection_agent or DataInjectionAgent(DataInjectionAgentConfig(api_key=config.api_key))
    self.mdx_builder_agent = mdx_builder_agent or MDXBuilderAgent(MDXBuilderAgentConfig(api_key=config.api_key))

  async def get_response_async(self, params: StoryTellerAgentInputSchema) -> StoryTellerAgentOutputSchema:
    # extract the text from urls and get the collection items
    urls: List[str] = params.urls
    collection_items: List[CollectionItem] = get_collection_items(params.collection_ids)
    scraped_text: str = await scrape_text_from_urls(urls)

    # get the relevant collection items
    try:
      data_scout_input: DataScoutAgentInputSchema = DataScoutAgentInputSchema(
        literature=scraped_text,
        collection_items=collection_items
      )
      data_scout_output: DataScoutAgentOutputSchema = await self.data_scout_agent.arun(data_scout_input)
    except Exception as e:
      raise Exception(f"DataScoutAgent failed: {str(e)}") from e

    # get the story script
    try:
      script_writer_input = ScriptWriterAgentInputSchema(
        literature_text=scraped_text,
        collection_items=data_scout_output.relevant_collection
      )
      script_writer_output: ScriptWriterAgentOutputSchema = await self.script_writer_agent.arun(script_writer_input)
    except Exception as e:
      raise Exception(f"ScriptWriterAgent failed: {str(e)}") from e

    # inject the data into the script
    try:
      data_injection_input = DataInjectionAgentInputSchema(
        collection_items=data_scout_output.relevant_collection,
        script=script_writer_output.script
      )
      data_injection_output: DataInjectionAgentOutputSchema = await self.data_injection_agent.arun(data_injection_input)
    except Exception as e:
      raise Exception(f"DataInjectionAgent failed: {str(e)}") from e

    # build the mdx
    try:
      mdx_builder_input = MDXBuilderAgentInputSchema(
        story_script=data_injection_output.script_with_data,
      )
      mdx_builder_output: MDXBuilderAgentOutputSchema = await self.mdx_builder_agent.arun(mdx_builder_input)
    except Exception as e:
      raise Exception(f"MDXBuilderAgent failed: {str(e)}") from e

    return StoryTellerAgentOutputSchema(story_mdx=mdx_builder_output.story_mdx)

  async def _arun(self, params: StoryTellerAgentInputSchema) -> StoryTellerAgentOutputSchema:
    result: StoryTellerAgentOutputSchema = await self.get_response_async(params)
    return result

