from typing import Optional

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig
from akd.agents.eie_agents.eie_extraction import TemporalExtent


class CollectionSearchInputSchema(InputSchema):
    """Input schema for Collection Search agent"""

    dataset_type: str = Field(..., description="Dataset type extracted from user query.")
    location: str = Field(..., description="Location extracted from user query.")
    frequency: str = Field(..., description="Frequency extracted from user query.")
    temporal_extent: Optional[TemporalExtent] = Field(
        default=None,
        description="Time interval (start and end dates) extracted from the query.",
    )


class CollectionSearchOutputSchema(OutputSchema):
    """Output schema for Collection Search agent"""

    response: str = Field(
        default=None,
        description="Response with the results of matching STAC collections with metadata.",
    )


class CollectionSearchAgentConfig(BaseAgentConfig):
    """Config  for Collection Search agent"""

    stac_root: str = Field(default="http://dev.ghg.center/api/stac", description="Base STAC API endpoint.")


class CollectionSearchAgent(BaseAgent):
    input_schema = CollectionSearchInputSchema
    output_schema = CollectionSearchOutputSchema
    config_schema = CollectionSearchAgentConfig

    async def _arun(self, params: CollectionSearchInputSchema, **kwargs) -> CollectionSearchOutputSchema:
        return await self.get_response_async(params, **kwargs)

    async def get_response_async(self, params: CollectionSearchInputSchema, **kwargs) -> CollectionSearchOutputSchema:
        """Filter STAC collections based on extracted parameters."""
        stac_root = self.config.stac_root.rstrip("/")
        try:
            # --- Step 1: Fetch collections ---
            with httpx.Client(timeout=30.0) as c:
                r = c.get(f"{stac_root}/collections")
                r.raise_for_status()
                data = r.json()

            collections = data.get("collections", [])
            tmp = [
                {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "description": c.get("description", ""),
                    "extent": c.get("extent", ""),
                }
                for c in collections
            ]

            # --- Step 2: Let LLM select the most relevant collections ---
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500, api_key=self.config.api_key)

            llm_prompt = f"""
            You are a geospatial data expert.
            The user is interested in datasets matching:
              - dataset_type: {params.dataset_type}
              - location: {params.location}
              - frequency: {params.frequency}

            Below is a list of available STAC collections (truncated to essentials):

            {tmp}

            Return the top 5 most relevant collections in JSON list form, like:
            [
              {{"id": "...", "title": "...", "description": "..."}}
            ]
            """

            response = await llm.ainvoke(llm_prompt)
            return CollectionSearchOutputSchema(response=response.content)

        except Exception as e:
            logger.error(f"Error in CollectionSearchAgent: {e}")
            return CollectionSearchOutputSchema(collections=[])
