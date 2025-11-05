from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig


class ExtractInputSchema(InputSchema):
    """Input schema for Extract agent"""

    query: str = Field(
        ...,
        description="Query from the user.",
    )


class ExtractOutputSchema(OutputSchema):
    """Output schema for the Extract agent."""

    dataset_type: str = Field(
        default="all",
        description=(
            "The dataset type extracted by the agent based on the query, e.g., methane, population, nitrogen dioxide."
        ),
    )

    location: str = Field(
        default="global",
        description="Human-readable location or place extracted by the agent based on the query.",
    )

    bbox: Optional[str] = Field(
        default=None,
        description="A GeoJSON string representing the bounding box coordinates of the location.",
    )

    frequency: str = Field(
        default="all",
        description="The periodicity or frequency of the dataset requested (e.g., daily, monthly, yearly, all).",
    )


class ExtractAgentConfig(BaseAgentConfig):
    """Configuration for Extract Agent"""


class ExtractAgent(BaseAgent):
    input_schema = ExtractInputSchema
    output_schema = ExtractOutputSchema
    config_schema = ExtractAgentConfig

    def __init__(self, config: ExtractAgentConfig | None = None, debug: bool = False):
        super().__init__(config=config, debug=debug)

    async def get_response_async(
        self,
        params: ExtractInputSchema,
        **kwargs,
    ) -> ExtractOutputSchema:
        """
        Obtains a response from the language model asynchronously. Makes a llm call with the query to extract dataset_type, location, bbox and frequency from the query.
        """
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            api_key=self.config.api_key,
        )

        prompt = f"""
            You are an expert in geospatial data extraction.
            Given a user query, extract the following structured fields:

            - dataset_type: the category of dataset (e.g. methane, population, nitrogen dioxide)
            - location: the place or region (human-readable)
            - bbox: the bounding box of the location in GeoJSON format (use None if unknown)
            - frequency: the temporal frequency (e.g. daily, monthly, yearly, all)

            Respond strictly in JSON format, like this:
            {{
            "dataset_type": "...",
            "location": "...",
            "bbox": "...",
            "frequency": "..."
            }}

            User query:
            {params.query}
            """

        try:
            response = await llm.ainvoke(prompt)
            content = getattr(response, "content", None)

            # convert to a dictionary
            import ast

            data = ast.literal_eval(content)

            # Validate and fill missing defaults
            return ExtractOutputSchema(
                dataset_type=data.get("dataset_type", "all"),
                location=data.get("location", "global"),
                bbox=data.get("bbox", None),
                frequency=data.get("frequency", "all"),
            )

        except Exception as e:
            logger.error(f"Error in ExtractAgent LLM call: {e}")
            # Gracefully degrade to defaults
            return ExtractOutputSchema(
                dataset_type="all",
                location="global",
                bbox=None,
                frequency="all",
            )

    async def _arun(self, params: ExtractInputSchema, **kwargs) -> ExtractOutputSchema:
        return await self.get_response_async(params, **kwargs)
