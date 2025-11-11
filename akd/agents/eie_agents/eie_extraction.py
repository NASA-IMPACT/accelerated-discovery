from datetime import date
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel
from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig


class TemporalExtent(BaseModel):
    """Structured temporal coverage for a dataset request."""

    start: Optional[str] = Field(
        default=None,
        description="Start date (ISO 8601 format: YYYY-MM-DD or YYYY).",
    )
    end: Optional[str] = Field(
        default=None,
        description="End date (ISO 8601 format: YYYY-MM-DD or YYYY).",
    )


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

    temporal_extent: Optional[Dict] = Field(
        default={"dates": {"start": "1900-01-01", "end": date.today().isoformat()}},
        description="Time interval or dates extracted from the query.",
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
        You are an expert in geospatial data and temporal reasoning.
        Given a user query, extract the following structured fields as a JSON object.

        Fields to extract:
        - dataset_type: Category of dataset (e.g. "methane", "population", "nitrogen dioxide").
        - location: Human-readable place or region name.
        - bbox: Bounding box of the location in GeoJSON format, or null if unknown.
        - frequency: Temporal frequency of the data (e.g. "daily", "monthly", "yearly", "all").
        - temporal_extent: Represents time information extracted from the query.
            - If an interval is mentioned, use an object with "start" and "end" keys (ISO 8601 format).
            Example: {{"start": "2019-01-01", "end": "2020-01-01"}}
            - If one or more discrete dates are mentioned, use a list of ISO date strings.
            Example: ["2021-06-01"] or ["2020-01-01", "2021-03-15"]
            - If no date is provided, set this field to null.
            - Resolve relative references such as "last year", "past month", or "present" using the current date.

        Respond strictly in **valid JSON format**, with no extra text or explanations.

        User query:
        {params.query}
        """

        try:
            print("Activating Extraction Agent...")
            response = await llm.ainvoke(prompt)
            content = getattr(response, "content", None)
            print("content ", content)

            import json

            data = json.loads(content)
            print("data ", data)

            # Validate and fill missing defaults
            return ExtractOutputSchema(
                dataset_type=data.get("dataset_type", "all"),
                location=data.get("location", "global"),
                bbox=data.get("bbox", None),
                frequency=data.get("frequency", "all"),
                temporal_extent={"dates": data.get("temporal_extent", {})},
            )

        except Exception as e:
            logger.error(f"Error in ExtractAgent LLM call: {e}")
            # Gracefully degrade to defaults
            return ExtractOutputSchema(
                dataset_type="all",
                location="global",
                bbox=None,
                frequency="all",
                temporal_extent={"dates": {"start": "1900-01-01", "end": date.today().isoformat()}},
            )

    async def _arun(self, params: ExtractInputSchema, **kwargs) -> ExtractOutputSchema:
        return await self.get_response_async(params, **kwargs)
