"""
CMR Query Generation Component.

Uses LLM to generate structured CMR search parameters based on scientific angles
and research queries.
"""

import asyncio
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from .prompt_loader import load_and_format_prompt, load_prompt_template
from .scientific_angles import ScientificAngle


class CMRCollectionSearchParams(BaseModel):
    """Parameters for CMR collection search that map directly to MCP interface."""

    keyword: Optional[str] = Field(
        None,
        description="Text search across collection metadata",
    )
    short_name: Optional[str] = Field(
        None,
        description="Collection short name (e.g., MOD09A1)",
    )
    platform: Optional[str] = Field(
        None,
        description="Single platform/satellite name (e.g., Terra, Aqua)",
    )
    instrument: Optional[str] = Field(
        None,
        description="Single instrument name (e.g., MODIS, VIIRS)",
    )
    temporal: Optional[str] = Field(
        None,
        description="Temporal range in ISO format: YYYY-MM-DDTHH:mm:ssZ,YYYY-MM-DDTHH:mm:ssZ",
    )
    bounding_box: Optional[str] = Field(
        None,
        description="Spatial bounding box as comma-separated string: west,south,east,north",
    )


class CMRQueryOutput(BaseModel):
    """Output from CMR query generation component."""

    search_queries: List[CMRCollectionSearchParams] = Field(
        ...,
        description="List of CMR search parameter sets for this scientific angle",
        min_items=1,
        max_items=5,
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these specific search parameters were chosen",
    )


class CMRQueryGenerationInputSchema(InputSchema):
    """Input schema for CMR query generation."""

    scientific_angle: ScientificAngle
    original_query: str


class CMRQueryGenerationComponent(
    InstructorBaseAgent[CMRQueryGenerationInputSchema, CMRQueryOutput],
):
    """
    Component for generating CMR search parameters using LLM.

    Takes a scientific angle and the original research query, then generates
    structured CMR search parameters that map directly to the MCP interface.
    """

    input_schema = CMRQueryGenerationInputSchema
    output_schema = CMRQueryOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the CMR query generation component."""
        # Set up specialized configuration for CMR query generation
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for CMR query generation
        config.system_prompt = load_prompt_template("cmr_query_generation_system")
        config.temperature = 0.0  # Very low temperature for precise, structured output

        super().__init__(config=config, debug=debug)

    async def process(
        self,
        scientific_angle: ScientificAngle,
        original_query: str,
    ) -> CMRQueryOutput:
        """
        Generate CMR search parameters for the given scientific angle.

        Args:
            scientific_angle: Scientific angle with title and justification
            original_query: Original research query for context

        Returns:
            CMRQueryOutput with structured search parameters
        """
        if self.debug:
            logger.debug(
                f"Generating CMR queries for angle: '{scientific_angle.title}'",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(scientific_angle, original_query)

        # Add user message to memory
        self.memory.append({"role": "user", "content": user_prompt})

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Generate CMR parameters using LLM
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(
                        f"Generated {len(response.search_queries)} CMR search queries",
                    )
                    for i, query in enumerate(response.search_queries, 1):
                        logger.debug(f"  {i}. {self._summarize_query(query)}")

                return response

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to generate CMR queries after {max_retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                # Check if it's a rate limit error
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = base_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})",
                        )
                    await asyncio.sleep(delay)
                else:
                    # Non-rate-limit error, don't retry
                    error_msg = f"Failed to generate CMR queries: {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

    def _format_user_prompt(
        self,
        scientific_angle: ScientificAngle,
        original_query: str,
    ) -> str:
        """Format the user prompt with scientific angle and query."""
        return load_and_format_prompt(
            "cmr_query_generation_user",
            original_query=original_query,
            scientific_angle_title=scientific_angle.title,
            scientific_angle_justification=scientific_angle.scientific_justification,
        )

    def _summarize_query(self, query) -> str:
        """Create a brief summary of a CMR query for logging."""
        parts = []
        if query.keyword:
            parts.append(f"keyword: {query.keyword}")
        if query.platform:
            parts.append(f"platform: {query.platform}")
        if query.instrument:
            parts.append(f"instrument: {query.instrument}")

        return ", ".join(parts) if parts else "empty query"

    async def _arun(
        self,
        params: CMRQueryGenerationInputSchema,
        **kwargs,
    ) -> CMRQueryOutput:
        """Execute the CMR query generation."""
        return await self.process(params.scientific_angle, params.original_query)
