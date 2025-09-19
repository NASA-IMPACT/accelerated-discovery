"""
Known Parameters Component.

Uses LLM to extract hard filters (instruments, temporal/spatial bounds, etc.)
that can be directly identified from scientific research context.
"""

import asyncio
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from .prompt_loader import load_and_format_prompt, load_prompt_template
from .scientific_decomposition import ScientificDecomposition
from .topic_splitting import Topic


class QueryApproach(BaseModel):
    """Parameters for a single CMR query approach using known parameters only."""

    instrument: Optional[str] = Field(
        None,
        description="Single instrument name (e.g., MODIS, VIIRS)",
    )
    platform: Optional[str] = Field(
        None,
        description="Single platform/satellite name (e.g., Terra, Aqua)",
    )
    processing_level: Optional[str] = Field(
        None,
        description="Data processing level (e.g., Level 1B, Level 2)",
    )
    temporal: Optional[str] = Field(
        None,
        description="Temporal range in ISO format: YYYY-MM-DDTHH:mm:ssZ,YYYY-MM-DDTHH:mm:ssZ",
    )
    bounding_box: Optional[str] = Field(
        None,
        description="Spatial bounding box as comma-separated string: west,south,east,north",
    )
    temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution (e.g., daily, weekly, monthly)",
    )
    spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution (e.g., 250m, 1km, 30m)",
    )


class KnownParametersOutput(BaseModel):
    """Output from known parameters component."""

    query_approaches: List[QueryApproach] = Field(
        ...,
        description="List of query approaches using known parameters (1-5 approaches)",
        min_items=1,
        max_items=5,
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these specific parameters were identified",
    )


class KnownParametersInputSchema(InputSchema):
    """Input schema for known parameters extraction."""

    original_query: str = Field(..., description="Original research question")
    topic: Topic = Field(..., description="Topic being processed")
    decomposition: ScientificDecomposition = Field(
        ...,
        description="Scientific decomposition",
    )


class KnownParametersComponent(
    InstructorBaseAgent[KnownParametersInputSchema, KnownParametersOutput],
):
    """
    Component for extracting known parameters using LLM.

    Takes a research context (query + topic + decomposition) and identifies
    hard filters that can be directly applied to CMR queries without search.
    """

    input_schema = KnownParametersInputSchema
    output_schema = KnownParametersOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the known parameters component."""
        # Set up specialized configuration for known parameters extraction
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for known parameters
        config.system_prompt = load_prompt_template("known_parameters_system")
        config.temperature = (
            0.0  # Very low temperature for precise parameter extraction
        )

        super().__init__(config=config, debug=debug)

    async def process(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
    ) -> KnownParametersOutput:
        """
        Extract known parameters from research context.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification

        Returns:
            KnownParametersOutput with query approaches
        """
        if self.debug:
            logger.debug(
                f"Extracting known parameters for decomposition: '{decomposition.title}'",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(original_query, topic, decomposition)

        # Add user message to memory
        self.memory.append({"role": "user", "content": user_prompt})

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Generate parameters using LLM
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(
                        f"Generated {len(response.query_approaches)} query approaches",
                    )
                    for i, approach in enumerate(response.query_approaches, 1):
                        logger.debug(f"  {i}. {self._summarize_approach(approach)}")

                return response

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to extract known parameters after {max_retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

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
                    error_msg = f"Failed to extract known parameters: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _format_user_prompt(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
    ) -> str:
        """Format the user prompt with research context."""
        return load_and_format_prompt(
            "known_parameters_user",
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
        )

    def _summarize_approach(self, approach: QueryApproach) -> str:
        """Create a brief summary of a query approach for logging."""
        parts = []
        if approach.instrument:
            parts.append(f"instrument: {approach.instrument}")
        if approach.platform:
            parts.append(f"platform: {approach.platform}")
        if approach.temporal:
            parts.append(
                f"temporal: {approach.temporal[:10]}...",
            )  # Truncate for readability
        if approach.bounding_box:
            parts.append(f"bbox: {approach.bounding_box}")

        return ", ".join(parts) if parts else "no specific parameters"

    async def _arun(
        self,
        params: KnownParametersInputSchema,
        **kwargs,
    ) -> KnownParametersOutput:
        """Execute the known parameters extraction."""
        return await self.process(
            params.original_query,
            params.topic,
            params.decomposition,
        )
