"""
Scientific Decomposition Component.

Uses LLM to decompose functional topics into specific observable phenomena
that can be measured with Earth science datasets.
"""

import asyncio
from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from ..utils.prompt_loader import load_and_format_prompt, load_prompt_template
from .topic_splitting import Topic


class ScientificDecomposition(BaseModel):
    """Individual scientific decomposition of a topic into measurable observables."""

    title: str = Field(
        ...,
        description="Concise title for this scientific decomposition",
    )
    scientific_justification: str = Field(
        ...,
        description="Scientific reasoning explaining how this observable relates to the topic and research question",
    )


class ScientificDecompositionOutput(BaseModel):
    """Output from scientific decomposition component."""

    decompositions: List[ScientificDecomposition] = Field(
        ...,
        description="List of scientific decompositions for the topic (1-6 decompositions)",
        min_items=1,
        max_items=6,
    )


class ScientificDecompositionInputSchema(InputSchema):
    """Input schema for scientific decomposition."""

    original_query: str = Field(
        ...,
        description="Original research question for context",
    )
    topic: Topic = Field(..., description="Topic to decompose into observables")


class ScientificDecompositionComponent(
    InstructorBaseAgent[
        ScientificDecompositionInputSchema,
        ScientificDecompositionOutput,
    ],
):
    """
    Component for generating scientific decompositions using LLM.

    Takes a functional topic and decomposes it into 1-6 specific observable phenomena
    that can be measured with satellite and Earth observation data.
    """

    input_schema = ScientificDecompositionInputSchema
    output_schema = ScientificDecompositionOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the scientific decomposition component."""
        # Set up specialized configuration for scientific decomposition
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for scientific decomposition
        config.system_prompt = load_prompt_template("scientific_decomposition_system")
        config.temperature = 0.1  # Low temperature for consistent, factual output

        super().__init__(config=config, debug=debug)

    async def process(
        self,
        original_query: str,
        topic: Topic,
    ) -> ScientificDecompositionOutput:
        """
        Generate scientific decompositions for a topic.

        Args:
            original_query: Original research question for context
            topic: Functional topic to decompose into observables

        Returns:
            ScientificDecompositionOutput with generated decompositions
        """
        if self.debug:
            logger.debug(
                f"Generating scientific decompositions for topic: '{topic.title}'",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(original_query, topic)

        # Add user message to memory
        self.memory.append({"role": "user", "content": user_prompt})

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Generate decompositions using LLM
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(
                        f"Generated {len(response.decompositions)} scientific decompositions",
                    )
                    for i, decomp in enumerate(response.decompositions, 1):
                        logger.debug(f"  {i}. {decomp.title}")

                return response

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to generate scientific decompositions after {max_retries + 1} attempts: {e}"
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
                    error_msg = f"Failed to generate scientific decompositions: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _format_user_prompt(self, original_query: str, topic: Topic) -> str:
        """Format the user prompt with research context and topic."""
        return load_and_format_prompt(
            "scientific_decomposition_user",
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
        )

    async def _arun(
        self,
        params: ScientificDecompositionInputSchema,
        **kwargs,
    ) -> ScientificDecompositionOutput:
        """Execute the scientific decomposition generation."""
        return await self.process(params.original_query, params.topic)
