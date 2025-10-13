"""
Scientific Decomposition Component.

Uses LLM to decompose functional topics into specific observable phenomena
that can be measured with Earth science datasets.
"""

from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema

from ..utils.prompt_loader import load_and_format_prompt
from ._base import BaseDataSearchComponent
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
        description="List of scientific decompositions for the topic (1-2 decompositions)",
        min_items=1,
        max_items=2,
    )


class ScientificDecompositionInputSchema(InputSchema):
    """Input schema for scientific decomposition."""

    original_query: str = Field(
        ...,
        description="Original research question for context",
    )
    topic: Topic = Field(..., description="Topic to decompose into observables")


class ScientificDecompositionComponent(
    BaseDataSearchComponent[
        ScientificDecompositionInputSchema,
        ScientificDecompositionOutput,
    ],
):
    """
    Component for generating scientific decompositions using LLM.

    Takes a functional topic and decomposes it into 1-2 specific observable phenomena
    that can be measured with satellite and Earth observation data.
    """

    input_schema = ScientificDecompositionInputSchema
    output_schema = ScientificDecompositionOutput

    # Base class configuration
    template_name = "scientific_decomposition"
    default_temperature = 0.1  # Low temperature for consistent, factual output

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
        self._add_user_message(user_prompt)

        # Execute with retry logic
        response = await self._execute_with_retry(
            operation_name="generate scientific decompositions",
            custom_error_prefix="Failed to generate scientific decompositions",
        )

        if self.debug:
            logger.debug(
                f"Generated {len(response.decompositions)} scientific decompositions",
            )
            for i, decomp in enumerate(response.decompositions, 1):
                logger.debug(f"  {i}. {decomp.title}")

        return response

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
