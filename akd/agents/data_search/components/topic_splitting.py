"""
Topic Splitting Component.

Uses LLM to identify distinct functional topics within scientific research questions
for separate data discovery workflows.
"""

from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema

from ..utils.prompt_loader import load_and_format_prompt
from ._base import BaseDataSearchComponent


class Topic(BaseModel):
    """Individual functional topic identified in a scientific query."""

    title: str = Field(..., description="Concise title for this functional topic")
    functional_context: str = Field(
        ...,
        description="Explanation of why this represents a distinct functional area requiring separate data discovery",
    )


class TopicSplittingOutput(BaseModel):
    """Output from topic splitting component."""

    topics: List[Topic] = Field(
        ...,
        description="List of functional topics identified in the research question (1-2 topics)",
        min_items=1,
        max_items=2,
    )


class TopicSplittingInputSchema(InputSchema):
    """Input schema for topic splitting."""

    query: str = Field(
        ...,
        description="Scientific research question to analyze for topics",
    )


class TopicSplittingComponent(
    BaseDataSearchComponent[TopicSplittingInputSchema, TopicSplittingOutput],
):
    """
    Component for identifying functional topics using LLM.

    Takes a scientific research question and identifies 1-2 distinct functional topics
    that represent separate areas of inquiry requiring different data discovery approaches.
    """

    input_schema = TopicSplittingInputSchema
    output_schema = TopicSplittingOutput

    # Base class configuration
    template_name = "topic_splitting"
    default_temperature = 0.0  # Low temperature for consistent, precise analysis

    async def process(self, query: str) -> TopicSplittingOutput:
        """
        Identify functional topics in a scientific research question.

        Args:
            query: Natural language scientific research query

        Returns:
            TopicSplittingOutput with identified topics
        """
        if self.debug:
            logger.debug(f"Analyzing query for functional topics: '{query}'")

        # Format the user prompt
        user_prompt = self._format_user_prompt(query)

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "topic_splitting",
            query[:50],  # Use truncated query as context
        )

        # Add user message to memory
        self._add_user_message(user_prompt)

        # Execute with retry logic
        response = await self._execute_with_retry(
            operation_name="identify topics",
            custom_error_prefix="Failed to identify topics",
        )

        if self.debug:
            logger.debug(f"Identified {len(response.topics)} functional topics")
            for i, topic in enumerate(response.topics, 1):
                logger.debug(f"  {i}. {topic.title}")

        return response

    def _format_user_prompt(self, query: str) -> str:
        """Format the user prompt with the research query."""
        # Get min/max values from output schema
        # metadata[0] = MinLen, metadata[1] = MaxLen
        min_topics = self.output_schema.model_fields["topics"].metadata[0].min_length
        max_topics = self.output_schema.model_fields["topics"].metadata[1].max_length

        return load_and_format_prompt(
            "topic_splitting_user",
            query=query,
            min_topics=min_topics,
            max_topics=max_topics,
        )

    async def _arun(
        self,
        params: TopicSplittingInputSchema,
        **kwargs,
    ) -> TopicSplittingOutput:
        """Execute the topic splitting analysis."""
        return await self.process(params.query)
