"""
Topic Splitting Component.

Uses LLM to identify distinct functional topics within scientific research questions
for separate data discovery workflows.
"""

import asyncio
from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from .prompt_loader import load_and_format_prompt, load_prompt_template


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
        description="List of functional topics identified in the research question (1-6 topics)",
        min_items=1,
        max_items=6,
    )


class TopicSplittingInputSchema(InputSchema):
    """Input schema for topic splitting."""

    query: str = Field(
        ...,
        description="Scientific research question to analyze for topics",
    )


class TopicSplittingComponent(
    InstructorBaseAgent[TopicSplittingInputSchema, TopicSplittingOutput],
):
    """
    Component for identifying functional topics using LLM.

    Takes a scientific research question and identifies 1-6 distinct functional topics
    that represent separate areas of inquiry requiring different data discovery approaches.
    """

    input_schema = TopicSplittingInputSchema
    output_schema = TopicSplittingOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the topic splitting component."""
        # Set up specialized configuration for topic splitting
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for topic splitting
        config.system_prompt = load_prompt_template("topic_splitting_system")
        config.temperature = 0.0  # Low temperature for consistent, precise analysis

        super().__init__(config=config, debug=debug)

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

        # Add user message to memory
        self.memory.append({"role": "user", "content": user_prompt})

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Generate topics using LLM
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(f"Identified {len(response.topics)} functional topics")
                    for i, topic in enumerate(response.topics, 1):
                        logger.debug(f"  {i}. {topic.title}")

                return response

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to identify topics after {max_retries + 1} attempts: {e}"
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
                    error_msg = f"Failed to identify topics: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _format_user_prompt(self, query: str) -> str:
        """Format the user prompt with the research query."""
        return load_and_format_prompt(
            "topic_splitting_user",
            query=query,
        )

    async def _arun(
        self,
        params: TopicSplittingInputSchema,
        **kwargs,
    ) -> TopicSplittingOutput:
        """Execute the topic splitting analysis."""
        return await self.process(params.query)
