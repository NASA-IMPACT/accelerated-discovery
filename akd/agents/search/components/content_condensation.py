"""
LLM-based content condensation for research synthesis.
"""

from typing import List

import tiktoken
from loguru import logger
from pydantic import Field

from akd._base import IOSchema
from akd.agents._base import BaseAgentConfig, LangBaseAgent
from akd.configs.prompts import CONTENT_CONDENSATION_PROMPT
from akd.structures import SearchResultItem


class ContentCondensationInputSchema(IOSchema):
    """Input schema for content condensation."""

    research_question: str = Field(
        description="The research question to extract relevant content for"
    )
    search_results: List[SearchResultItem] = Field(
        description="Search results with full text content to condense"
    )
    max_tokens: int = Field(
        default=4000, description="Maximum total tokens for condensed output"
    )


class ContentCondensationOutputSchema(IOSchema):
    """Output schema for content condensation."""

    condensed_results: List[SearchResultItem] = Field(
        description="Search results with condensed content"
    )
    total_tokens_reduced: int = Field(
        description="Total tokens reduced through condensation"
    )
    compression_ratio: float = Field(
        description="Ratio of final to original token count"
    )


class ContentCondensationConfig(BaseAgentConfig):
    """Configuration for content condensation."""

    model_name: str = Field(
        default="gpt-4o-mini", description="Model to use for content condensation"
    )
    temperature: float = Field(
        default=0.1, description="Temperature for content condensation"
    )


class ContentCondensationComponent(LangBaseAgent):
    """
    Simple LLM-based component that condenses SearchResultItem content to extract
    only information relevant to a research question, honoring token limits.

    Uses dependency injection for better testability and flexibility.
    """

    input_schema = ContentCondensationInputSchema
    output_schema = ContentCondensationOutputSchema
    config_schema = ContentCondensationConfig

    def __init__(
        self,
        config: ContentCondensationConfig | None = None,
        debug: bool = False,
    ):
        config = config or ContentCondensationConfig()
        super().__init__(config=config, debug=debug)

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    async def _condense_single_result(
        self, result: SearchResultItem, research_question: str, target_tokens: int
    ) -> SearchResultItem:
        """Condense content in a single search result."""
        if not result.content or len(result.content.strip()) < 100:
            return result

        original_tokens = self._count_tokens(result.content)
        if original_tokens <= target_tokens:
            return result

        prompt = CONTENT_CONDENSATION_PROMPT.format(
            research_question=research_question,
            source_title=result.title or "Unknown",
            source_url=result.url,
            content=result.content,
            target_tokens=target_tokens,
        )

        try:
            response = await self.client.ainvoke([{"role": "user", "content": prompt}])

            condensed_content = response.content.strip()

            # Check if content was deemed irrelevant
            if (
                condensed_content == "[NO RELEVANT CONTENT]"
                or len(condensed_content) < 10
            ):
                condensed_content = ""

            # Create new result with condensed content
            condensed_result = result.model_copy()
            condensed_result.content = condensed_content

            if self.debug:
                new_tokens = self._count_tokens(condensed_content)
                logger.debug(
                    f"Condensed {result.url}: {original_tokens} -> {new_tokens} tokens"
                )

            return condensed_result

        except Exception as e:
            if self.debug:
                logger.warning(f"Error condensing {result.url}: {e}")
            return result

    async def _arun(
        self, params: ContentCondensationInputSchema, **kwargs
    ) -> ContentCondensationOutputSchema:
        """
        Condense content in search results to extract only information relevant
        to the research question.
        """

        # Filter to only results with substantial content
        results_with_content = [
            r
            for r in params.search_results
            if r.content and len(r.content.strip()) >= 100
        ]

        if not results_with_content:
            return ContentCondensationOutputSchema(
                condensed_results=params.search_results,
                total_tokens_reduced=0,
                compression_ratio=1.0,
            )

        # Calculate original token count
        original_tokens = sum(
            self._count_tokens(r.content) for r in results_with_content
        )

        if self.debug:
            logger.debug(
                f"Condensing {len(results_with_content)} results with {original_tokens} total tokens"
            )

        # If already under limit, return as-is
        if original_tokens <= params.max_tokens:
            return ContentCondensationOutputSchema(
                condensed_results=params.search_results,
                total_tokens_reduced=0,
                compression_ratio=1.0,
            )

        # Allocate tokens per result
        tokens_per_result = params.max_tokens // len(results_with_content)

        # Condense each result
        condensed_results = []
        for result in params.search_results:
            if result.content and len(result.content.strip()) >= 100:
                condensed = await self._condense_single_result(
                    result, params.research_question, tokens_per_result
                )
                condensed_results.append(condensed)
            else:
                condensed_results.append(result)

        # Calculate final metrics
        final_tokens = sum(
            self._count_tokens(r.content) for r in condensed_results if r.content
        )

        tokens_reduced = original_tokens - final_tokens
        compression_ratio = (
            final_tokens / original_tokens if original_tokens > 0 else 1.0
        )

        if self.debug:
            logger.debug(
                f"Condensation complete: {original_tokens} -> {final_tokens} tokens "
                f"(reduced {tokens_reduced}, ratio: {compression_ratio:.3f})"
            )

        return ContentCondensationOutputSchema(
            condensed_results=condensed_results,
            total_tokens_reduced=tokens_reduced,
            compression_ratio=compression_ratio,
        )
