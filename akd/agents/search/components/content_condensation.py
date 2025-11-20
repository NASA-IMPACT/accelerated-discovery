"""
LLM-based content condensation for research synthesis.
"""

from typing import List

from litellm import token_counter
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, IOSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.configs.prompts import CONTENT_CONDENSATION_PROMPT
from akd.structures import SearchResultItem


# Private schemas for internal single condensation agent
class _SingleContentCondensationInputSchema(InputSchema):
    """Input schema for single content condensation (internal use only)."""

    research_question: str = Field(
        description="The research question to extract relevant content for",
    )
    search_result: SearchResultItem = Field(
        description="Single search result with content to condense",
    )
    target_tokens: int = Field(
        description="Target token count for condensed output",
    )


class _SingleContentCondensationOutputSchema(OutputSchema):
    """Output schema for single content condensation (internal use only)."""

    condensed_content: str = Field(
        description="The condensed content relevant to the research question",
    )


# Private agent for single content condensation
class _SingleContentCondensationAgent(
    LiteLLMInstructorBaseAgent[
        _SingleContentCondensationInputSchema,
        _SingleContentCondensationOutputSchema,
    ],
):
    """
    Internal agent for condensing a single search result's content.
    Not intended for external use - used internally by ContentCondensationComponent.
    """

    input_schema = _SingleContentCondensationInputSchema
    output_schema = _SingleContentCondensationOutputSchema
    config_schema = BaseAgentConfig  # Uses same config as parent

    async def _arun(
        self,
        params: _SingleContentCondensationInputSchema,
        **kwargs,
    ) -> _SingleContentCondensationOutputSchema:
        """Condense content for a single source."""
        result = params.search_result

        # Build the condensation prompt
        prompt = CONTENT_CONDENSATION_PROMPT.format(
            research_question=params.research_question,
            source_title=result.title or "Unknown",
            source_url=result.url,
            content=result.content,
            target_tokens=params.target_tokens,
        )

        # Create messages
        messages = [
            self._default_system_message(),
            {"role": "user", "content": prompt},
        ]

        # Get structured response
        response = await self.get_response_async(
            messages=messages,
            response_model=self.output_schema,
        )

        return response


class ContentCondensationInputSchema(IOSchema):
    """Input schema for content condensation."""

    research_question: str = Field(
        description="The research question to extract relevant content for",
    )
    search_results: List[SearchResultItem] = Field(
        description="Search results with full text content to condense",
    )
    max_tokens: int = Field(
        default=40000,
        description="Maximum total tokens for condensed output",
    )


class ContentCondensationOutputSchema(IOSchema):
    """Output schema for content condensation."""

    condensed_results: List[SearchResultItem] = Field(
        description="Search results with condensed content",
    )
    total_tokens_reduced: int = Field(
        description="Total tokens reduced through condensation",
    )
    compression_ratio: float = Field(
        description="Ratio of final to original token count",
    )


class ContentCondensationConfig(BaseAgentConfig):
    """Configuration for content condensation."""

    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model to use for content condensation",
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for content condensation",
    )
    min_content_length: int = Field(
        default=100,
        description="Minimum content length to consider for condensation",
    )


class ContentCondensationComponent(LiteLLMInstructorBaseAgent):
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

    def _post_init(self):
        """Initialize the private single condensation agent."""
        super()._post_init()
        # Cast config to BaseAgentConfig, excluding extra fields like min_content_length
        base_config = BaseAgentConfig(**self.config.model_dump(exclude={"min_content_length"}))
        self._condenser = _SingleContentCondensationAgent(
            config=base_config,
            debug=self.debug,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return token_counter(text=text, model=self.config.model_name)

    async def _condense_single_result(
        self,
        result: SearchResultItem,
        research_question: str,
        target_tokens: int,
    ) -> SearchResultItem:
        """Condense content in a single search result."""
        if not result.content or len(result.content.strip()) < self.config.min_content_length:
            return result

        original_tokens = self._count_tokens(result.content)
        if original_tokens <= target_tokens:
            return result

        try:
            if self.debug:
                logger.debug(
                    f"Condensation input preview | url: {result.url} | tokens: {original_tokens} -> target: {target_tokens}",
                )

            # Use the private condensation agent
            condensation_input = _SingleContentCondensationInputSchema(
                research_question=research_question,
                search_result=result,
                target_tokens=target_tokens,
            )

            response = await self._condenser.arun(condensation_input)
            condensed_content = response.condensed_content.strip()

            # Check if content was deemed irrelevant
            if condensed_content == "[NO RELEVANT CONTENT]" or len(condensed_content) < 10:
                condensed_content = result.content or ""

            # Create new result with condensed content
            condensed_result = result.model_copy()
            condensed_result.content = condensed_content

            if self.debug:
                new_tokens = self._count_tokens(condensed_content)
                logger.debug(
                    f"Condensed {result.url}: {original_tokens} -> {new_tokens} tokens",
                )
                logger.debug(
                    f"Condensation output preview | content: {condensed_content[:200]}",
                )

            return condensed_result

        except Exception as e:
            if self.debug:
                logger.warning(f"Error condensing {result.url}: {e}")
            return result

    async def _arun(
        self,
        params: ContentCondensationInputSchema,
        **kwargs,
    ) -> ContentCondensationOutputSchema:
        """
        Condense content in search results to extract only information relevant
        to the research question.
        """

        # Filter to only results with substantial content
        results_with_content = [
            r for r in params.search_results if r.content and len(r.content.strip()) >= self.config.min_content_length
        ]

        if not results_with_content:
            return ContentCondensationOutputSchema(
                condensed_results=params.search_results,
                total_tokens_reduced=0,
                compression_ratio=1.0,
            )

        # Calculate original token count
        original_tokens = sum(self._count_tokens(r.content) for r in results_with_content)

        if self.debug:
            logger.debug(
                f"Condensing {len(results_with_content)} results with {original_tokens} total tokens",
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
            if result.content and len(result.content.strip()) >= self.config.min_content_length:
                condensed = await self._condense_single_result(
                    result,
                    params.research_question,
                    tokens_per_result,
                )
                condensed_results.append(condensed)
            else:
                condensed_results.append(result)

        # Calculate final metrics
        final_tokens = sum(self._count_tokens(r.content) for r in condensed_results if r.content)

        tokens_reduced = original_tokens - final_tokens
        compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0

        if self.debug:
            logger.debug(
                f"Condensation complete: {original_tokens} -> {final_tokens} tokens "
                f"(reduced {tokens_reduced}, ratio: {compression_ratio:.3f})",
            )

        return ContentCondensationOutputSchema(
            condensed_results=condensed_results,
            total_tokens_reduced=tokens_reduced,
            compression_ratio=compression_ratio,
        )
