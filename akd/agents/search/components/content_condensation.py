"""
Content condensation component for reducing full-text content to question-relevant excerpts.
"""

import tiktoken
from typing import List, Dict, Any
from loguru import logger

from akd.agents._base import LangBaseAgent, BaseAgentConfig
from akd.configs.prompts import CONTENT_CONDENSATION_PROMPT
from akd._base import IOSchema
from pydantic import Field


class ContentCondensationInputSchema(IOSchema):
    """Input schema for content condensation."""
    
    research_questions: str = Field(
        description="The specific research questions that need to be answered"
    )
    full_text_content: str = Field(
        description="The complete text content to be condensed"
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum token limit for the condensed output"
    )
    source_url: str = Field(
        default="",
        description="URL or identifier of the source document"
    )
    source_title: str = Field(
        default="",
        description="Title of the source document"
    )


class ContentCondensationOutputSchema(IOSchema):
    """Output schema for content condensation."""
    
    condensed_content: str = Field(
        description="The condensed content that addresses the research questions"
    )
    original_tokens: int = Field(
        description="Number of tokens in the original content"
    )
    condensed_tokens: int = Field(
        description="Number of tokens in the condensed content"
    )
    compression_ratio: float = Field(
        description="Ratio of condensed to original tokens"
    )
    has_relevant_content: bool = Field(
        description="Whether any relevant content was found"
    )


class ContentCondensationConfig(BaseAgentConfig):
    """Configuration for content condensation component."""
    
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model to use for content condensation"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for content condensation"
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retries for condensation"
    )


class ContentCondensationComponent(LangBaseAgent):
    """
    Component that condenses full-text content to extract only information 
    relevant to specific research questions, staying within token limits.
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
            # Fallback to a default encoding if model-specific encoding not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _prepare_condensation_prompt(
        self, 
        research_questions: str, 
        content: str, 
        max_tokens: int
    ) -> str:
        """Prepare the prompt for content condensation."""
        return f"""RESEARCH QUESTIONS:
{research_questions}

CONTENT TO CONDENSE (Target max {max_tokens} tokens):
{content}

Condense this content to extract only the information that directly addresses the research questions above. Be extremely selective and aggressive in filtering out irrelevant content."""
    
    async def _arun(
        self, 
        params: ContentCondensationInputSchema,
        **kwargs
    ) -> ContentCondensationOutputSchema:
        """
        Condense content to extract only question-relevant information.
        """
        
        original_tokens = self._count_tokens(params.full_text_content)
        
        if self.debug:
            logger.debug(
                f"Condensing content: {original_tokens} tokens -> target {params.max_tokens} tokens"
            )
        
        # If content is already under the limit, return as-is
        if original_tokens <= params.max_tokens:
            return ContentCondensationOutputSchema(
                condensed_content=params.full_text_content,
                original_tokens=original_tokens,
                condensed_tokens=original_tokens,
                compression_ratio=1.0,
                has_relevant_content=True
            )
        
        # Prepare the condensation prompt
        condensation_prompt = self._prepare_condensation_prompt(
            params.research_questions,
            params.full_text_content,
            params.max_tokens
        )
        
        try:
            # Use the CONTENT_CONDENSATION_PROMPT as system prompt
            system_prompt = CONTENT_CONDENSATION_PROMPT.format(
                max_tokens=params.max_tokens
            )
            
            # Get condensed content from LLM
            response = await self._agent.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": condensation_prompt}
            ])
            
            condensed_content = response.content.strip()
            condensed_tokens = self._count_tokens(condensed_content)
            
            # Check if content was deemed irrelevant
            has_relevant_content = not (
                condensed_content == "[NO RELEVANT CONTENT]" or
                condensed_tokens < 10
            )
            
            compression_ratio = condensed_tokens / original_tokens if original_tokens > 0 else 0
            
            if self.debug:
                logger.debug(
                    f"Condensation complete: {original_tokens} -> {condensed_tokens} tokens "
                    f"(ratio: {compression_ratio:.3f})"
                )
            
            return ContentCondensationOutputSchema(
                condensed_content=condensed_content,
                original_tokens=original_tokens,
                condensed_tokens=condensed_tokens,
                compression_ratio=compression_ratio,
                has_relevant_content=has_relevant_content
            )
            
        except Exception as e:
            logger.error(f"Error during content condensation: {e}")
            
            # Fallback: return truncated content if condensation fails
            fallback_content = params.full_text_content[:params.max_tokens * 4]  # Rough char/token ratio
            fallback_tokens = self._count_tokens(fallback_content)
            
            return ContentCondensationOutputSchema(
                condensed_content=fallback_content,
                original_tokens=original_tokens,
                condensed_tokens=fallback_tokens,
                compression_ratio=fallback_tokens / original_tokens if original_tokens > 0 else 0,
                has_relevant_content=True
            )