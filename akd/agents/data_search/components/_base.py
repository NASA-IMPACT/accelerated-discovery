"""
Base classes for data search components.

Provides common functionality for LLM-powered components including:
- Automatic prompt loading from templates
- Retry logic with exponential backoff for rate limiting
- Standardized initialization patterns
- Memory management
"""

import asyncio
from pathlib import Path
from typing import Generic, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from ..utils.prompt_loader import load_prompt_template

TInput = TypeVar("TInput", bound=InputSchema)
TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseDataSearchComponent(
    InstructorBaseAgent[TInput, TOutput],
    Generic[TInput, TOutput],
):
    """
    Base class for all data search LLM components.

    Provides common patterns:
    - Automatic prompt template loading based on template_name
    - Configurable retry logic with exponential backoff
    - Standardized error handling and logging
    - Memory management helpers

    Subclasses should:
    - Set input_schema and output_schema class attributes
    - Set template_name class attribute (e.g., "topic_splitting")
    - Override default_temperature if needed
    - Implement process() method for main logic
    - Implement _format_user_prompt() if using custom prompt formatting
    """

    # These will be set by subclasses - marking as None to avoid metaclass errors
    input_schema: Optional[type[InputSchema]] = None
    output_schema: Optional[type[BaseModel]] = None

    # Class attributes to override in subclasses
    template_name: Optional[str] = None  # e.g., "topic_splitting"
    default_temperature: float = 0.0
    retry_enabled: bool = True
    max_retries: int = 3
    retry_base_delay: float = 1.0

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        template_name: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the component with automatic prompt loading.

        Args:
            config: Agent configuration (will create default if None)
            debug: Enable debug logging
            template_name: Override class-level template_name
            prompts_dir: Optional directory for repository-specific prompts
        """
        # Allow instance-level template name override
        if template_name:
            self.template_name = template_name

        # Store prompts_dir for later use
        self.prompts_dir = prompts_dir

        # Create default config if not provided
        if config is None:
            config = BaseAgentConfig()

        # Load system prompt template if template_name is set
        if self.template_name:
            config.system_prompt = load_prompt_template(
                f"{self.template_name}_system",
                prompts_dir=prompts_dir,
            )

        # Set temperature
        if not hasattr(config, "temperature") or config.temperature is None:
            config.temperature = self.default_temperature

        # Call parent init
        super().__init__(config=config, debug=debug)

        # Load user prompt template if available
        self.user_prompt_template = None
        if self.template_name:
            try:
                self.user_prompt_template = load_prompt_template(
                    f"{self.template_name}_user",
                    prompts_dir=prompts_dir,
                )
            except FileNotFoundError:
                # User template is optional
                pass

    async def _execute_with_retry(
        self,
        operation_name: str,
        custom_error_prefix: Optional[str] = None,
    ) -> TOutput:
        """
        Execute LLM call with retry logic and exponential backoff.

        Args:
            operation_name: Name of operation for logging (e.g., "identify topics")
            custom_error_prefix: Custom prefix for error messages

        Returns:
            Response from LLM

        Raises:
            RuntimeError: If all retries fail or non-retryable error occurs
        """
        if not self.retry_enabled:
            # No retry - execute directly
            return await self.get_response_async()

        error_prefix = custom_error_prefix or f"Failed to {operation_name}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(f"{operation_name} completed successfully")

                return response

            except Exception as e:
                # Check if we've exhausted retries
                if attempt == self.max_retries:
                    error_msg = (
                        f"{error_prefix} after {self.max_retries + 1} attempts: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # Check if it's a rate limit error (429 or mentions "rate")
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = self.retry_base_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{self.max_retries + 1})",
                        )
                    await asyncio.sleep(delay)
                else:
                    # Non-rate-limit error - don't retry
                    error_msg = f"{error_prefix}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    async def _execute_with_retry_custom(
        self,
        callable_func,
        operation_name: str,
        custom_error_prefix: Optional[str] = None,
    ):
        """
        Execute a custom async callable with retry logic and exponential backoff.

        Useful for operations that need custom LLM calls (e.g., schema overriding).

        Args:
            callable_func: Async function to execute
            operation_name: Name of operation for logging
            custom_error_prefix: Custom prefix for error messages

        Returns:
            Result from callable_func

        Raises:
            RuntimeError: If all retries fail or non-retryable error occurs
        """
        if not self.retry_enabled:
            # No retry - execute directly
            return await callable_func()

        error_prefix = custom_error_prefix or f"Failed to {operation_name}"

        for attempt in range(self.max_retries + 1):
            try:
                result = await callable_func()

                if self.debug:
                    logger.debug(f"{operation_name} completed successfully")

                return result

            except Exception as e:
                # Check if we've exhausted retries
                if attempt == self.max_retries:
                    error_msg = (
                        f"{error_prefix} after {self.max_retries + 1} attempts: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # Check if it's a rate limit error (429 or mentions "rate")
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = self.retry_base_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{self.max_retries + 1})",
                        )
                    await asyncio.sleep(delay)
                else:
                    # Non-rate-limit error - don't retry
                    error_msg = f"{error_prefix}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _add_user_message(self, content: str):
        """
        Add a user message to the conversation memory.

        Args:
            content: User message content
        """
        self.memory.append({"role": "user", "content": content})

    def _set_messages(self, content: str):
        """
        Set messages directly (used by some components instead of memory).

        Args:
            content: User message content
        """
        self.messages = [{"role": "user", "content": content}]

    def _format_user_prompt_from_template(self, **kwargs) -> str:
        """
        Format user prompt using loaded template.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string

        Raises:
            RuntimeError: If no user prompt template is loaded
        """
        if not self.user_prompt_template:
            raise RuntimeError(
                f"No user prompt template loaded for {self.__class__.__name__}",
            )

        return self.user_prompt_template.format(**kwargs)

    async def process(self, *args, **kwargs) -> TOutput:
        """
        Process the component's main logic.

        This method should be overridden by subclasses to implement
        their specific processing logic.

        Returns:
            Component-specific output

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process() method",
        )
