from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

import instructor
import openai
from litellm import acompletion, get_model_info
from litellm.utils import trim_messages
from loguru import logger
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT


class BaseAgentConfig(BaseConfig):
    """Configuration class for base agents."""

    base_url: AnyUrl | None = Field(default=CONFIG.model_config_settings.base_url)
    api_key: str | None = Field(default=CONFIG.model_config_settings.api_keys.openai)
    model_name: str | None = Field(default=CONFIG.model_config_settings.model_name)
    temperature: float = Field(
        default=CONFIG.model_config_settings.temperature,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    system_prompt: str | None = Field(default=DEFAULT_SYSTEM_PROMPT)
    llm_timeout: float | None = Field(
        default=180.0,
        description="Timeout in seconds for individual LLM API calls. Set to None to disable. Timeouts trigger retry with exponential backoff.",
    )
    stateless: bool = Field(
        default=True,
        description="Whether to maintain conversation history/state",
    )
    input_hints: bool = Field(
        default=False,
        description="Whether to include input schema field information in system prompt",
    )

    # Token management
    max_tokens: int = Field(
        default=CONFIG.model_config_settings.max_tokens,
        ge=5,
        le=1_000_000,  # hard max to 1M tokens
        description="Maximum tokens for input message context",
    )
    trim_ratio: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Target ratio after trimming (0.75 = use 75% of max)",
    )
    enable_trimming: bool = Field(
        default=True,
        description="Enable automatic message trimming",
    )
    num_retries: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of retries for LLM calls",
    )

    @model_validator(mode="after")
    def validate_max_tokens_against_model(self):
        """Validate that max_tokens doesn't exceed the model's actual capacity."""
        if not (self.model_name and self.max_tokens):
            return self
        try:
            model_info = get_model_info(self.model_name)
        except Exception as e:
            logger.error(f"Could not retrieve model info for '{self.model_name}': {e}")
            return self

        model_limit = model_info.get("max_input_tokens")

        if model_limit and self.max_tokens > model_limit:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) exceeds model '{self.model_name}' capacity ({model_limit} tokens)",
            )
        return self

    @field_validator("input_hints", mode="before")
    @classmethod
    def warn_input_hints_deprecated(cls, v):
        """Emit deprecation warning when input_hints is explicitly set."""
        # Only warn if a non-default value is being set
        if v is not None:
            logger.warning(
                "The 'input_hints' parameter is deprecated and will be removed in a future version. "
                "Please use 'io_hints' instead, which is now available in the base BaseConfig class."
                "Setting input_hints doesn't have any effect.",
            )
        return v


class BaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](AbstractBase):
    """
    Base class for chat agents that interact with a language model.

    This class provides the basic structure for an agent that can handle
    asynchronous operations, manage memory, and utilize a language model
    for generating responses based on user input.

    Notes:
    - We internally use `_system_prompt` to access actual system prompt that the model sees.
    - The `_system_prompt` has enhanced prompt based on `input_hints` flag.
    - We also have `_default_system_message` in InstructorBaseAgent that creates a dict
    """

    config_schema = BaseAgentConfig

    @property
    def memory(self) -> Any:
        """
        Returns the memory of the agent, implemented by subclasses.
        This property should return the memory structure used by the agent,
        which typically includes past messages or interactions.
        Raises:
            NotImplementedError: If the property is not implemented in a subclass.
        Args:
            None

        Returns:
            list[Any | BaseModel]: The memory of the agent.
        """
        raise NotImplementedError("Attribute 'memory' not implemented.")

    def reset_memory(self) -> None:
        pass

    @property
    def _system_prompt(self) -> str:
        """
        Enhanced system prompt with optional input hints.

        Returns:
            str: System prompt with input schema information if enabled.
        """
        content = self.system_prompt

        # Early return if input hints disabled
        if not self.input_hints:
            return content

        # Add agent description if available
        if self.description:
            content += f"\n\nAGENT DESCRIPTION:\n{self.description}"
        return content

    @abstractmethod
    async def get_response_async(
        self,
        *args,
        **kwargs,
    ) -> OutputSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Optional[OutputSchema]):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            OutputSchema: The response from the language model.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class InstructorBaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](BaseAgent):
    """Base class for instructor-based chat agents.
    Note:
        The object attributes (like `api_key`, `model_name` etc.) are dynamically set from the config.
    """

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config, debug=debug)

        # Create the OpenAI client
        self.client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=str(self.base_url),
            ),
        )

        # Initialize memory
        self._memory = []

    @property
    def memory(self) -> list[dict[str, str]]:
        return self._memory

    def reset_memory(self) -> None:
        """
        Resets the memory of the agent.
        This method clears the chat message history, effectively resetting the agent's memory.
        """
        self.memory.clear()

    def _default_system_message(self) -> dict[str, str]:
        """
        Returns the default system message.

        Returns:
            dict[str, str]: System message dictionary with role and content.
        """
        return {
            "role": "system",
            "content": self._system_prompt,
        }

    def _create_instructor_compatible_model(self, response_model: type[OutputSchema]):
        """Create a model that's compatible with instructor but avoids IOSchema validation."""

        # Get the fields from the original model
        fields = {}
        for field_name, field_info in response_model.model_fields.items():
            fields[field_name] = (field_info.annotation, field_info)

        # Create a new model that inherits from BaseModel directly (not IOSchema)
        # This avoids the docstring validation issue
        instructor_model = create_model(
            response_model.__name__,
            __base__=BaseModel,
            **fields,
        )

        # Copy over the docstring and other metadata
        instructor_model.__doc__ = response_model.__doc__

        return instructor_model

    async def get_response_async(
        self,
        messages: list[dict[str, str]],
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            messages (list[dict[str, str]], optional):
                The messages to send to the model. If not provided,
                builds from system prompt and memory.
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        import asyncio
        import time

        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ] + self.memory

        # GPT-5 series only supports temperature=1, override if needed
        temperature = 1.0 if self.model_name and self.model_name.startswith("gpt-5") else self.temperature

        # Debug logging with detailed metrics
        if self.debug:
            import json

            task = asyncio.current_task()
            task_name = task.get_name() if task else "unknown"
            from loguru import logger

            # Calculate prompt size (approximate token count)
            prompt_text = json.dumps(messages)
            prompt_chars = len(prompt_text)
            prompt_tokens_est = prompt_chars // 4  # Rough estimate: 1 token ≈ 4 chars

            logger.debug(
                f"[{task_name}] LLM call starting: model={self.model_name}, schema={response_model.__name__}, memory_msgs={len(self.memory)}, prompt_tokens~{prompt_tokens_est}",
            )

        start_time = time.time()
        api_call_start = None

        # Create timeout warning task
        async def log_slow_request():
            """Log warnings if request takes too long."""
            thresholds = [5, 10, 30, 60, 120, 300]  # seconds
            for threshold in thresholds:
                await asyncio.sleep(threshold)
                elapsed = time.time() - start_time
                if self.debug:
                    from loguru import logger

                    logger.warning(
                        f"[{task_name}] ⏱️  LLM call still running after {elapsed:.0f}s (threshold: {threshold}s)",
                    )

        # Start timeout warning task (will be cancelled when request completes)
        timeout_task = asyncio.create_task(log_slow_request())

        try:
            api_call_start = time.time()

            # Log the actual API call details
            if self.debug:
                from loguru import logger

                logger.debug(
                    f"[{task_name}] 🌐 Making OpenAI API call: base_url={self.base_url}, model={self.model_name}",
                )

            # Get timeout from config
            timeout = getattr(self.config, "llm_timeout", 45.0)

            # Use rate limiter to prevent OpenAI throttling of concurrent requests
            from akd.agents.data_search.utils.rate_limiter import get_rate_limiter

            rate_limiter = get_rate_limiter(max_concurrent_requests=2)

            if self.debug:
                from loguru import logger

                available = rate_limiter.available_slots()
                timeout_str = f"{timeout}s" if timeout is not None else "∞"
                logger.debug(
                    f"[{task_name}] 🚦 Rate limiter: {available}/{rate_limiter.max_concurrent} slots available, timeout={timeout_str}",
                )

            # Define API call as async function for timeout wrapping
            async def make_api_call():
                async with rate_limiter:
                    return await self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=temperature,
                        response_model=instructor_model,
                    )

            # Apply timeout if configured
            if timeout is not None:
                try:
                    response = await asyncio.wait_for(make_api_call(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Re-raise with clear marker for retry logic
                    elapsed = time.time() - start_time
                    raise RuntimeError(
                        f"LLM timeout: Call exceeded {timeout}s limit (elapsed: {elapsed:.1f}s, model: {self.model_name})",
                    )
            else:
                response = await make_api_call()

            # Cancel timeout warning task
            timeout_task.cancel()

            api_call_duration = time.time() - api_call_start

            if self.debug:
                from loguru import logger

                # Get response metadata if available
                response_tokens = (
                    getattr(response, "usage", {}).get("completion_tokens", "N/A")
                    if hasattr(response, "usage")
                    else "N/A"
                )
                request_id = getattr(response, "id", "N/A") if hasattr(response, "id") else "N/A"
                total_duration = time.time() - start_time

                # Log completion with metadata
                logger.debug(
                    f"[{task_name}] LLM call completed: api_time={api_call_duration:.2f}s, total={total_duration:.2f}s, response_tokens~{response_tokens}",
                )

                # Log request ID for support tickets
                if request_id != "N/A":
                    logger.debug(f"[{task_name}] OpenAI request_id: {request_id}")

                # Warn if request was unusually slow
                if api_call_duration > 10:
                    logger.warning(
                        f"[{task_name}] ⚠️  Unusually slow API call: {api_call_duration:.1f}s (>10s threshold)",
                    )

        except asyncio.CancelledError:
            # Timeout warning task cancelled, this is expected
            timeout_task.cancel()
            raise
        except Exception as e:
            # Cancel timeout warning task
            timeout_task.cancel()

            duration = time.time() - start_time
            if self.debug:
                from loguru import logger

                error_type = type(e).__name__
                error_msg = str(e)

                # Check for rate limit errors
                if "rate_limit" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.error(
                        f"[{task_name}] 🚫 RATE LIMIT ERROR after {duration:.2f}s: {error_msg}",
                    )
                elif "timeout" in error_msg.lower():
                    logger.error(
                        f"[{task_name}] ⏱️  TIMEOUT ERROR after {duration:.2f}s: {error_msg}",
                    )
                elif "connection" in error_msg.lower():
                    logger.error(
                        f"[{task_name}] 🔌 CONNECTION ERROR after {duration:.2f}s: {error_msg}",
                    )
                else:
                    logger.error(
                        f"[{task_name}] LLM call failed after {duration:.2f}s: {error_type}: {error_msg}",
                    )
            raise

        # Parse and validate response
        parse_start = time.time()
        response_data = response.model_dump()

        # Log validation and parsing time
        if self.debug:
            from loguru import logger

            parse_duration = time.time() - parse_start
            response_size = len(str(response_data))
            logger.debug(
                f"[{task_name}] Response parsed: parse_time={parse_duration:.3f}s, response_size={response_size}b, fields={list(response_data.keys())}",
            )

        response = response_model(**response_data)
        return cast(OutSchema, response)

    async def _arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """
        Runs the chat agent with the given user input asynchronously.

        Args:
            user_input (Optional[InputSchema]):
                The input from the user.
                If not provided, skips adding to memory.

        Returns:
            OutputSchema: The response from the chat agent.
        """

        # start fresh if no tracking required
        messages = [] if self.stateless else self.memory

        # if empty, add system message
        if not messages:
            messages.append(self._default_system_message())

        # add user message
        if params:
            messages.append(
                dict(
                    role="user",
                    content=params.model_dump_json(exclude={"type"}),
                ),
            )

        response = await self.get_response_async(
            messages=messages,
            response_model=self.output_schema,
        )

        messages.append(
            dict(
                role="assistant",
                content=response.model_dump_json(exclude={"type"}),
            ),
        )

        # update memory only if stateful
        if not self.stateless:
            self._memory = messages

        return response

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation to handle unpickleable attributes.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Manually copy attributes, re-initializing the client
        for k, v in self.__dict__.items():
            if k == "client":
                # Re-create the client instead of copying it
                setattr(
                    result,
                    k,
                    instructor.from_openai(
                        openai.AsyncOpenAI(
                            api_key=self.api_key,
                            base_url=str(self.base_url),
                        ),
                    ),
                )
            else:
                setattr(result, k, __import__("copy").deepcopy(v, memo))

        return result


class LiteLLMInstructorBaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](InstructorBaseAgent):
    """InstructorBaseAgent with LiteLLM integration and automatic message trimming.

    This agent extends InstructorBaseAgent to use LiteLLM with automatic message trimming
    to prevent token limit errors. It maintains full compatibility with the base class
    while adding intelligent context management.
    """

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        # Initialize base class but we'll replace the client
        super().__init__(config=config, debug=debug)

        # Replace instructor client with LiteLLM version
        self.client = instructor.from_litellm(acompletion)

    async def get_response_async(
        self,
        messages: list[dict[str, str]],
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously with automatic message trimming.

        Args:
            messages (list[dict[str, str]], optional):
                The messages to send to the model. If not provided,
                builds from system prompt and memory.
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        # Trim messages if enabled to prevent token limit errors
        if self.enable_trimming:
            messages = trim_messages(
                messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                trim_ratio=self.trim_ratio,
            )

        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            response_model=instructor_model,
            api_base=str(self.base_url).rstrip("/") if self.base_url else None,
            api_key=self.api_key,
            num_retries=self.num_retries,
        )

        response_data = response.model_dump()
        response = response_model(**response_data)
        return cast(OutSchema, response)
