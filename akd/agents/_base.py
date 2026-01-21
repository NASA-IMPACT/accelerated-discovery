from __future__ import annotations

import copy
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, cast

import instructor
import openai
from litellm import acompletion
from litellm.utils import get_model_info, supports_reasoning, trim_messages
from loguru import logger
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from akd._base import (
    AbstractBase,
    BaseConfig,
    InputSchema,
    OutputSchema,
    ParamExposureMixin,
)
from akd._base.streaming import StreamEvent, StreamEventType
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT
from akd.tools._base import BaseTool


class BaseAgentConfig(BaseConfig):
    """Configuration class for base agents."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

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

    # Reasoning parameters (for o1, o3, Claude extended thinking, etc.)
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Reasoning effort level for supported models (o1, o3, Claude, etc.)",
    )
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = Field(
        default=None,
        description="How to present reasoning output in response",
    )
    tools: list[BaseTool] = Field(
        default_factory=list,
        description="List of tools available to the agent",
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

    @model_validator(mode="after")
    def validate_reasoning_params(self):
        """Warn if reasoning params set for non-reasoning models."""
        if not self.model_name or not (self.reasoning_effort or self.reasoning_summary):
            return self

        try:
            if not supports_reasoning(model=self.model_name):
                logger.warning(f"Model '{self.model_name}' may not support reasoning params")
        except Exception:
            pass

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
](AbstractBase, ParamExposureMixin):
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

    def __deepcopy__(self, memo: dict[int, Any]) -> InstructorBaseAgent:
        """Custom deepcopy that recreates the client instead of copying it.

        The instructor/openai client contains httpx connections and async state
        that cannot be properly deepcopied. We copy all other attributes and
        recreate the client fresh.

        Args:
            memo: Dictionary of already copied objects (used by copy.deepcopy).

        Returns:
            A deep copy of this agent with a fresh client.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except the client
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        # Recreate the client fresh (api_key/base_url are dynamic properties from metaclass)
        api_key = getattr(result, "api_key", None)
        base_url = getattr(result, "base_url", None)
        result.client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=api_key,
                base_url=str(base_url) if base_url else None,
            ),
        )

        return result

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

    def _create_partial_model(self, model: type[OutputSchema]) -> type[BaseModel]:
        """Create partial model with all fields Optional for streaming validation."""
        fields = {}
        for name, info in model.model_fields.items():
            fields[name] = (Optional[info.annotation], Field(default=None))
        return create_model(f"Partial{model.__name__}", **fields)

    def _build_response_format_schema(self, model: type[OutputSchema]) -> dict[str, Any]:
        """Build JSON schema for response_format (OpenAI strict mode)."""
        schema = model.model_json_schema()
        schema.pop("$defs", None)
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        return schema

    def _try_parse_json(self, content: str) -> dict[str, Any] | None:
        """Try to parse content as JSON."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

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
        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            response_model=instructor_model,
        )

        response_data = response.model_dump()
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

    def __deepcopy__(self, memo: dict[int, Any]) -> LiteLLMInstructorBaseAgent:
        """Custom deepcopy that recreates the LiteLLM client instead of copying it.

        Args:
            memo: Dictionary of already copied objects (used by copy.deepcopy).

        Returns:
            A deep copy of this agent with a fresh LiteLLM client.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except the client
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        # Recreate the LiteLLM client fresh
        result.client = instructor.from_litellm(acompletion)

        return result

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

        # Build kwargs for completion call
        completion_kwargs = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "response_model": instructor_model,
            "api_base": str(self.base_url).rstrip("/") if self.base_url else None,
            "api_key": self.api_key,
            "num_retries": self.num_retries,
            "drop_params": True,  # Safely drop unsupported params at runtime
        }

        # Only add reasoning params if set (don't send None)
        if self.reasoning_effort:
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary:
            completion_kwargs["reasoning_summary"] = self.reasoning_summary

        response = await self.client.chat.completions.create(**completion_kwargs)

        response_data = response.model_dump()
        response = response_model(**response_data)
        return cast(OutSchema, response)

    async def _stream_llm_response(
        self,
        messages: list[dict[str, str]],
        response_model: type[OutputSchema] | None = None,
        token_batch_size: int = 1,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM response with thinking tokens and validated partial output.

        Uses raw LiteLLM streaming to access thinking tokens (reasoning_content).

        Args:
            messages: Chat messages
            response_model: Output schema (defaults to self.output_schema)
            token_batch_size: Batch N tokens before emitting STREAMING event (default=1)

        Yields:
            - {"type": "streaming", "token": str} for batched raw tokens
            - {"type": "thinking", "content": str} for reasoning tokens
            - {"type": "partial", "partial": PartialModel} for validated partials
            - {"type": "final", "output": OutputSchema} for final output
        """
        response_model = response_model or self.output_schema
        PartialModel = self._create_partial_model(response_model)

        completion_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "api_base": str(self.base_url).rstrip("/") if self.base_url else None,
            "api_key": self.api_key,
            "drop_params": True,  # Allow unsupported params to be dropped
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": self._build_response_format_schema(response_model),
                },
            },
        }

        if self.reasoning_effort:
            completion_kwargs["reasoning_effort"] = self.reasoning_effort

        accumulated = ""
        token_buffer = ""
        last_partial_dict = None

        response = await acompletion(**completion_kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta

            # Thinking tokens (Claude extended thinking, o1 reasoning)
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta,
                "thinking",
                None,
            )
            if reasoning:
                yield {"type": "thinking", "content": reasoning}

            # Content tokens - batch, then accumulate and validate as partial
            if delta.content:
                token_buffer += delta.content
                accumulated += delta.content

                # Emit batched tokens
                if len(token_buffer) >= token_batch_size:
                    yield {"type": "streaming", "token": token_buffer}
                    token_buffer = ""

                # Try to validate as partial
                parsed = self._try_parse_json(accumulated)
                if parsed and parsed != last_partial_dict:
                    last_partial_dict = parsed
                    try:
                        partial = PartialModel.model_validate(parsed)
                        yield {"type": "partial", "partial": partial}
                    except Exception:
                        pass  # Skip invalid partials

        # Emit remaining tokens
        if token_buffer:
            yield {"type": "streaming", "token": token_buffer}

        # Final validation
        output = response_model.model_validate_json(accumulated)
        yield {"type": "final", "output": output}

    async def astream(
        self,
        params: InSchema,
        context: dict[str, Any] | None = None,
        token_batch_size: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution with thinking tokens and partial output.

        Overrides base astream() to emit:
        - STREAMING events for raw tokens (batched)
        - THINKING events for reasoning tokens (Claude extended thinking, o1)
        - GENERATING events for partial structured output as it streams

        Args:
            params: Input parameters
            context: Execution context (node_id, query, etc.)
            token_batch_size: Batch N characters before emitting STREAMING event (default=10)
            **kwargs: Additional keyword arguments

        Yields:
            StreamEvent: STARTING, RUNNING, STREAMING, GENERATING, then COMPLETED or FAILED

        Example:
            async for event in agent.astream(input_data, token_batch_size=20):
                match event.event_type:
                    case StreamEventType.STREAMING:
                        print(event.token, end="")  # Raw tokens
                    case StreamEventType.THINKING:
                        print(f"Reasoning: {event.thinking_content}")
                    case StreamEventType.GENERATING:
                        print(f"Partial: {event.partial_output}")
                    case StreamEventType.COMPLETED:
                        print(f"Final: {event.output}")
                    case StreamEventType.FAILED:
                        print(f"Error: {event.error}")
        """
        class_name = self.__class__.__name__

        # Auto-generate run_id for event correlation
        run_context = context.copy() if context else {}
        if "run_id" not in run_context:
            run_context["run_id"] = uuid.uuid4().hex[:8]

        yield StreamEvent(
            event_type=StreamEventType.STARTING,
            source=class_name,
            message=f"Starting {class_name}",
            context=run_context,
        )

        try:
            # Validate input
            params = self._validate_input(params)

            # Build messages (same logic as _arun)
            messages = [] if self.stateless else self.memory
            if not messages:
                messages.append(self._default_system_message())

            if params:
                messages.append(
                    {
                        "role": "user",
                        "content": params.model_dump_json(exclude={"type"}),
                    },
                )

            # Apply trimming if enabled
            if self.enable_trimming:
                messages = trim_messages(
                    messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    trim_ratio=self.trim_ratio,
                )

            yield StreamEvent(
                event_type=StreamEventType.RUNNING,
                source=class_name,
                message=f"Running {class_name}",
                context=run_context,
            )

            # Stream LLM response with raw tokens, thinking, and partial output
            final_output = None
            async for chunk in self._stream_llm_response(messages, token_batch_size=token_batch_size):
                if chunk["type"] == "streaming":
                    yield StreamEvent(
                        event_type=StreamEventType.STREAMING,
                        source=class_name,
                        data={"token": chunk["token"]},
                        context=run_context,
                    )
                elif chunk["type"] == "thinking":
                    yield StreamEvent(
                        event_type=StreamEventType.THINKING,
                        source=class_name,
                        message="Reasoning...",
                        data={"thinking_content": chunk["content"]},
                        context=run_context,
                    )
                elif chunk["type"] == "partial":
                    yield StreamEvent(
                        event_type=StreamEventType.GENERATING,
                        source=class_name,
                        message="Generating...",
                        data={"partial_output": chunk["partial"]},
                        context=run_context,
                    )
                elif chunk["type"] == "final":
                    final_output = chunk["output"]

            if final_output is None:
                raise ValueError("No output received from LLM")

            # Validate output
            output = self._validate_output(final_output)

            # Update memory if stateful
            messages.append(
                {
                    "role": "assistant",
                    "content": output.model_dump_json(exclude={"type"}),
                },
            )
            if not self.stateless:
                self._memory = messages

            yield StreamEvent(
                event_type=StreamEventType.COMPLETED,
                source=class_name,
                message=f"Completed {class_name}",
                data={"output": output},
                context=run_context,
            )

        except Exception as e:
            yield StreamEvent(
                event_type=StreamEventType.FAILED,
                source=class_name,
                message=f"Failed: {e!s}",
                data={"error": str(e), "error_type": type(e).__name__},
                context=run_context,
            )
            raise
