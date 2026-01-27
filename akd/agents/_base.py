from __future__ import annotations

import copy
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import instructor
import openai
from litellm import acompletion
from litellm.utils import get_model_info, supports_reasoning, trim_messages
from loguru import logger
from pydantic import AnyUrl, BaseModel, Field, create_model, model_validator

from akd._base import (
    AbstractBase,
    BaseConfig,
    InputSchema,
    OutputSchema,
    ParamExposureMixin,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallingMixin,
)
from akd._base.errors import (
    MaxToolCallsExceeded,
    MaxToolIterationsExceeded,
    UnexpectedModelBehavior,
)
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT
from akd.tools._base import BaseTool
from akd.tools.output import OutputTool
from akd.utils import PartialModel


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
    max_tool_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tool calling iterations (ReAct loop turns) before stopping",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum total tool calls across the run (None = unlimited)",
    )
    reflection_prompt: str | None = Field(
        default=None,
        description="Reflection prompt injected after tool results to force reasoning. If None, no reflection step.",
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
        Enhanced system prompt with agent description.

        The description includes IO field hints if io_hints=True (default).

        Returns:
            str: System prompt with agent description if available.
        """
        content = self.system_prompt

        # Add agent description (includes IO hints if io_hints=True)
        if self.description:
            content += f"\n\nAGENT DESCRIPTION:\n{self.description}"
        return content

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Convert tools to function calling format."""
        return [tool.as_tool_definition() for tool in self.tools]

    def _prepare_messages(self, params: InputSchema | None = None) -> list[dict[str, Any]]:
        """Prepare working messages for a run.

        Returns a COPY of memory (stateful) or fresh list (stateless).
        Subclasses can override to add custom behavior (e.g., trimming).

        Args:
            params: Optional input parameters to add as user message.

        Returns:
            Working message list.
        """
        raise NotImplementedError("Subclasses must implement _prepare_messages")

    def _persist_memory(self, messages: list[dict[str, Any]]) -> None:
        """Persist messages to memory.

        This is the SINGLE point where memory should be persisted.
        Subclasses can override to add custom behavior (e.g., trimming).

        Args:
            messages: The complete message list to persist.
        """
        raise NotImplementedError("Subclasses must implement _persist_memory")

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

    def _prepare_messages(self, params: InSchema | None = None) -> list[dict[str, Any]]:
        """Prepare working messages for a run.

        Returns a COPY of memory (stateful) or fresh list (stateless).

        Args:
            params: Optional input parameters to add as user message.

        Returns:
            Working message list (copy of memory or fresh).
        """
        if self.stateless:
            messages = []
        else:
            messages = list(self._memory)  # Shallow copy

        if not messages:
            messages.append(self._default_system_message())

        if params:
            messages.append(
                {
                    "role": "user",
                    "content": params.model_dump_json(exclude={"type"}),
                },
            )

        return messages

    def _persist_memory(self, messages: list[dict[str, Any]]) -> None:
        """Persist messages to memory.

        Called only on successful completion.

        Args:
            messages: The complete message list to persist.
        """
        if self.stateless:
            return

        self._memory = messages

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
            params: The input from the user.

        Returns:
            OutputSchema: The response from the chat agent.
        """
        # Prepare working messages (copy of memory or fresh list)
        messages = self._prepare_messages(params)

        response = await self.get_response_async(
            messages=messages,
            response_model=self.output_schema,
        )

        # Add assistant response
        messages.append(
            {
                "role": "assistant",
                "content": response.model_dump_json(exclude={"type"}),
            },
        )

        # Single point of memory persistence
        self._persist_memory(messages)

        return response


class LiteLLMInstructorBaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](ToolCallingMixin, InstructorBaseAgent):
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

    def _persist_memory(self, messages: list[dict[str, Any]]) -> None:
        """Persist messages to memory with trimming.

        This is the SINGLE point where memory should be persisted.
        Called only on successful completion, not mid-run.

        Trimming is applied here to keep _memory bounded, preventing
        unbounded growth across multiple runs.

        Args:
            messages: The complete message list to persist.
        """
        if self.stateless:
            return

        if self.enable_trimming:
            messages = trim_messages(
                messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                trim_ratio=self.trim_ratio,
            )

        self._memory = messages

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

    async def _arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """Run agent with optional tool calling support.

        If tools are configured, uses the tool loop (ReAct pattern).
        Otherwise, uses direct LLM call with structured output.

        Args:
            params: Input parameters matching input_schema.

        Returns:
            Output matching output_schema.
        """
        class_name = self.__class__.__name__

        # Prepare working messages (copy of memory or fresh list)
        messages = self._prepare_messages(params)

        output: OutSchema | None = None

        if self.tools:
            # === TOOL CALLING MODE ===
            # Use _run_tool_loop and consume events to get final output
            # Note: Tool loop mutates `messages` in place during iterations
            run_context = {"run_id": uuid.uuid4().hex[:8]}
            async for event in self._run_tool_loop(
                messages,
                class_name,
                run_context,
            ):
                if event.event_type == StreamEventType.COMPLETED:
                    output = event.output
                    break

            if output is None:
                raise UnexpectedModelBehavior("Tool loop completed without producing output")

        else:
            # === DIRECT MODE (no tools) ===
            # Apply trimming for LLM call (get_response_async also trims, but be explicit)
            if self.enable_trimming:
                trimmed_messages = trim_messages(
                    messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    trim_ratio=self.trim_ratio,
                )
            else:
                trimmed_messages = messages

            output = await self.get_response_async(
                messages=trimmed_messages,
                response_model=self.output_schema,
            )

        # Add final assistant message
        messages.append(
            {
                "role": "assistant",
                "content": output.model_dump_json(exclude={"type"}),
            },
        )

        # Single point of memory persistence (with trimming)
        self._persist_memory(messages)

        return output

    async def _stream_llm_response(
        self,
        messages: list[dict[str, str]],
        response_model: type[OutputSchema] | None = None,
        token_batch_size: int = 10,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM response with thinking tokens and validated partial output.

        Uses raw LiteLLM streaming to access thinking tokens (reasoning_content).

        Args:
            messages: Chat messages
            response_model: Output schema (defaults to self.output_schema)
            token_batch_size: Batch N tokens before emitting STREAMING event (default=1)

        Yields:
            - {"type": StreamEventType.STREAMING, "token": str} for batched raw tokens
            - {"type": StreamEventType.THINKING, "content": str} for reasoning tokens
            - {"type": StreamEventType.PARTIAL, "partial": PartialModel} for validated partials
            - {"type": StreamEventType.COMPLETED, "output": OutputSchema} for final output
        """
        response_model = response_model or self.output_schema
        PartialResponseModel = PartialModel[response_model]

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
                yield {"type": StreamEventType.THINKING, "content": reasoning}

            # Content tokens - batch, then accumulate and validate as partial
            if delta.content:
                token_buffer += delta.content
                accumulated += delta.content

                # Emit batched tokens
                if len(token_buffer) >= token_batch_size:
                    yield {"type": StreamEventType.STREAMING, "token": token_buffer}
                    token_buffer = ""

                # Try to validate as partial
                parsed = self._try_parse_json(accumulated)
                if parsed and parsed != last_partial_dict:
                    last_partial_dict = parsed
                    try:
                        partial = PartialResponseModel.model_validate(parsed)
                        yield {"type": StreamEventType.PARTIAL, "partial": partial}
                    except Exception:
                        pass  # Skip invalid partials

        # Emit remaining tokens
        if token_buffer:
            yield {"type": StreamEventType.STREAMING, "token": token_buffer}

        # Final validation
        output = response_model.model_validate_json(accumulated)
        yield {"type": StreamEventType.COMPLETED, "output": output}

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        class_name: str,
        run_context: dict[str, Any],
        token_batch_size: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """ReAct loop: call LLM with tools until final answer.

        Streams thinking/reasoning tokens in real-time.
        Tool calls are IO events (TOOL_CALLING, TOOL_RESULT).

        Args:
            messages: Conversation history (mutated in place)
            class_name: For event source
            run_context: For event context
            token_batch_size: Batch N characters before emitting STREAMING event

        Yields:
            StreamEvent: STREAMING, THINKING, TOOL_CALLING, TOOL_RESULT, COMPLETED
        """
        # For partial output validation (only for final answer)
        PartialResponseModel = PartialModel[self.output_schema]

        # Add output tool for structured final answer (Pydantic AI pattern)
        output_tool = OutputTool(self.output_schema)
        # Add to tools list so _find_tool can find it
        all_tool_instances = self.tools + [output_tool]
        all_tool_definitions = self.tool_definitions + [output_tool.as_tool_definition()]

        total_tool_calls = 0  # Track total tool calls for max_tool_calls limit

        for iteration in range(self.max_tool_iterations):
            # Call LLM with tools (no response_format - output tool handles structured output)
            completion_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "tools": all_tool_definitions,
                "temperature": self.temperature,
                "api_base": str(self.base_url).rstrip("/") if self.base_url else None,
                "api_key": self.api_key,
                "drop_params": True,
                "stream": True,
            }
            if self.reasoning_effort:
                completion_kwargs["reasoning_effort"] = self.reasoning_effort

            response = await acompletion(**completion_kwargs)

            # Accumulate streamed response
            accumulated_content = ""
            token_buffer = ""  # Batch tokens before emitting
            last_partial_dict: dict[str, Any] | None = None  # For deduplicating PARTIAL events
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
            buffered_partials: list[StreamEvent] = []  # Buffer partials until we know it's final

            async for chunk in response:
                delta = chunk.choices[0].delta

                # Stream reasoning/thinking tokens (o1, o3, Claude extended thinking)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta,
                    "thinking",
                    None,
                )
                if reasoning:
                    yield StreamEvent(
                        event_type=StreamEventType.THINKING,
                        source=class_name,
                        message=reasoning,
                        data={"streaming": True, "reasoning_content": reasoning},
                        context=run_context,
                    )

                # Batch and stream content tokens
                if delta.content:
                    accumulated_content += delta.content
                    token_buffer += delta.content

                    # Emit batched tokens when buffer is full
                    if len(token_buffer) >= token_batch_size:
                        yield StreamEvent(
                            event_type=StreamEventType.STREAMING,
                            source=class_name,
                            message=token_buffer,
                            data={"token": token_buffer},
                            context=run_context,
                        )
                        token_buffer = ""

                    # Try to validate as partial (buffer until we know it's final)
                    parsed = self._try_parse_json(accumulated_content)
                    if parsed and parsed != last_partial_dict:
                        last_partial_dict = parsed
                        try:
                            partial = PartialResponseModel.model_validate(parsed)
                            buffered_partials.append(
                                StreamEvent(
                                    event_type=StreamEventType.PARTIAL,
                                    source=class_name,
                                    message="Partial output",
                                    data={"partial": partial},
                                    context=run_context,
                                ),
                            )
                        except Exception:
                            pass  # Skip invalid partials

                # Accumulate tool calls (may be split across chunks)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": tc.function.arguments if tc.function else "",
                            }
                        else:
                            # Append to existing tool call
                            if tc.id:
                                accumulated_tool_calls[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                accumulated_tool_calls[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

            # Emit remaining tokens in buffer
            if token_buffer:
                yield StreamEvent(
                    event_type=StreamEventType.STREAMING,
                    source=class_name,
                    message=token_buffer,
                    data={"token": token_buffer},
                    context=run_context,
                )

            # After streaming: check for tool calls
            if not accumulated_tool_calls:
                # No tool calls - model should have called final_answer but didn't
                # Fallback: try to parse content as JSON
                for partial_event in buffered_partials:
                    yield partial_event

                if not accumulated_content:
                    raise UnexpectedModelBehavior("LLM returned empty response without tool calls")

                logger.warning(
                    "Model gave direct response instead of calling final_answer tool. "
                    "Attempting to parse as JSON fallback.",
                )
                output = self.output_schema.model_validate_json(accumulated_content)

                yield StreamEvent(
                    event_type=StreamEventType.COMPLETED,
                    source=class_name,
                    message=f"Completed {class_name}",
                    data={"output": output},
                    context=run_context,
                )
                return

            # Convert accumulated tool calls to ToolCall objects
            tool_calls: list[ToolCall] = []
            tool_calls_for_message: list[dict[str, Any]] = []
            for idx in sorted(accumulated_tool_calls.keys()):
                tc_data = accumulated_tool_calls[idx]
                tool_call = ToolCall(
                    tool_call_id=tc_data["id"],
                    tool_name=tc_data["name"],
                    arguments=json.loads(tc_data["arguments"]),
                )
                tool_calls.append(tool_call)
                tool_calls_for_message.append(
                    {
                        "id": tc_data["id"],
                        "type": "function",
                        "function": {"name": tc_data["name"], "arguments": tc_data["arguments"]},
                    },
                )
                yield StreamEvent(
                    event_type=StreamEventType.TOOL_CALLING,
                    source=class_name,
                    message=f"Calling {tool_call.tool_name}",
                    data=tool_call.model_dump(),
                    context=run_context,
                )

            # Check max_tool_calls limit before executing
            if self.max_tool_calls is not None:
                if total_tool_calls + len(tool_calls) > self.max_tool_calls:
                    # Persist memory before raising (conversation history is still valid)
                    self._persist_memory(messages)
                    raise MaxToolCallsExceeded(
                        f"Exceeded {self.max_tool_calls} total tool calls "
                        f"(current: {total_tool_calls}, requested: {len(tool_calls)})",
                    )

            # Execute all tools in parallel (including OutputTool if called)
            # Temporarily include OutputTool so _find_tool can find it
            original_tools = self.tools
            self.tools = all_tool_instances
            try:
                results = await self._execute_tools_parallel(tool_calls)
            finally:
                self.tools = original_tools

            total_tool_calls += len(tool_calls)

            # Check if final_answer was called - signals completion
            for result in results:
                if result.tool_name == "final_answer":
                    # Validation done in _execute_tool via input_schema(**arguments)
                    # result.content is dict from model_dump(), reconstruct the model
                    output = self.output_schema.model_validate(result.content)
                    yield StreamEvent(
                        event_type=StreamEventType.COMPLETED,
                        source=class_name,
                        message=f"Completed {class_name}",
                        data={"output": output},
                        context=run_context,
                    )
                    return

            # Yield TOOL_RESULT events for regular tools
            for result in results:
                yield StreamEvent(
                    event_type=StreamEventType.TOOL_RESULT,
                    source=class_name,
                    message=f"Result from {result.tool_name}",
                    data=result.model_dump(),
                    context=run_context,
                )

            # Add assistant message (reconstructed from stream) and tool results to history
            messages.append(
                {
                    "role": "assistant",
                    "content": accumulated_content or None,
                    "tool_calls": tool_calls_for_message,
                },
            )
            for result in results:
                content = json.dumps(result.content) if result.content else (result.error or "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": content,
                    },
                )

            # Note: Memory persistence moved to caller (_arun/_astream)
            # Tool loop only mutates working messages, doesn't persist

            # Inject reflection prompt if configured (forces reasoning before next iteration)
            if self.reflection_prompt:
                messages.append(
                    {
                        "role": "user",
                        "content": self.reflection_prompt,
                    },
                )
                yield StreamEvent(
                    event_type=StreamEventType.THINKING,
                    source=class_name,
                    message="Reflecting on results...",
                    data={"reflection_prompt": self.reflection_prompt},
                    context=run_context,
                )

        # Loop exhausted - persist memory before raising (conversation history is still valid)
        self._persist_memory(messages)
        raise MaxToolIterationsExceeded(f"Exceeded {self.max_tool_iterations} tool iterations")

    async def _astream(
        self,
        params: InSchema,
        context: dict[str, Any] | None = None,
        token_batch_size: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Internal streaming with thinking tokens and partial output.

        Emits:
        - STREAMING events for raw tokens (batched)
        - THINKING events for reasoning tokens (Claude extended thinking, o1)
        - PARTIAL events for partial structured output as it streams

        Note: Input/output validation is handled by parent astream() method.

        Args:
            params: Input parameters (already validated by astream())
            context: Execution context (node_id, query, etc.)
            token_batch_size: Batch N characters before emitting STREAMING event (default=10)
            **kwargs: Additional keyword arguments

        Yields:
            StreamEvent: STARTING, RUNNING, STREAMING, PARTIAL, then COMPLETED or FAILED

        Example:
            async for event in agent.astream(input_data, token_batch_size=20):
                match event.event_type:
                    case StreamEventType.STREAMING:
                        print(event.token, end="")  # Raw tokens
                    case StreamEventType.THINKING:
                        print(f"Reasoning: {event.thinking_content}")
                    case StreamEventType.PARTIAL:
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
            # Prepare working messages (copy of memory or fresh list)
            messages = self._prepare_messages(params)

            yield StreamEvent(
                event_type=StreamEventType.RUNNING,
                source=class_name,
                message=f"Running {class_name}",
                context=run_context,
            )

            # Route: tool calling mode vs streaming mode
            output = None

            if self.tools:
                # === TOOL CALLING MODE ===
                # Yields STREAMING, THINKING, TOOL_CALLING, TOOL_RESULT, then COMPLETED
                # Tool loop mutates `messages` in place during iterations
                async for event in self._run_tool_loop(
                    messages,
                    class_name,
                    run_context,
                    token_batch_size=token_batch_size,
                ):
                    if event.event_type == StreamEventType.COMPLETED:
                        output = event.output
                    yield event

                # Persistence happens after loop completes (single point)
                if output is not None:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": output.model_dump_json(exclude={"type"}),
                        },
                    )
                    self._persist_memory(messages)

            else:
                # === STREAMING MODE (no tools) ===
                # Yields STREAMING, THINKING, PARTIAL, then COMPLETED
                async for chunk in self._stream_llm_response(messages, token_batch_size=token_batch_size):
                    if chunk["type"] == StreamEventType.STREAMING:
                        yield StreamEvent(
                            event_type=StreamEventType.STREAMING,
                            source=class_name,
                            data={"token": chunk["token"]},
                            context=run_context,
                        )
                    elif chunk["type"] == StreamEventType.THINKING:
                        yield StreamEvent(
                            event_type=StreamEventType.THINKING,
                            source=class_name,
                            message="Reasoning...",
                            data={"thinking_content": chunk["content"]},
                            context=run_context,
                        )
                    elif chunk["type"] == StreamEventType.PARTIAL:
                        yield StreamEvent(
                            event_type=StreamEventType.PARTIAL,
                            source=class_name,
                            message="Partial...",
                            data={"partial_output": chunk["partial"]},
                            context=run_context,
                        )
                    elif chunk["type"] == StreamEventType.COMPLETED:
                        output = chunk["output"]

                if output is None:
                    raise UnexpectedModelBehavior("No output received from LLM")

                # Add final assistant message and persist (single point)
                messages.append(
                    {
                        "role": "assistant",
                        "content": output.model_dump_json(exclude={"type"}),
                    },
                )
                self._persist_memory(messages)

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
