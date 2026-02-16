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
from litellm.utils import get_model_info, supports_reasoning
from loguru import logger
from pydantic import AnyUrl, BaseModel, Field, create_model, model_validator

from akd._base import (
    AbstractBase,
    BaseConfig,
    InputSchema,
    Memory,
    OutputSchema,
    ParamExposureMixin,
    RunContext,
    StreamEvent,
    StreamEventType,
    TextInput,
    TextOutput,
    ToolCall,
    ToolCallingMixin,
)
from akd._base.errors import (
    MaxToolCallsExceeded,
    MaxToolIterationsExceeded,
    UnexpectedModelBehavior,
)
from akd._base.streaming import (
    CompletedEvent,
    CompletedEventData,
    FailedEvent,
    FailedEventData,
    HumanInputRequiredEvent,
    HumanInputRequiredEventData,
    HumanResponseEvent,
    HumanResponseEventData,
    PartialEventData,
    PartialOutputEvent,
    RunningEvent,
    StartingEvent,
    StartingEventData,
    StreamingEventData,
    StreamingTokenEvent,
    ThinkingEvent,
    ThinkingEventData,
    ToolCallingEvent,
    ToolCallingEventData,
    ToolResultEvent,
    ToolResultEventData,
)
from akd._base.structures import RunUsage
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT
from akd.tools._base import BaseTool
from akd.tools.human import HumanTool, HumanToolInput
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
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    hitl_instruction: str = Field(
        default="""HUMAN INTERACTION\nWhen you need clarification, confirmation, or any missing information from the user, you CAN use your available tools to ask them.""".strip(),
        description="Instruction to inject for Human-in-the-Loop (HITL) scenarios",
    )
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
        default=25,
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

    @model_validator(mode="before")
    @classmethod
    def inject_hitl_instruction(cls, data: Any) -> Any:
        """Inject HITL instruction into system prompt if HumanTool is present.

        Uses mode="before" so it sees raw tools before downstream field_validators
        (e.g., OpenAIBaseAgentConfig) convert them to non-BaseTool formats.
        """
        if not isinstance(data, dict):
            logger.warning(f"inject_hitl_instruction: expected dict, got {type(data).__name__}, skipping")
            return data

        hitl_instruction = data.get("hitl_instruction", cls.model_fields["hitl_instruction"].default)
        system_prompt = data.get("system_prompt", cls.model_fields["system_prompt"].default)

        if (
            hitl_instruction
            and system_prompt
            and any(isinstance(t, HumanTool) for t in data.get("tools", []))
            and hitl_instruction not in system_prompt
        ):
            data["system_prompt"] = f"{system_prompt}\n\n{hitl_instruction}"

        return data

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

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        memory: Memory | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.memory = memory or Memory()

    def reset_memory(self) -> None:
        """Clear memory."""
        self.memory.clear()

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

    def _default_system_message(self) -> dict[str, str]:
        """Return default system message."""
        return {
            "role": "system",
            "content": self._system_prompt,
        }

    async def arun(
        self,
        params: InSchema,
        run_context: RunContext | None = None,
        **kwargs,
    ) -> OutSchema:
        """Run the agent with the provided parameters asynchronously.

        Args:
            params: The structured input parameters for the agent.
            run_context: Optional RunContext for execution context
                (messages, human_response, run_id, etc.)
            **kwargs: Additional keyword arguments.

        Returns:
            Output matching the agent's output_schema.
        """
        # only reference, no copy
        run_context = run_context or RunContext()
        run_context.run_id = run_context.run_id or uuid.uuid4().hex[:8]
        output = await super().arun(params, run_context=run_context, **kwargs)
        # attach run_context (with accumulated usage) to final output
        object.__setattr__(output, "run_context", run_context)
        return output

    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution with run_context initialization.

        Centralizes run_context creation and run_id assignment,
        then delegates to parent astream().
        """
        # only reference, no copy
        run_context = run_context or RunContext()
        run_context.run_id = run_context.run_id or uuid.uuid4().hex[:8]
        async for event in super().astream(params, run_context=run_context, **kwargs):
            yield event

    async def achat(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> TextOutput:
        """Convenience method for simple text-based conversations.

        Converts any agent into a chat-style interface by routing through _arun(),
        preserving tool calling, guardrails, and streaming support. Wraps input
        in TextInput and converts the agent's response to TextOutput.

        Note: Messages in memory will be JSON-serialized (e.g. {"content": "hello"})
        rather than plain text, since this goes through the standard _arun() pipeline.

        Args:
            params: The input content. If already a TextInput, used directly.
                    Otherwise, stringified and wrapped in TextInput.
            run_context: Optional context for resumption after human input.

        Returns:
            TextOutput: The agent's text response.

        Example:
            # Works with any agent, not just TextInput/TextOutput agents
            response = await search_agent.achat("Find papers on quantum computing")
            print(response.content)

            # Multi-turn with stateless=False
            agent = MyAgent(config=BaseAgentConfig(stateless=False))
            await agent.achat("My name is Alice")
            response = await agent.achat("What's my name?")  # response.content == "Alice"
        """
        text_input = params if isinstance(params, TextInput) else TextInput(content=str(params))
        # only reference, no copy
        run_context = run_context or RunContext()
        run_context.run_id = run_context.run_id or uuid.uuid4().hex[:8]
        result = await self._arun(text_input, run_context=run_context, **kwargs)
        object.__setattr__(result, "run_context", run_context)
        if isinstance(result, TextOutput):
            return result
        return TextOutput(content=result._response or str(result))

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
        memory: Memory | None = None,
        debug: bool = False,
    ) -> None:
        import warnings

        if not isinstance(self, LiteLLMInstructorBaseAgent):
            warnings.warn(
                "InstructorBaseAgent is deprecated. Use LiteLLMInstructorBaseAgent instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(config=config, memory=memory, debug=debug)

        # Create the OpenAI client
        self.client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=str(self.base_url),
            ),
        )

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
        run_context: RunContext | None = None,
        **kwargs,
    ) -> OutSchema:
        """
        Runs the chat agent with the given user input asynchronously.

        Args:
            params: The input from the user.
            run_context: Optional RunContext with messages, human_response, etc.

        Returns:
            OutputSchema: The response from the chat agent.
        """
        async with self.memory.asession(
            stateless=self.stateless,
            run_context=run_context,
            enable_trimming=self.enable_trimming,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            trim_ratio=self.trim_ratio,
        ) as messages:
            # Add system message if empty
            if not messages:
                messages.append(self._default_system_message())

            # Add user message (skip if resuming with human response)
            if params and not (run_context and run_context.human_response):
                messages.append(
                    {
                        "role": "user",
                        "content": params.model_dump_json(exclude={"type"}),
                    },
                )

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
        memory: Memory | None = None,
        debug: bool = False,
    ) -> None:
        # Initialize base class but we'll replace the client
        super().__init__(config=config, memory=memory, debug=debug)

        # Replace instructor client with LiteLLM version
        self.client = instructor.from_litellm(acompletion)

    def _post_init(self):
        super()._post_init()

        # reformat to response format for openai gpt-5* models
        # only when reasoning is enabled
        # only applies to litellm so
        if (
            self.config.model_name.startswith("gpt-5")
            and self.config.reasoning_effort
            and self.config.reasoning_summary
            and supports_reasoning(model=self.config.model_name)
        ):
            logger.info(
                f"Reformatting model name to {self.config.model_name} to openai/responses/{self.config.model_name}",
            )
            self.config.model_name = f"openai/responses/{self.config.model_name}"

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
        run_context: RunContext | None = None,
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

        response, completion = await self.client.chat.completions.create_with_completion(
            **completion_kwargs,
        )

        # Capture token usage from the raw completion
        if run_context is not None:
            run_context.usage += self._extract_usage(completion)

        response_data = response.model_dump()
        response = response_model(**response_data)
        return cast(OutSchema, response)

    @staticmethod
    def _extract_usage(completion: Any) -> RunUsage:
        """Extract token usage from a raw LLM completion.

        Args:
            completion: Raw completion object from create_with_completion.

        Returns:
            RunUsage with extracted token counts (zeros if unavailable).
        """
        run_usage = RunUsage()
        usage = getattr(completion, "usage", None)
        if usage:
            run_usage.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            run_usage.output_tokens = getattr(usage, "completion_tokens", 0) or 0
            run_usage.requests = 1

            # Capture provider-specific details (reasoning tokens, cached tokens, etc.)
            for details_attr in ("completion_tokens_details", "prompt_tokens_details"):
                details_obj = getattr(usage, details_attr, None)
                if details_obj is None:
                    continue
                # details_obj can be a dict or a Pydantic model
                items = (
                    details_obj.items()
                    if isinstance(details_obj, dict)
                    else ((k, v) for k, v in vars(details_obj).items() if isinstance(v, int) and v > 0)
                )
                for k, v in items:
                    if isinstance(v, int) and v > 0:
                        run_usage.details[f"{details_attr}.{k}"] = v

        return run_usage

    async def _arun(
        self,
        params: InSchema,
        run_context: RunContext,
        **kwargs,
    ) -> OutSchema:
        """Run agent with optional tool calling support.

        If tools are configured, uses the tool loop (ReAct pattern).
        Otherwise, uses direct LLM call with structured output.

        Args:
            params: Input parameters matching input_schema.
            run_context: Optional RunContext with messages, human_response, etc.

        Returns:
            Output matching output_schema.
        """
        async with self.memory.asession(
            stateless=self.stateless,
            run_context=run_context,
            enable_trimming=self.enable_trimming,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            trim_ratio=self.trim_ratio,
        ) as messages:
            # Add system message if empty
            if not messages:
                messages.append(self._default_system_message())

            # Add user message (skip if resuming with human response)
            if params and not (run_context and run_context.human_response):
                messages.append(
                    {
                        "role": "user",
                        "content": params.model_dump_json(exclude={"type"}),
                    },
                )

            output: OutSchema | None = None

            if self.tools:
                # === TOOL CALLING MODE ===
                # Use _run_tool_loop and consume events to get final output
                # Note: Tool loop mutates `messages` in place during iterations
                async for event in self._run_tool_loop(
                    messages,
                    run_context,
                ):
                    if event.event_type == StreamEventType.COMPLETED:
                        output = event.output
                        break

                if output is None:
                    raise UnexpectedModelBehavior("Tool loop completed without producing output")

            else:
                # === DIRECT MODE (no tools) ===
                output = await self.get_response_async(
                    messages=messages,
                    response_model=self.output_schema,
                    run_context=run_context,
                )

            # Add final assistant message
            messages.append(
                {
                    "role": "assistant",
                    "content": output.model_dump_json(exclude={"type"}),
                },
            )

            return output

    async def _stream_llm_response(
        self,
        messages: list[dict[str, str]],
        run_context: RunContext,
        response_model: type[OutputSchema] | None = None,
        token_batch_size: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """Stream LLM response with thinking tokens and validated partial output.

        Uses raw LiteLLM streaming to access thinking tokens (reasoning_content).

        Args:
            messages: Chat messages
            run_context: Execution context for event correlation
            response_model: Output schema (defaults to self.output_schema)
            token_batch_size: Batch N tokens before emitting STREAMING event (default=1)

        Yields:
            - StreamingTokenEvent for batched raw tokens
            - ThinkingEvent for reasoning tokens
            - PartialOutputEvent for validated partials
            - CompletedEvent for final output
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
            "stream_options": {"include_usage": True},
        }

        if self.reasoning_effort:
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary:
            completion_kwargs["reasoning"] = {"summary": self.reasoning_summary}

        class_name = self.__class__.__name__

        # Handle human response for non-tool agents (inject as user message)
        if run_context.human_response:
            content = run_context.human_response.content
            messages.append(
                {
                    "role": "user",
                    "content": content if isinstance(content, str) else json.dumps(content),
                },
            )
            yield HumanResponseEvent(
                source=class_name,
                message="Resumed with human input",
                data=HumanResponseEventData(
                    tool_call_id=run_context.human_response.tool_call_id,
                    response=content,
                ),
                run_context=run_context,
            )

        accumulated = ""
        token_buffer = ""
        thinking_buffer = ""
        last_partial_dict = None

        response = await acompletion(**completion_kwargs)
        async for chunk in response:
            run_context.usage += self._extract_usage(chunk)

            # Check if chunk.choices is not empty
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Thinking tokens (Claude extended thinking, o1 reasoning)
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta,
                "thinking",
                None,
            )
            # Note: Not recommended to track thinking tokens in history
            if reasoning:
                thinking_buffer += reasoning
                if len(thinking_buffer) >= token_batch_size:
                    yield ThinkingEvent(
                        source=class_name,
                        message="Reasoning...",
                        data=ThinkingEventData(thinking_content=thinking_buffer),
                        run_context=run_context,
                    )
                    thinking_buffer = ""

            # Content tokens - batch, then accumulate and validate as partial
            if delta.content:
                token_buffer += delta.content
                accumulated += delta.content

                # Emit batched tokens
                if len(token_buffer) >= token_batch_size:
                    yield StreamingTokenEvent(
                        source=class_name,
                        data=StreamingEventData(token=token_buffer),
                        run_context=run_context,
                    )
                    token_buffer = ""

                # Try to validate as partial
                parsed = self._try_parse_json(accumulated)
                if parsed and parsed != last_partial_dict:
                    last_partial_dict = parsed
                    try:
                        partial = PartialResponseModel.model_validate(parsed)
                        yield PartialOutputEvent(
                            source=class_name,
                            message="Partial...",
                            data=PartialEventData(partial_output=partial),
                            run_context=run_context,
                        )
                    except Exception:
                        pass  # Skip invalid partials

        # Emit remaining thinking tokens
        if thinking_buffer:
            yield ThinkingEvent(
                source=class_name,
                message="Reasoning...",
                data=ThinkingEventData(thinking_content=thinking_buffer),
                run_context=run_context,
            )

        # Emit remaining tokens
        if token_buffer:
            yield StreamingTokenEvent(
                source=class_name,
                data=StreamingEventData(token=token_buffer),
                run_context=run_context,
            )

        # Final validation
        output = response_model.model_validate_json(accumulated)
        yield CompletedEvent(
            source=class_name,
            message=f"Completed {class_name}",
            data=CompletedEventData(output=output),
            run_context=run_context,
        )

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        run_context: RunContext,
        token_batch_size: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """ReAct loop: call LLM with tools until final answer.

        Streams thinking/reasoning tokens in real-time.
        Tool calls are IO events (TOOL_CALLING, TOOL_RESULT).
        Human tool calls yield HUMAN_INPUT_REQUIRED and return.

        Args:
            messages: Conversation history (mutated in place)
            run_context: RunContext (may contain human_response for resumption)
            token_batch_size: Batch N characters before emitting STREAMING event

        Yields:
            StreamEvent: STREAMING, THINKING, TOOL_CALLING, TOOL_RESULT,
                        HUMAN_INPUT_REQUIRED, or COMPLETED
        """
        # Attach messages reference to run_context (caller can access live state via any event)
        class_name = self.__class__.__name__
        run_context.messages = messages

        # Check for human response continuation (from previous HUMAN_INPUT_REQUIRED)
        human_response = run_context.human_response
        if human_response:
            tool_call_id = human_response.tool_call_id
            content = human_response.content

            # Inject human's response as tool result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(content),
                },
            )

            # Emit event so caller knows we resumed with human input
            yield HumanResponseEvent(
                source=class_name,
                message="Resumed with human input",
                data=HumanResponseEventData(tool_call_id=tool_call_id, response=content),
                run_context=run_context,
            )

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
                "stream_options": {"include_usage": True},
            }
            if self.reasoning_effort:
                completion_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.reasoning_summary:
                completion_kwargs["reasoning"] = {"summary": self.reasoning_summary}

            response = await acompletion(**completion_kwargs)

            # Accumulate streamed response
            accumulated_content = ""
            token_buffer = ""  # Batch tokens before emitting
            last_partial_dict: dict[str, Any] | None = None  # For deduplicating PARTIAL events
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
            buffered_partials: list[StreamEvent] = []  # Buffer partials until we know it's final
            thinking_buffer = ""

            async for chunk in response:
                run_context.usage += self._extract_usage(chunk)
                delta = chunk.choices[0].delta

                # Stream reasoning/thinking tokens (o1, o3, Claude extended thinking)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta,
                    "thinking",
                    None,
                )
                if reasoning:
                    thinking_buffer += reasoning
                    if len(thinking_buffer) >= token_batch_size:
                        yield ThinkingEvent(
                            source=class_name,
                            message="Reasoning...",
                            data=ThinkingEventData(
                                thinking_content=thinking_buffer,
                                streaming=True,
                            ),
                            run_context=run_context,
                        )
                        thinking_buffer = ""

                # Batch and stream content tokens
                if delta.content:
                    accumulated_content += delta.content
                    token_buffer += delta.content

                    # Emit batched tokens when buffer is full
                    if len(token_buffer) >= token_batch_size:
                        yield StreamingTokenEvent(
                            source=class_name,
                            message=token_buffer,
                            data=StreamingEventData(token=token_buffer),
                            run_context=run_context,
                        )
                        token_buffer = ""

                    # Try to validate as partial (buffer until we know it's final)
                    parsed = self._try_parse_json(accumulated_content)
                    if parsed and parsed != last_partial_dict:
                        last_partial_dict = parsed
                        try:
                            partial = PartialResponseModel.model_validate(parsed)
                            buffered_partials.append(
                                PartialOutputEvent(
                                    source=class_name,
                                    message="Partial output",
                                    data=PartialEventData(partial_output=partial),
                                    run_context=run_context,
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

            # Emit remaining thinking tokens in buffer
            if thinking_buffer:
                yield ThinkingEvent(
                    source=class_name,
                    message="Reasoning...",
                    data=ThinkingEventData(
                        thinking_content=thinking_buffer,
                        streaming=True,
                    ),
                    run_context=run_context,
                )

            # Emit remaining tokens in buffer
            if token_buffer:
                yield StreamingTokenEvent(
                    source=class_name,
                    message=token_buffer,
                    data=StreamingEventData(token=token_buffer),
                    run_context=run_context,
                )

            # After streaming: check for tool calls
            if not accumulated_tool_calls:
                for partial_event in buffered_partials:
                    yield partial_event

                if not accumulated_content:
                    raise UnexpectedModelBehavior("LLM returned empty response without tool calls")

                # LLM responded without tool calls. Try JSON parse first — the model
                # may have returned valid structured output without calling final_answer.
                try:
                    output = self.output_schema.model_validate_json(accumulated_content)
                    logger.warning(
                        "Model gave direct response instead of calling final_answer tool. Parsed as JSON fallback.",
                    )
                    yield CompletedEvent(
                        source=class_name,
                        message=f"Completed {class_name}",
                        data=CompletedEventData[self.output_schema](output=output),
                        run_context=run_context,
                    )
                    return
                except Exception:
                    pass

                # Not valid JSON — treat as internal reasoning and retry.
                # Follows the Pydantic AI pattern: append the text as assistant message
                # and a validation feedback nudge as user message.
                # See: https://github.com/pydantic/pydantic-ai/issues/1993
                logger.warning("LLM responded with text instead of calling a tool. Retrying.")
                yield ThinkingEvent(
                    source=class_name,
                    message="Internal reasoning...",
                    data=ThinkingEventData(thinking_content=accumulated_content),
                    run_context=run_context,
                )
                messages.append({"role": "assistant", "content": accumulated_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Validation feedback:\nPlease call one of the provided tool functions instead.",
                    },
                )
                continue

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
                yield ToolCallingEvent(
                    source=class_name,
                    message=f"Calling {tool_call.tool_name}",
                    data=ToolCallingEventData(tool_call=tool_call),
                    run_context=run_context,
                )

            # Check for human tool BEFORE execution - intercept and yield HUMAN_INPUT_REQUIRED
            for tool_call in tool_calls:
                if isinstance(self._find_tool(tool_call.tool_name), HumanTool):
                    try:
                        human_input = HumanToolInput(**tool_call.arguments)
                    except Exception:
                        human_input = HumanToolInput(
                            question=str(tool_call.arguments.get("question", "Input needed")),
                        )

                    # Add assistant message with tool_calls to memory
                    messages.append(
                        {
                            "role": "assistant",
                            "content": accumulated_content or None,
                            "tool_calls": tool_calls_for_message,
                        },
                    )

                    # Store messages in run_context for resumption
                    run_context.messages = list(messages)

                    # Yield HUMAN_INPUT_REQUIRED with full state for resumption
                    yield HumanInputRequiredEvent(
                        source=class_name,
                        message=f"Human input required: {human_input.question}",
                        data=HumanInputRequiredEventData(
                            human_input=human_input,
                            tool_call_id=tool_call.tool_call_id,
                            tool_name=tool_call.tool_name,
                        ),
                        run_context=run_context,
                    )
                    # End generator gracefully - caller will resume with fresh astream() call
                    return

            # Check max_tool_calls limit before executing
            if self.max_tool_calls is not None:
                if total_tool_calls + len(tool_calls) > self.max_tool_calls:
                    raise MaxToolCallsExceeded(
                        f"Exceeded {self.max_tool_calls} total tool calls "
                        f"(current: {total_tool_calls}, requested: {len(tool_calls)})",
                    )

            # Execute all tools in parallel (including OutputTool if called)
            # Pass all_tool_instances directly to avoid mutating self.tools (race condition)
            results = await self._execute_tools_parallel(tool_calls, tools=all_tool_instances)

            total_tool_calls += len(tool_calls)

            # Check if final_answer was called - signals completion
            for result in results:
                if result.tool_name == "final_answer":
                    if result.error:
                        # Validation failed — fall through to regular tool result handling
                        # so the error gets fed back to the LLM as a tool message,
                        # giving it a chance to retry with correct schema fields.
                        logger.warning(
                            f"final_answer validation failed, feeding error back to LLM for retry: {result.error}",
                        )
                        break
                    # Validation done in _execute_tool via input_schema(**arguments)
                    # result.content is dict from model_dump(), reconstruct the model
                    output = self.output_schema.model_validate(result.content)
                    yield CompletedEvent(
                        source=class_name,
                        message=f"Completed {class_name}",
                        data=CompletedEventData[self.output_schema](output=output),
                        run_context=run_context,
                    )
                    return

            # Yield TOOL_RESULT events for regular tools
            for result in results:
                yield ToolResultEvent(
                    source=class_name,
                    message=f"Result from {result.tool_name}",
                    data=ToolResultEventData(result=result),
                    run_context=run_context,
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
                yield ThinkingEvent(
                    source=class_name,
                    message="Reflecting on results...",
                    data=ThinkingEventData(reflection_prompt=self.reflection_prompt),
                    run_context=run_context,
                )

        raise MaxToolIterationsExceeded(f"Exceeded {self.max_tool_iterations} tool iterations")

    async def _astream(
        self,
        params: InSchema,
        run_context: RunContext,
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
            run_context: RunContext with messages, human_response, run_id, etc.
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

        yield StartingEvent(
            source=class_name,
            message=f"Starting {class_name}",
            data=StartingEventData[self.input_schema](params=params),
            run_context=run_context,
        )

        try:
            async with self.memory.asession(
                stateless=self.stateless,
                run_context=run_context,
                enable_trimming=self.enable_trimming,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                trim_ratio=self.trim_ratio,
            ) as messages:
                # Add system message if empty
                if not messages:
                    messages.append(self._default_system_message())

                # Add user message (skip if resuming with human response)
                if params and not (run_context and run_context.human_response):
                    messages.append(
                        {
                            "role": "user",
                            "content": params.model_dump_json(exclude={"type"}),
                        },
                    )

                yield RunningEvent(
                    source=class_name,
                    message=f"Running {class_name}",
                    run_context=run_context,
                )

                # Route: tool calling mode vs streaming mode
                output = None
                streamer = (
                    self._run_tool_loop(messages, run_context, token_batch_size=token_batch_size)
                    if self.tools
                    else self._stream_llm_response(messages, run_context, token_batch_size=token_batch_size)
                )

                async for event in streamer:
                    if isinstance(event, CompletedEvent):
                        output = event.data.output
                    yield event
                    if isinstance(event, HumanInputRequiredEvent):
                        return

                if output is None:
                    raise UnexpectedModelBehavior("No output received from LLM")

                messages.append(
                    {
                        "role": "assistant",
                        "content": output.model_dump_json(exclude={"type"}),
                    },
                )

        except Exception as e:
            yield FailedEvent(
                source=class_name,
                message=f"Failed: {e!s}",
                data=FailedEventData(error=str(e), error_type=type(e).__name__),
                run_context=run_context,
            )
            raise


Agent = LiteLLMInstructorBaseAgent
