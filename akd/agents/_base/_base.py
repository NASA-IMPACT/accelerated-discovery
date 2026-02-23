from __future__ import annotations

import copy
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, cast, get_args, get_origin

import instructor
import openai
from litellm import acompletion
from litellm.utils import get_model_info, supports_reasoning
from loguru import logger
from pydantic import AnyUrl, BaseModel, Field, create_model, model_validator

from akd._base import (
    AbstractBase,
    AgentSession,
    BaseConfig,
    InputSchema,
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
    HumanInputRequired,
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

from .providers import LiteLLMAdapter
from .providers._base import ProviderAdapter
from .providers.contracts import ProviderEventType, ProviderRequest, ProviderResponse
from .utils import UnifiedOutput, output_tool_name_for_schema


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
    output_mode: Literal["multi_tool", "unified_schema"] = Field(
        default="multi_tool",
        description="Output mode. multi_tool uses one final tool per schema branch.",
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
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

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

    def _build_run_context(self, run_context: RunContext | None) -> RunContext:
        """Create a run context copy and ensure it has a run_id."""
        ctx = (run_context or RunContext()).model_copy()
        ctx.run_id = ctx.run_id or uuid.uuid4().hex[:8]
        return ctx

    def _open_session(
        self,
        run_context: RunContext,
        mode: Literal["run", "stream", "chat"],
    ) -> AgentSession:
        """Create an agent session for this run context."""
        return AgentSession(
            run_context=run_context,
            stateless=self.stateless,
            enable_trimming=self.enable_trimming,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            trim_ratio=self.trim_ratio,
        )

    def _get_provider_adapter(self) -> ProviderAdapter:
        """Return provider adapter used by the shared run engine.

        Provider-specific subclasses should override this when migrating to
        adapter-driven orchestration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_provider_adapter()",
        )

    def _build_provider_request(
        self,
        *,
        token_batch_size: int = 10,
        output_schema: type[Any] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> ProviderRequest:
        """Build provider request payload for adapter calls."""
        return ProviderRequest(
            model_name=self.model_name,
            temperature=self.temperature,
            tools=self.tool_definitions,
            token_batch_size=token_batch_size,
            output_schema=output_schema or self._get_effective_output_schema(),
            provider_kwargs=provider_kwargs or {},
        )

    def _resolved_output_schemas(self) -> list[type[OutputSchema]]:
        """Return normalized output schema candidates for this agent."""
        declared = self.output_schema
        origin = get_origin(declared)
        if origin is None:
            schemas = [declared]
        else:
            schemas = [arg for arg in get_args(declared) if isinstance(arg, type) and issubclass(arg, OutputSchema)]
        unique: list[type[OutputSchema]] = []
        for schema in schemas:
            if schema not in unique:
                unique.append(schema)
        return unique

    def _unwrap_unified_output(self, output: Any) -> OutputSchema | None:
        """Unwrap single-envelope union output to a concrete schema branch."""
        if self.output_mode != "unified_schema":
            return None
        schemas = self._resolved_output_schemas()
        if len(schemas) <= 1:
            return None
        envelope_model = self._get_effective_output_schema()
        if not isinstance(output, envelope_model):
            return None
        kind = getattr(output, "kind", None)
        selected_schema: type[OutputSchema] | None = None

        if isinstance(kind, str):
            selected_schema = next((schema for schema in schemas if schema.__name__ == kind), None)
        else:
            populated = [schema for schema in schemas if getattr(output, schema.__name__, None) is not None]
            if len(populated) == 1:
                selected_schema = populated[0]

        if selected_schema is None:
            return None

        branch = getattr(output, selected_schema.__name__, None)
        if isinstance(branch, OutputSchema):
            return branch
        if isinstance(branch, dict):
            return selected_schema.model_validate(branch)
        return None

    def _route_output_from_event(self, event: StreamEvent) -> OutputSchema | None:
        """Extract concrete output from a stream event if it represents final output.

        Handles both modes transparently:
        - multi_tool: resolves output from ToolResultEvent via output tool match
        - unified_schema: unwraps envelope from CompletedEvent
        - single schema: passes through CompletedEvent output directly
        """
        if isinstance(event, ToolResultEvent):
            return self._resolve_output_from_tool_result(event.data.result)
        if isinstance(event, CompletedEvent):
            raw = event.data.output
            unwrapped = self._unwrap_unified_output(raw)
            return unwrapped if unwrapped is not None else raw
        return None

    def _get_effective_output_schema(self) -> type[OutputSchema]:
        """Return schema used for provider structured-output requests."""
        schemas = self._resolved_output_schemas()
        if not schemas:
            raise TypeError("output_schema must declare at least one OutputSchema type")
        if len(schemas) <= 1:
            return schemas[0]
        if self.output_mode == "unified_schema":
            return UnifiedOutput(*schemas)
        return schemas[0]

    def _build_output_tools(self) -> list[OutputTool]:
        """Build output tools according to union output mode."""
        schemas = self._resolved_output_schemas()
        if len(schemas) <= 1:
            return [OutputTool(schemas[0])]
        if self.output_mode != "multi_tool":
            return [OutputTool(schemas[0])]
        return [OutputTool(schema, name=output_tool_name_for_schema(schema)) for schema in schemas]

    def _resolve_output_from_tool_result(self, result: Any) -> OutputSchema | None:
        """Resolve completed output from a tool-result payload if it is an output tool."""
        tool_name = getattr(result, "tool_name", None)
        if not isinstance(tool_name, str) or getattr(result, "error", None):
            return None
        schemas = self._resolved_output_schemas()
        schema: type[OutputSchema] | None = (
            schemas[0]
            if len(schemas) == 1 and tool_name == "final_answer"
            else next((s for s in schemas if output_tool_name_for_schema(s) == tool_name), None)
        )
        if schema is None:
            return None
        content = getattr(result, "content", None)
        if isinstance(content, schema):
            return content
        if isinstance(content, dict):
            try:
                return schema.model_validate(content)
            except Exception:
                return None
        return None

    def _validate_output(self, output: Any) -> OutSchema:
        """Validate output against single or union output schemas."""
        unwrapped = self._unwrap_unified_output(output)
        candidate = unwrapped if unwrapped is not None else output
        for schema in self._resolved_output_schemas():
            if isinstance(candidate, schema):
                return cast(OutSchema, candidate)
        raise TypeError(
            "Output must be an instance of one of: "
            + ", ".join(schema.__name__ for schema in self._resolved_output_schemas()),
        )

    def _append_user_turn(
        self,
        run_context: RunContext,
        params: InSchema,
        mode: Literal["run", "stream", "chat"],
    ) -> None:
        """Append user turn unless this is a human-response resume."""
        if run_context.human_response or params is None:
            return
        if mode == "chat" and isinstance(params, TextInput):
            payload = params.content
        else:
            payload = params.model_dump_json(exclude={"type"})
        run_context.messages.append({"role": "user", "content": payload})

    def _finalize_success(self, run_context: RunContext, output: OutputSchema) -> None:
        """Append final assistant turn and persist run state."""
        run_context.messages.append(
            {
                "role": "assistant",
                "content": output.model_dump_json(exclude={"type"}),
            },
        )

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
        params = self._validate_input(params)
        run_context = self._build_run_context(run_context)

        async with self._open_session(run_context, "run") as session:
            if not session.messages:
                session.append(self._default_system_message())
            self._append_user_turn(run_context, params, "run")

            output = await self._arun(params, run_context=run_context, **kwargs)
            output = self._validate_output(output)
            self._finalize_success(run_context, output)
            output._run_context = run_context
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
        params = self._validate_input(params)
        run_context = self._build_run_context(run_context)

        async with self._open_session(run_context, "stream") as session:
            if not session.messages:
                session.append(self._default_system_message())
            self._append_user_turn(run_context, params, "stream")

            async for event in self._astream(params, run_context=run_context, **kwargs):
                if event.event_type == StreamEventType.COMPLETED:
                    output = self._validate_output(event.output)
                    self._finalize_success(run_context, output)
                    yield CompletedEvent(
                        source=event.source,
                        message=event.message,
                        data=CompletedEventData(output=output),
                        run_context=run_context,
                    )
                    continue
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
        params = params if isinstance(params, TextInput) else TextInput(content=str(params))

        run_context = self._build_run_context(run_context)

        async with self._open_session(run_context, "chat") as session:
            if not session.messages:
                session.append(self._default_system_message())
            self._append_user_turn(run_context, params, "chat")
            output = await self._arun(params, run_context=run_context, **kwargs)
            output = self._validate_output(output)
            self._finalize_success(run_context, output)

        chat_output = (
            TextOutput(content=output._response or str(output)) if not isinstance(output, TextOutput) else output
        )
        chat_output._run_context = run_context
        return chat_output

    @abstractmethod
    async def get_response_async(
        self,
        *,
        run_context: RunContext,
        response_model: type[OutputSchema] | None = None,
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

    async def _run_engine_stream(
        self,
        run_context: RunContext,
        token_batch_size: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """Unified provider-driven stream engine for tools and non-tools."""
        class_name = self.__class__.__name__
        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before _run_engine_stream")
        response_model = self._get_effective_output_schema()
        partial_response_model = PartialModel[response_model]

        request = self._build_provider_request(
            token_batch_size=token_batch_size,
            output_schema=response_model,
        )
        adapter = self._get_provider_adapter()

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
        last_partial_dict: dict[str, Any] | None = None

        async for provider_event in adapter.request_stream(
            run_context=run_context,
            request=request,
        ):
            if provider_event.kind == ProviderEventType.USAGE:
                usage = provider_event.data.get("usage")
                if isinstance(usage, RunUsage):
                    run_context.usage += usage
                continue

            if provider_event.kind == ProviderEventType.REASONING_DELTA:
                reasoning = provider_event.data.get("text")
                if isinstance(reasoning, str) and reasoning:
                    thinking_buffer += reasoning
                    if len(thinking_buffer) >= token_batch_size:
                        yield ThinkingEvent(
                            source=class_name,
                            message="Reasoning...",
                            data=ThinkingEventData(thinking_content=thinking_buffer),
                            run_context=run_context,
                        )
                        thinking_buffer = ""
                continue

            if provider_event.kind == ProviderEventType.TEXT_DELTA:
                content = provider_event.data.get("text")
                if isinstance(content, str) and content:
                    token_buffer += content
                    accumulated += content

                    if len(token_buffer) >= token_batch_size:
                        yield StreamingTokenEvent(
                            source=class_name,
                            data=StreamingEventData(token=token_buffer),
                            run_context=run_context,
                        )
                        token_buffer = ""

                    try:
                        parsed = json.loads(accumulated)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and parsed != last_partial_dict:
                        last_partial_dict = parsed
                        try:
                            partial = partial_response_model.model_validate(parsed)
                            yield PartialOutputEvent(
                                source=class_name,
                                message="Partial...",
                                data=PartialEventData(partial_output=partial),
                                run_context=run_context,
                            )
                        except Exception:
                            pass
                continue

            if provider_event.kind == ProviderEventType.FINAL_OUTPUT:
                final_content = provider_event.data.get("content")
                if isinstance(final_content, str):
                    accumulated = final_content
                continue

            if provider_event.kind == ProviderEventType.ERROR:
                error = provider_event.data.get("error")
                raise UnexpectedModelBehavior(
                    str(error) if error is not None else "Provider adapter emitted error event",
                )

        if thinking_buffer:
            yield ThinkingEvent(
                source=class_name,
                message="Reasoning...",
                data=ThinkingEventData(thinking_content=thinking_buffer),
                run_context=run_context,
            )

        if token_buffer:
            yield StreamingTokenEvent(
                source=class_name,
                data=StreamingEventData(token=token_buffer),
                run_context=run_context,
            )

        output = response_model.model_validate_json(accumulated)
        yield CompletedEvent(
            source=class_name,
            message=f"Completed {class_name}",
            data=CompletedEventData(output=output),
            run_context=run_context,
        )

    async def _astream(
        self,
        params: InSchema,
        run_context: RunContext,
        token_batch_size: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Default internal streaming template using _run_engine_stream().

        Mode-agnostic event observer: streams from the engine and uses
        _route_output_from_event() to detect final output regardless of
        whether the engine uses multi_tool or unified_schema mode.
        """
        class_name = self.__class__.__name__

        yield StartingEvent(
            source=class_name,
            message=f"Starting {class_name}",
            data=StartingEventData[self.input_schema](params=params),
            run_context=run_context,
        )

        try:
            yield RunningEvent(
                source=class_name,
                message=f"Running {class_name}",
                run_context=run_context,
            )

            output: OutputSchema | None = None
            async for event in self._run_engine_stream(
                run_context=run_context,
                token_batch_size=token_batch_size,
            ):
                resolved = self._route_output_from_event(event)
                if resolved is not None:
                    output = resolved
                    yield CompletedEvent(
                        source=class_name,
                        message=f"Completed {class_name}",
                        data=CompletedEventData(output=resolved),
                        run_context=run_context,
                    )
                    return
                if isinstance(event, HumanInputRequiredEvent):
                    yield event
                    return
                yield event

            if output is None:
                raise UnexpectedModelBehavior("No output received from LLM")

        except Exception as e:
            yield FailedEvent(
                source=class_name,
                message=f"Failed: {e!s}",
                data=FailedEventData(error=str(e), error_type=type(e).__name__),
                run_context=run_context,
            )
            raise


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
        import warnings

        if not isinstance(self, LiteLLMInstructorBaseAgent):
            warnings.warn(
                "InstructorBaseAgent is deprecated. Use LiteLLMInstructorBaseAgent instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(config=config, debug=debug)

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
            **cast(dict[str, Any], fields),
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
        *,
        run_context: RunContext,
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before get_response_async")

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
        run_context: RunContext,
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
        # NOTE: Memory session and messages prep handled by BaseAgent.arun
        # We rely on run_context.messages being set.
        if run_context.messages is None:
            raise ValueError("run_context.messages must be set (did you call via BaseAgent.arun?)")

        # NOTE: Not appending response to messages here anymore;
        # BaseAgent.arun handles that now.

        return await self.get_response_async(
            run_context=run_context,
            response_model=self.output_schema,
        )


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

    def _build_provider_request(
        self,
        *,
        token_batch_size: int = 10,
        output_schema: type[Any] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> ProviderRequest:
        """Build provider request with LiteLLM-specific kwargs."""
        base_kwargs: dict[str, Any] = {
            "api_base": str(self.base_url).rstrip("/") if self.base_url else None,
            "api_key": self.api_key,
            "num_retries": self.num_retries,
            "drop_params": True,
        }
        if self.reasoning_effort:
            base_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary:
            base_kwargs["reasoning_summary"] = self.reasoning_summary
        if provider_kwargs:
            base_kwargs.update(provider_kwargs)

        is_stream = base_kwargs.get("stream") is True
        has_tools = bool(base_kwargs.get("tools"))
        if is_stream:
            base_kwargs.setdefault("stream_options", {"include_usage": True})
            if output_schema is not None and not has_tools:
                base_kwargs.setdefault(
                    "response_format",
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_schema.__name__,
                            "strict": True,
                            "schema": self._build_response_format_schema(cast(type[OutputSchema], output_schema)),
                        },
                    },
                )
            if self.reasoning_summary:
                base_kwargs.setdefault("reasoning", {"summary": self.reasoning_summary})

        return super()._build_provider_request(
            token_batch_size=token_batch_size,
            output_schema=output_schema,
            provider_kwargs=base_kwargs,
        )

    def _get_provider_adapter(self) -> ProviderAdapter:
        """Return the LiteLLM provider adapter for this agent."""
        return LiteLLMAdapter(client=self.client, completion_callable=acompletion)

    async def get_response_async(
        self,
        *,
        run_context: RunContext,
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously with automatic message trimming.

        Args:
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        response_model = response_model or self.output_schema
        request = self._build_provider_request(
            output_schema=response_model,
            provider_kwargs={"stream": False},
        )
        adapter = self._get_provider_adapter()
        provider_response = await adapter.request_once(
            run_context=run_context,
            request=request,
        )
        run_context.usage += provider_response.usage

        response = self._parse_provider_response(provider_response, response_model)
        return cast(OutSchema, response)

    def _parse_provider_response(
        self,
        provider_response: ProviderResponse,
        response_model: type[OutputSchema],
    ) -> OutputSchema:
        """Parse a normalized provider response into output schema."""
        content = provider_response.content
        if content is None:
            raise UnexpectedModelBehavior("Provider returned no content for structured output")
        try:
            return response_model.model_validate_json(content)
        except Exception as exc:
            raise UnexpectedModelBehavior(f"Failed to parse provider response content: {exc}") from exc

    async def _arun(
        self,
        params: InSchema,
        run_context: RunContext,
        **kwargs,
    ) -> OutSchema:
        """Run agent with optional tool calling support.

        Direct mode (no tools): one-shot instructor extraction via get_response_async.
        Tool mode: consumes _run_engine_stream via _route_output_from_event.

        Args:
            params: Input parameters matching input_schema.
            run_context: Optional RunContext with messages, human_response, etc.

        Returns:
            Output matching output_schema.
        """
        token_batch_size = kwargs.pop("token_batch_size", 10)
        has_union = len(self._resolved_output_schemas()) > 1
        use_tool_loop = bool(self.tools) or (has_union and self.output_mode == "multi_tool")

        if not use_tool_loop:
            return await self.get_response_async(
                run_context=run_context,
                response_model=self._get_effective_output_schema(),
            )

        async for event in self._run_engine_stream(
            run_context=run_context,
            token_batch_size=token_batch_size,
        ):
            resolved = self._route_output_from_event(event)
            if resolved is not None:
                return cast(OutSchema, resolved)
            if isinstance(event, HumanInputRequiredEvent):
                raise HumanInputRequired(str(event.data.human_input))

        raise UnexpectedModelBehavior("Tool loop completed without producing output")

    async def _run_engine_stream(
        self,
        run_context: RunContext,
        token_batch_size: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """LiteLLM-specific unified stream engine for tools and non-tools."""
        class_name = self.__class__.__name__
        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before _run_engine_stream")

        response_model = self._get_effective_output_schema()
        tool_defs: list[dict[str, Any]] = []
        tool_instances: list[BaseTool] = []
        partial_response_model = PartialModel[response_model]
        output_tools: list[OutputTool] = []
        has_union = len(self._resolved_output_schemas()) > 1
        use_tool_loop = bool(self.tools) or (has_union and self.output_mode == "multi_tool")
        if use_tool_loop:
            output_tools = self._build_output_tools()
            tool_instances = self.tools + output_tools
            tool_defs = self.tool_definitions + [tool.as_tool_definition() for tool in output_tools]
        output_tool_names = {tool.name for tool in output_tools}

        human_response = run_context.human_response
        if human_response:
            if tool_defs:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": human_response.tool_call_id,
                        "content": json.dumps(human_response.content),
                    },
                )
            else:
                content = human_response.content
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
                    tool_call_id=human_response.tool_call_id,
                    response=human_response.content,
                ),
                run_context=run_context,
            )

        total_tool_calls = 0
        max_turns = self.max_tool_iterations if tool_defs else max(1, self.max_tool_iterations)
        adapter = self._get_provider_adapter()
        for _ in range(max_turns):
            request = self._build_provider_request(
                token_batch_size=token_batch_size,
                output_schema=response_model,
                provider_kwargs={
                    "stream": True,
                    "tools": tool_defs or None,
                    "stream_options": {"include_usage": True},
                    "reasoning": {"summary": self.reasoning_summary} if self.reasoning_summary else None,
                },
            )

            accumulated_content = ""
            token_buffer = ""
            last_partial_dict: dict[str, Any] | None = None
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}
            buffered_partials: list[StreamEvent] = []
            thinking_buffer = ""

            async for provider_event in adapter.request_stream(
                run_context=run_context,
                request=request,
            ):
                if provider_event.kind == ProviderEventType.USAGE:
                    usage = provider_event.data.get("usage")
                    if isinstance(usage, RunUsage):
                        run_context.usage += usage
                    continue

                if provider_event.kind == ProviderEventType.REASONING_DELTA:
                    reasoning = provider_event.data.get("text")
                    if isinstance(reasoning, str) and reasoning:
                        thinking_buffer += reasoning
                        if len(thinking_buffer) >= token_batch_size:
                            yield ThinkingEvent(
                                source=class_name,
                                message="Reasoning...",
                                data=ThinkingEventData(thinking_content=thinking_buffer, streaming=True),
                                run_context=run_context,
                            )
                            thinking_buffer = ""
                    continue

                if provider_event.kind == ProviderEventType.TEXT_DELTA:
                    content = provider_event.data.get("text")
                    if isinstance(content, str) and content:
                        accumulated_content += content
                        token_buffer += content
                        if len(token_buffer) >= token_batch_size:
                            yield StreamingTokenEvent(
                                source=class_name,
                                message=token_buffer,
                                data=StreamingEventData(token=token_buffer),
                                run_context=run_context,
                            )
                            token_buffer = ""

                        parsed = self._try_parse_json(accumulated_content)
                        if parsed and parsed != last_partial_dict:
                            last_partial_dict = parsed
                            try:
                                partial = partial_response_model.model_validate(parsed)
                                buffered_partials.append(
                                    PartialOutputEvent(
                                        source=class_name,
                                        message="Partial output",
                                        data=PartialEventData(partial_output=partial),
                                        run_context=run_context,
                                    ),
                                )
                            except Exception:
                                pass
                    continue

                if provider_event.kind == ProviderEventType.TOOL_CALL:
                    if not tool_defs:
                        raise UnexpectedModelBehavior(
                            "Provider emitted TOOL_CALL event but no tools are configured",
                        )
                    idx = int(provider_event.data.get("index", 0))
                    tc_id = provider_event.data.get("id")
                    tc_name = provider_event.data.get("name")
                    tc_args_delta = provider_event.data.get("arguments_delta", "")
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_id or "",
                            "name": tc_name or "",
                            "arguments": tc_args_delta if isinstance(tc_args_delta, str) else "",
                        }
                    else:
                        if tc_id:
                            accumulated_tool_calls[idx]["id"] = tc_id
                        if tc_name:
                            accumulated_tool_calls[idx]["name"] = tc_name
                        if isinstance(tc_args_delta, str) and tc_args_delta:
                            accumulated_tool_calls[idx]["arguments"] += tc_args_delta
                    continue

                if provider_event.kind == ProviderEventType.FINAL_OUTPUT:
                    final_content = provider_event.data.get("content")
                    if isinstance(final_content, str):
                        accumulated_content = final_content
                    continue

                if provider_event.kind == ProviderEventType.ERROR:
                    error = provider_event.data.get("error")
                    raise UnexpectedModelBehavior(
                        str(error) if error is not None else "Provider adapter emitted error event",
                    )

            if thinking_buffer:
                yield ThinkingEvent(
                    source=class_name,
                    message="Reasoning...",
                    data=ThinkingEventData(thinking_content=thinking_buffer, streaming=True),
                    run_context=run_context,
                )

            if token_buffer:
                yield StreamingTokenEvent(
                    source=class_name,
                    message=token_buffer,
                    data=StreamingEventData(token=token_buffer),
                    run_context=run_context,
                )

            if not accumulated_tool_calls:
                for partial_event in buffered_partials:
                    yield partial_event
                if not accumulated_content:
                    raise UnexpectedModelBehavior("LLM returned empty response without tool calls")

                try:
                    output = response_model.model_validate_json(accumulated_content)
                    yield CompletedEvent(
                        source=class_name,
                        message=f"Completed {class_name}",
                        data=CompletedEventData[response_model](output=output),
                        run_context=run_context,
                    )
                    return
                except Exception as exc:
                    if any(s is TextOutput for s in self._resolved_output_schemas()) and accumulated_content.strip():
                        yield CompletedEvent(
                            source=class_name,
                            message=f"Completed {class_name}",
                            data=CompletedEventData[TextOutput](output=TextOutput(content=accumulated_content)),
                            run_context=run_context,
                        )
                        return
                    yield ThinkingEvent(
                        source=class_name,
                        message="Internal reasoning...",
                        data=ThinkingEventData(thinking_content=accumulated_content),
                        run_context=run_context,
                    )
                    messages.append({"role": "assistant", "content": accumulated_content})
                    if tool_defs:
                        messages.append(
                            {
                                "role": "user",
                                "content": "Validation feedback:\nPlease call one of the provided tool functions instead.",
                            },
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Validation feedback:\n"
                                    "Your previous JSON did not match the required output schema. "
                                    "Return ONLY valid JSON that satisfies the schema.\n"
                                    f"Validation error: {exc}"
                                ),
                            },
                        )
                    continue

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

            for tool_call in tool_calls:
                if isinstance(self._find_tool(tool_call.tool_name), HumanTool):
                    try:
                        human_input = HumanToolInput(**tool_call.arguments)
                    except Exception:
                        human_input = HumanToolInput(
                            question=str(tool_call.arguments.get("question", "Input needed")),
                        )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": accumulated_content or None,
                            "tool_calls": tool_calls_for_message,
                        },
                    )
                    run_context.messages = list(messages)
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
                    return

            if self.max_tool_calls is not None and total_tool_calls + len(tool_calls) > self.max_tool_calls:
                raise MaxToolCallsExceeded(
                    f"Exceeded {self.max_tool_calls} total tool calls "
                    f"(current: {total_tool_calls}, requested: {len(tool_calls)})",
                )

            results = await self._execute_tools_parallel(tool_calls, tools=tool_instances)
            total_tool_calls += len(tool_calls)

            for result in results:
                yield ToolResultEvent(
                    source=class_name,
                    message=f"Result from {result.tool_name}",
                    data=ToolResultEventData(result=result),
                    run_context=run_context,
                )

            if any(result.tool_name in output_tool_names and not result.error for result in results):
                return

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

        if tool_defs:
            raise MaxToolIterationsExceeded(f"Exceeded {self.max_tool_iterations} tool iterations")
        raise UnexpectedModelBehavior("Non-tool streaming ended without completion")


Agent = LiteLLMInstructorBaseAgent
