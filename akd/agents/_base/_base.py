from __future__ import annotations

import copy
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, cast, get_args, get_origin

import instructor
from litellm import acompletion
from litellm.utils import get_model_info, supports_reasoning
from loguru import logger
from pydantic import AnyUrl, Field, model_validator

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

from .providers import LiteLLMAdapter
from .providers._base import ProviderAdapter
from .providers.contracts import ProviderEventType, ProviderRequest
from .utils import UnifiedOutput


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
    """

    config_schema = BaseAgentConfig

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

    def _post_init(self):
        super()._post_init()
        self.output_tools: list[OutputTool] = self._build_output_tools()

    @property
    def _system_prompt(self) -> str:
        """Enhanced system prompt with agent description."""
        content = self.system_prompt
        if self.description:
            content += f"\n\nAGENT DESCRIPTION:\n{self.description}"
        return content

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
        mode: Literal["run", "stream"],
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
        """Return provider adapter. Provider subclasses must override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_provider_adapter()",
        )

    def _build_provider_request(
        self,
        *,
        output_schema: type[Any] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> ProviderRequest:
        """Build provider request payload for adapter calls."""
        return ProviderRequest(
            model_name=self.model_name,
            temperature=self.temperature,
            tools=[t.as_tool_definition() for t in self.tools],
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
        return [OutputTool(schema, name=f"final_{schema.__name__}") for schema in schemas]

    def _resolve_output_from_tool_result(self, result: Any) -> OutputSchema | None:
        """Resolve completed output from a tool-result payload if it is an output tool."""
        tool_name = getattr(result, "tool_name", None)
        if not isinstance(tool_name, str) or getattr(result, "error", None):
            return None
        schemas = self._resolved_output_schemas()
        schema: type[OutputSchema] | None = (
            schemas[0]
            if len(schemas) == 1 and tool_name == "final_answer"
            else next((s for s in schemas if f"final_{s.__name__}" == tool_name), None)
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
        mode: Literal["run", "stream"],
    ) -> None:
        """Append user turn unless this is a human-response resume."""
        if run_context.human_response or params is None:
            return
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
        """Run the agent with the provided parameters asynchronously."""
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
        """Stream agent execution with run_context initialization."""
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

    async def _run_engine_stream(
        self,
        run_context: RunContext,
    ) -> AsyncIterator[StreamEvent]:
        """Provider-specific stream engine. Must be implemented by provider subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _run_engine_stream()")
        # Make this an async generator
        yield  # pragma: no cover

    async def _astream(
        self,
        params: InSchema,
        run_context: RunContext,
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
            async for event in self._run_engine_stream(run_context=run_context):
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


class LiteLLMInstructorBaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](ToolCallingMixin, BaseAgent):
    """LiteLLM-based agent with tool calling and automatic message trimming.

    Uses LiteLLM for completion calls and instructor for structured output.
    Provides ReAct-style tool calling loop, HITL support, and streaming.
    """

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config, debug=debug)
        self.client = instructor.from_litellm(acompletion)

    def _post_init(self):
        super()._post_init()

    def __deepcopy__(self, memo: dict[int, Any]) -> LiteLLMInstructorBaseAgent:
        """Custom deepcopy that recreates the LiteLLM client instead of copying it."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        result.client = instructor.from_litellm(acompletion)
        return result

    def _build_response_format_schema(self, model: type[OutputSchema]) -> dict[str, Any]:
        """Build JSON schema for response_format (OpenAI strict mode)."""
        schema = model.model_json_schema()
        schema.pop("$defs", None)
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        return schema

    def _build_provider_request(
        self,
        *,
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
        model_name = self.model_name

        if is_stream:
            # Use OpenAI responses API route for gpt-5 streaming with reasoning.
            # Non-streaming (instructor) path uses the original model name since
            # instructor TOOLS mode is incompatible with the responses API route.
            if (
                self.model_name.startswith("gpt-5")
                and self.reasoning_effort
                and self.reasoning_summary
                and supports_reasoning(model=self.model_name)
            ):
                model_name = f"openai/responses/{self.model_name}"
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

        request = super()._build_provider_request(
            output_schema=output_schema,
            provider_kwargs=base_kwargs,
        )
        request.model_name = model_name
        return request

    def _get_provider_adapter(self) -> ProviderAdapter:
        """Return the LiteLLM provider adapter for this agent."""
        return LiteLLMAdapter(client=self.client, completion_callable=acompletion)

    async def get_response_async(
        self,
        *,
        run_context: RunContext,
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """Obtain a structured response via instructor (non-streaming)."""
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

        content = provider_response.content
        if content is None:
            raise UnexpectedModelBehavior("Provider returned no content for structured output")
        try:
            response = response_model.model_validate_json(content)
        except Exception as exc:
            raise UnexpectedModelBehavior(f"Failed to parse provider response content: {exc}") from exc
        return cast(OutSchema, response)

    async def _arun(
        self,
        params: InSchema,
        run_context: RunContext,
        **kwargs,
    ) -> OutSchema:
        """Run agent with optional tool calling support.

        Direct mode (no tools): one-shot instructor extraction via get_response_async.
        Tool mode: consumes _run_engine_stream via _route_output_from_event.
        """
        has_union = len(self._resolved_output_schemas()) > 1
        use_tool_loop = bool(self.tools) or (has_union and self.output_mode == "multi_tool")

        if not use_tool_loop:
            return await self.get_response_async(
                run_context=run_context,
                response_model=self._get_effective_output_schema(),
            )

        async for event in self._run_engine_stream(run_context=run_context):
            resolved = self._route_output_from_event(event)
            if resolved is not None:
                return cast(OutSchema, resolved)
            if isinstance(event, HumanInputRequiredEvent):
                raise HumanInputRequired(str(event.data.human_input))

        raise UnexpectedModelBehavior("Tool loop completed without producing output")

    async def _run_engine_stream(
        self,
        run_context: RunContext,
    ) -> AsyncIterator[StreamEvent]:
        """LiteLLM-specific stream engine with ReAct tool calling loop."""
        class_name = self.__class__.__name__
        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before _run_engine_stream")

        response_model = self._get_effective_output_schema()

        # Tool setup using pre-computed output_tools
        all_tools = self.tools + self.output_tools
        tool_defs = [t.as_tool_definition() for t in all_tools] if all_tools else []
        output_tool_names = {t.name for t in self.output_tools}

        # HITL resume
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
        max_turns = self.max_tool_iterations
        adapter = self._get_provider_adapter()

        for _ in range(max_turns):
            request = self._build_provider_request(
                output_schema=response_model,
                provider_kwargs={
                    "stream": True,
                    "tools": tool_defs or None,
                },
            )

            accumulated_content = ""
            tool_calls: list[ToolCall] = []

            # Consume adapter events — yield deltas immediately, no buffering
            async for provider_event in adapter.request_stream(
                run_context=run_context,
                request=request,
            ):
                if provider_event.kind == ProviderEventType.USAGE:
                    usage = provider_event.data.get("usage")
                    if isinstance(usage, RunUsage):
                        run_context.usage += usage

                elif provider_event.kind == ProviderEventType.REASONING_DELTA:
                    text = provider_event.data.get("text", "")
                    if text:
                        yield ThinkingEvent(
                            source=class_name,
                            message="Reasoning...",
                            data=ThinkingEventData(thinking_content=text, streaming=True),
                            run_context=run_context,
                        )

                elif provider_event.kind == ProviderEventType.TEXT_DELTA:
                    text = provider_event.data.get("text", "")
                    if text:
                        accumulated_content += text
                        yield StreamingTokenEvent(
                            source=class_name,
                            data=StreamingEventData(token=text),
                            run_context=run_context,
                        )

                elif provider_event.kind == ProviderEventType.TOOL_CALL:
                    # Tool calls are assembled by the adapter
                    tc_data = provider_event.data
                    tc = ToolCall(
                        tool_call_id=tc_data["id"],
                        tool_name=tc_data["name"],
                        arguments=json.loads(tc_data["arguments"]),
                    )
                    tool_calls.append(tc)
                    yield ToolCallingEvent(
                        source=class_name,
                        message=f"Calling {tc.tool_name}",
                        data=ToolCallingEventData(tool_call=tc),
                        run_context=run_context,
                    )

                elif provider_event.kind == ProviderEventType.ERROR:
                    error = provider_event.data.get("error")
                    raise UnexpectedModelBehavior(
                        str(error) if error is not None else "Provider adapter emitted error event",
                    )

            # No tool calls — try to parse output
            if not tool_calls:
                if not accumulated_content:
                    raise UnexpectedModelBehavior("LLM returned empty response without tool calls")

                try:
                    output = response_model.model_validate_json(accumulated_content)
                    yield CompletedEvent(
                        source=class_name,
                        message=f"Completed {class_name}",
                        data=CompletedEventData(output=output),
                        run_context=run_context,
                    )
                    return
                except Exception as exc:
                    # TextOutput fallback
                    if any(s is TextOutput for s in self._resolved_output_schemas()) and accumulated_content.strip():
                        yield CompletedEvent(
                            source=class_name,
                            message=f"Completed {class_name}",
                            data=CompletedEventData(output=TextOutput(content=accumulated_content)),
                            run_context=run_context,
                        )
                        return
                    # Validation retry — append feedback and continue loop
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

            # HITL detection
            tool_calls_for_message = [
                {
                    "id": tc.tool_call_id,
                    "type": "function",
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]
            for tc in tool_calls:
                if isinstance(self._find_tool(tc.tool_name, all_tools), HumanTool):
                    try:
                        human_input = HumanToolInput(**tc.arguments)
                    except Exception:
                        human_input = HumanToolInput(
                            question=str(tc.arguments.get("question", "Input needed")),
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
                            tool_call_id=tc.tool_call_id,
                            tool_name=tc.tool_name,
                        ),
                        run_context=run_context,
                    )
                    return

            # Max tool calls check
            if self.max_tool_calls is not None and total_tool_calls + len(tool_calls) > self.max_tool_calls:
                raise MaxToolCallsExceeded(
                    f"Exceeded {self.max_tool_calls} total tool calls "
                    f"(current: {total_tool_calls}, requested: {len(tool_calls)})",
                )

            # Execute tools
            results = await self._execute_tools_parallel(tool_calls, tools=all_tools)
            total_tool_calls += len(tool_calls)

            for result in results:
                yield ToolResultEvent(
                    source=class_name,
                    message=f"Result from {result.tool_name}",
                    data=ToolResultEventData(result=result),
                    run_context=run_context,
                )

            # Check if an output tool completed
            if any(result.tool_name in output_tool_names and not result.error for result in results):
                return

            # Append tool turn to messages
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


# Backward compatibility aliases
InstructorBaseAgent = LiteLLMInstructorBaseAgent
Agent = LiteLLMInstructorBaseAgent
