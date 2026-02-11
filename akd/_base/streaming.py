"""Streaming support for akd agents and tools."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .structures import RunContext
from .tool_calling import ToolCall, ToolResult


class StreamEventType(str, Enum):
    """Types of events emitted during streaming."""

    # Lifecycle
    STARTING = "starting"
    COMPLETED = "completed"
    FAILED = "failed"

    # Execution
    RUNNING = "running"
    STREAMING = "streaming"  # Raw tokens as they arrive
    THINKING = "thinking"  # Reasoning tokens (Claude extended thinking, o1)
    PARTIAL = "partial"  # Partial structured output as it streams

    # Tool calling (Stage 2)
    TOOL_CALLING = "tool_calling"
    TOOL_RESULT = "tool_result"

    # Human interaction (Stage 3)
    HUMAN_INPUT_REQUIRED = "human_input_required"
    HUMAN_RESPONSE = "human_response"  # Resumed with human input


# ---------------------------------------------------------------------------
# Typed Event Data Models
# ---------------------------------------------------------------------------


class StartingEventData[T](BaseModel):
    """Data for STARTING events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: T | None = None


class RunningEventData(BaseModel):
    """Data for RUNNING events."""

    model_config = ConfigDict(extra="allow")


class CompletedEventData[T](BaseModel):
    """Data for COMPLETED events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: T


class FailedEventData(BaseModel):
    """Data for FAILED events."""

    error: str
    error_type: str


class StreamingEventData(BaseModel):
    """Data for STREAMING events (raw token chunks)."""

    token: str


class ThinkingEventData(BaseModel):
    """Data for THINKING events (reasoning tokens)."""

    thinking_content: str | None = None
    reflection_prompt: str | None = None
    streaming: bool = False


class PartialEventData[T](BaseModel):
    """Data for PARTIAL events (partial structured output)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    partial_output: T


class ToolCallingEventData(BaseModel):
    """Data for TOOL_CALLING events."""

    tool_call: ToolCall


class ToolResultEventData(BaseModel):
    """Data for TOOL_RESULT events."""

    result: ToolResult


class HumanInputRequiredEventData(BaseModel):
    """Data for HUMAN_INPUT_REQUIRED events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    human_input: Any
    tool_call_id: str
    tool_name: str


class HumanResponseEventData(BaseModel):
    """Data for HUMAN_RESPONSE events."""

    tool_call_id: str
    response: Any


EventData = (
    StartingEventData
    | RunningEventData
    | CompletedEventData
    | FailedEventData
    | StreamingEventData
    | ThinkingEventData
    | PartialEventData
    | ToolCallingEventData
    | ToolResultEventData
    | HumanInputRequiredEventData
    | HumanResponseEventData
)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    """Streaming event with CloudEvents-inspired design.

    Supports both typed EventData models and legacy dict payloads.
    Use factory methods (``mk_*``) for type-safe construction.
    Convenience properties check typed data first, then fall back to dict.

    Core Fields:
        event_type: Type of event (STARTING, COMPLETED, FAILED, etc.)
        timestamp: UTC timestamp when event was created
        event_id: Unique identifier for this event
        source: Class name that generated this event
        message: Human-readable description
        data: Typed EventData model or legacy dict payload
        run_context: RunContext (extra="allow", accepts arbitrary keys)

    Example (isinstance checks - typed access):
        async for event in agent.astream(input_data):
            if isinstance(event, CompletedEvent):
                print(event.data.output)  # fully typed
            elif isinstance(event, StreamingTokenEvent):
                print(event.data.token)  # fully typed

    Example (convenience properties):
        async for event in agent.astream(input_data):
            if event.event_type == StreamEventType.COMPLETED:
                print(event.output)
    """

    model_config = ConfigDict(use_enum_values=True)

    # Core envelope
    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str | None = None
    message: str | None = None

    # Typed EventData or legacy dict
    data: EventData | dict[str, Any] = Field(default_factory=dict)

    # Execution context (RunContext has extra="allow" so arbitrary keys work)
    run_context: RunContext = Field(
        default_factory=RunContext,
        description="Execution context (run_id, human_response, messages, etc.)",
    )

    # -----------------------------------------------------------------------
    # Convenience properties (typed data first, dict fallback)
    # -----------------------------------------------------------------------

    @property
    def output(self) -> Any | None:
        """COMPLETED event output."""
        if isinstance(self.data, CompletedEventData):
            return self.data.output
        return self.data.get("output") if isinstance(self.data, dict) else None

    @property
    def error(self) -> str | None:
        """FAILED event error message."""
        if isinstance(self.data, FailedEventData):
            return self.data.error
        return self.data.get("error") if isinstance(self.data, dict) else None

    @property
    def tool_name(self) -> str | None:
        """TOOL_CALLING/TOOL_RESULT tool name."""
        if isinstance(self.data, ToolCallingEventData):
            return self.data.tool_call.tool_name
        if isinstance(self.data, ToolResultEventData):
            return self.data.result.tool_name
        return self.data.get("tool_name") if isinstance(self.data, dict) else None

    @property
    def tool_input(self) -> dict[str, Any] | None:
        """TOOL_CALLING input parameters."""
        if isinstance(self.data, ToolCallingEventData):
            return self.data.tool_call.arguments
        return self.data.get("tool_input") if isinstance(self.data, dict) else None

    @property
    def tool_output(self) -> Any | None:
        """TOOL_RESULT output value."""
        if isinstance(self.data, ToolResultEventData):
            return self.data.result.content
        return self.data.get("tool_output") if isinstance(self.data, dict) else None

    @property
    def partial_output(self) -> Any | None:
        """PARTIAL event partial output."""
        if isinstance(self.data, PartialEventData):
            return self.data.partial_output
        return self.data.get("partial_output") if isinstance(self.data, dict) else None

    @property
    def thinking_content(self) -> str | None:
        """THINKING event reasoning content."""
        if isinstance(self.data, ThinkingEventData):
            return self.data.thinking_content
        return self.data.get("thinking_content") if isinstance(self.data, dict) else None

    @property
    def token(self) -> str | None:
        """STREAMING event raw token."""
        if isinstance(self.data, StreamingEventData):
            return self.data.token
        return self.data.get("token") if isinstance(self.data, dict) else None

    @property
    def human_prompt(self) -> str | None:
        """HUMAN_INPUT_REQUIRED event prompt."""
        if isinstance(self.data, HumanInputRequiredEventData):
            return getattr(self.data.human_input, "question", None)
        return self.data.get("prompt") if isinstance(self.data, dict) else None

    @property
    def message_history(self) -> list[dict[str, Any]] | None:
        """Message history for resumption (from run_context)."""
        return self.run_context.messages


# ---------------------------------------------------------------------------
# Inherited Event Subclasses (Literal discriminated)
# ---------------------------------------------------------------------------


class StartingEvent(StreamEvent):
    """Typed STARTING event."""

    event_type: Literal[StreamEventType.STARTING] = StreamEventType.STARTING
    data: StartingEventData = Field(default_factory=StartingEventData)


class RunningEvent(StreamEvent):
    """Typed RUNNING event."""

    event_type: Literal[StreamEventType.RUNNING] = StreamEventType.RUNNING
    data: RunningEventData = Field(default_factory=RunningEventData)


class CompletedEvent(StreamEvent):
    """Typed COMPLETED event."""

    event_type: Literal[StreamEventType.COMPLETED] = StreamEventType.COMPLETED
    data: CompletedEventData


class FailedEvent(StreamEvent):
    """Typed FAILED event."""

    event_type: Literal[StreamEventType.FAILED] = StreamEventType.FAILED
    data: FailedEventData


class StreamingTokenEvent(StreamEvent):
    """Typed STREAMING event."""

    event_type: Literal[StreamEventType.STREAMING] = StreamEventType.STREAMING
    data: StreamingEventData


class ThinkingEvent(StreamEvent):
    """Typed THINKING event."""

    event_type: Literal[StreamEventType.THINKING] = StreamEventType.THINKING
    data: ThinkingEventData = Field(default_factory=ThinkingEventData)


class PartialOutputEvent(StreamEvent):
    """Typed PARTIAL event."""

    event_type: Literal[StreamEventType.PARTIAL] = StreamEventType.PARTIAL
    data: PartialEventData


class ToolCallingEvent(StreamEvent):
    """Typed TOOL_CALLING event."""

    event_type: Literal[StreamEventType.TOOL_CALLING] = StreamEventType.TOOL_CALLING
    data: ToolCallingEventData


class ToolResultEvent(StreamEvent):
    """Typed TOOL_RESULT event."""

    event_type: Literal[StreamEventType.TOOL_RESULT] = StreamEventType.TOOL_RESULT
    data: ToolResultEventData


class HumanInputRequiredEvent(StreamEvent):
    """Typed HUMAN_INPUT_REQUIRED event."""

    event_type: Literal[StreamEventType.HUMAN_INPUT_REQUIRED] = StreamEventType.HUMAN_INPUT_REQUIRED
    data: HumanInputRequiredEventData


class HumanResponseEvent(StreamEvent):
    """Typed HUMAN_RESPONSE event."""

    event_type: Literal[StreamEventType.HUMAN_RESPONSE] = StreamEventType.HUMAN_RESPONSE
    data: HumanResponseEventData


class StreamingMixin:
    """Mixin that adds astream() capability with validation.

    Pattern:
        astream() → validate_input → _astream() → validate_output on COMPLETED → yield

    Subclasses override _astream() for custom streaming logic.
    Default _astream() yields STARTING/RUNNING, calls _arun(), yields COMPLETED/FAILED.

    Usage:
        # Just get result
        result = await agent.arun(input_data)

        # Observe execution with streaming
        async for event in agent.astream(input_data):
            if event.event_type == StreamEventType.COMPLETED:
                result = event.output
    """

    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Public streaming API with input/output validation.

        Wraps _astream() with validation:
        - Input validation before any events are yielded
        - Output validation on COMPLETED event before yielding

        Args:
            params: Input parameters
            run_context: RunContext with run_id, messages, human_response, etc.
            **kwargs: Passed to _astream()

        Yields:
            StreamEvent: Events from _astream() (typically STARTING, RUNNING, COMPLETED/FAILED)

        Raises:
            Exception: Re-raises after _astream() yields FAILED event.
        """
        # Input validation (before any events are yielded)
        params = self._validate_input(params)

        # Stream from internal implementation
        async for event in self._astream(params, run_context, **kwargs):
            if event.event_type == StreamEventType.COMPLETED:
                # Validate output before yielding COMPLETED
                output = event.output
                if output is not None:
                    output = self._validate_output(output)
                    yield CompletedEvent(
                        source=event.source,
                        message=event.message,
                        data=CompletedEventData(output=output),
                        run_context=event.run_context,
                    )
                    continue
            yield event

    async def _astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Internal streaming implementation. Override for custom streaming.

        Default implementation yields STARTING/RUNNING, calls _arun(), yields COMPLETED/FAILED.

        Subclasses should:
        - Yield STARTING at the beginning
        - Yield RUNNING events for progress updates
        - Yield COMPLETED with {"output": result} on success
        - Yield FAILED with {"error": str, "error_type": str} on error, then re-raise

        Note: Input/output validation is handled by the parent astream() method.
        Do NOT call _validate_input() or _validate_output() here.

        Args:
            params: Input parameters (already validated by astream())
            run_context: RunContext with run_id, messages, human_response, etc.
            **kwargs: Additional arguments

        Yields:
            StreamEvent: STARTING, RUNNING, then COMPLETED or FAILED
        """
        class_name = self.__class__.__name__

        # Auto-generate run_id for event correlation
        run_context = (run_context or RunContext()).model_copy()
        run_context.run_id = run_context.run_id or uuid.uuid4().hex[:8]

        yield StartingEvent(
            source=class_name,
            message=f"Starting {class_name}",
            run_context=run_context,
        )

        try:
            yield RunningEvent(
                source=class_name,
                message=f"Running {class_name}",
                run_context=run_context,
            )

            output = await self._arun(params, **kwargs)

            yield CompletedEvent(
                source=class_name,
                message=f"Completed {class_name}",
                data=CompletedEventData(output=output),
                run_context=run_context,
            )
        except Exception as e:
            yield FailedEvent(
                source=class_name,
                message=f"Failed: {e!s}",
                data=FailedEventData(error=str(e), error_type=type(e).__name__),
                run_context=run_context,
            )
            raise


__all__ = [
    # Core
    "StreamEvent",
    "StreamEventType",
    "StreamingMixin",
    # Event data models
    "StartingEventData",
    "RunningEventData",
    "CompletedEventData",
    "FailedEventData",
    "StreamingEventData",
    "ThinkingEventData",
    "PartialEventData",
    "ToolCallingEventData",
    "ToolResultEventData",
    "HumanInputRequiredEventData",
    "HumanResponseEventData",
    "EventData",
    # Event subclasses
    "StartingEvent",
    "RunningEvent",
    "CompletedEvent",
    "FailedEvent",
    "StreamingTokenEvent",
    "ThinkingEvent",
    "PartialOutputEvent",
    "ToolCallingEvent",
    "ToolResultEvent",
    "HumanInputRequiredEvent",
    "HumanResponseEvent",
]
