"""Streaming support for akd agents and tools."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    GENERATING = "generating"  # Partial structured output as it streams

    # Tool calling (Stage 2)
    TOOL_CALLING = "tool_calling"
    TOOL_RESULT = "tool_result"


class StreamEvent(BaseModel):
    """Streaming event with CloudEvents-inspired design.

    Minimal core envelope with flexible payload. All event-specific
    data goes in `data` dict. Convenience properties provide ergonomic
    access to common fields.

    Core Fields:
        event_type: Type of event (STARTING, COMPLETED, FAILED, etc.)
        timestamp: UTC timestamp when event was created
        event_id: Unique identifier for this event
        source: Class name that generated this event
        message: Human-readable description
        data: Flexible payload for all event-specific data

    Data Keys by Event Type:
        COMPLETED: {"output": <OutputSchema>}
        FAILED: {"error": str, "error_type": str}
        STREAMING: {"token": str} for raw tokens as they arrive
        THINKING: {"thinking_content": str} for reasoning tokens (Claude/o1)
        GENERATING: {"partial_output": Any} for partial structured output
        TOOL_CALLING: {"tool_name": str, "tool_input": dict}
        TOOL_RESULT: {"tool_name": str, "tool_output": Any}

    Example:
        async for event in agent.astream(input_data):
            match event.event_type:
                case StreamEventType.THINKING:
                    print(f"Reasoning: {event.thinking_content}")
                case StreamEventType.GENERATING:
                    print(f"Partial: {event.partial_output}")
                case StreamEventType.COMPLETED:
                    result = event.output
                case StreamEventType.FAILED:
                    print(f"Error: {event.error}")
    """

    model_config = ConfigDict(use_enum_values=True)

    # Core envelope
    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str | None = None  # Class name that generated this event
    message: str | None = None

    # Flexible payload
    data: dict[str, Any] = Field(default_factory=dict)

    # Execution context
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution context (node_id, query, etc.)",
    )

    # Convenience properties

    @property
    def output(self) -> Any | None:
        """COMPLETED event output."""
        return self.data.get("output")

    @property
    def error(self) -> str | None:
        """FAILED event error message."""
        return self.data.get("error")

    @property
    def tool_name(self) -> str | None:
        """TOOL_CALLING/TOOL_RESULT tool name."""
        return self.data.get("tool_name")

    @property
    def tool_input(self) -> dict[str, Any] | None:
        """TOOL_CALLING input parameters."""
        return self.data.get("tool_input")

    @property
    def tool_output(self) -> Any | None:
        """TOOL_RESULT output value."""
        return self.data.get("tool_output")

    @property
    def partial_output(self) -> Any | None:
        """THINKING event partial output (partial structured response as it builds)."""
        return self.data.get("partial_output")

    @property
    def thinking_content(self) -> str | None:
        """THINKING event reasoning content (Claude extended thinking, o1 reasoning)."""
        return self.data.get("thinking_content")

    @property
    def token(self) -> str | None:
        """STREAMING event raw token."""
        return self.data.get("token")


class StreamingMixin:
    """Mixin that adds astream() capability.

    Wraps arun() with streaming events. arun() handles validation,
    logging, and calls _arun() internally.

    Flow:
        astream() → STARTING → arun() → COMPLETED/FAILED

    Override astream() for custom streaming (tool calling, thinking events).

    Usage:
        # Just get result
        result = await agent.arun(input_data)

        # Observe execution
        async for event in agent.astream(input_data):
            if event.event_type == StreamEventType.COMPLETED:
                result = event.output
    """

    async def astream(
        self,
        params: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution with events.

        Wraps arun() with STARTING/COMPLETED/FAILED events.

        Args:
            params: Input parameters
            context: Execution context (node_id, query, etc.). Auto-generates run_id if not provided.
            **kwargs: Passed to arun()

        Yields:
            StreamEvent: STARTING, then COMPLETED (with output) or FAILED

        Raises:
            Exception: Re-raises after yielding FAILED event.
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
            yield StreamEvent(
                event_type=StreamEventType.RUNNING,
                source=class_name,
                message=f"Running {class_name}",
                context=run_context,
            )

            output = await self.arun(params, **kwargs)

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


__all__ = [
    "StreamEvent",
    "StreamEventType",
    "StreamingMixin",
]
