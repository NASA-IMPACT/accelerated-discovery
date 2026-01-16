"""Streaming support for akd agents and tools."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
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
    THINKING = "thinking"

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
        THINKING: {"content": str, "source": str}
        TOOL_CALLING: {"tool_name": str, "tool_input": dict}
        TOOL_RESULT: {"tool_name": str, "tool_output": Any}

    Example:
        async for event in agent.astream(input_data):
            match event.event_type:
                case StreamEventType.COMPLETED:
                    result = event.output
                case StreamEventType.FAILED:
                    print(f"Error: {event.error}")
    """

    model_config = ConfigDict(use_enum_values=True)

    # Core envelope
    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str | None = None  # Class name that generated this event
    message: str | None = None

    # Flexible payload
    data: dict[str, Any] = Field(default_factory=dict)

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
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution with events.

        Wraps arun() with STARTING/COMPLETED/FAILED events.

        Args:
            params: Input parameters
            **kwargs: Passed to arun()

        Yields:
            StreamEvent: STARTING, then COMPLETED (with output) or FAILED

        Raises:
            Exception: Re-raises after yielding FAILED event.
        """
        class_name = self.__class__.__name__

        yield StreamEvent(
            event_type=StreamEventType.STARTING,
            source=class_name,
            message=f"Starting {class_name}",
        )

        try:
            yield StreamEvent(
                event_type=StreamEventType.RUNNING,
                source=class_name,
                message=f"Running {class_name}",
            )

            output = await self.arun(params, **kwargs)

            yield StreamEvent(
                event_type=StreamEventType.COMPLETED,
                source=class_name,
                message=f"Completed {class_name}",
                data={"output": output},
            )
        except Exception as e:
            yield StreamEvent(
                event_type=StreamEventType.FAILED,
                source=class_name,
                message=f"Failed: {e!s}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
            raise


__all__ = [
    "StreamEvent",
    "StreamEventType",
    "StreamingMixin",
]
