"""Observability (Logfire) bootstrap. Call init_observability() from script entrypoints."""

from __future__ import annotations

from akd.observability._logfire import (
    init_observability,
    is_observability_initialized,
    log_stream_event,
    redact_for_telemetry,
    span_agent_run,
    span_tool_execution,
)

__all__ = [
    "init_observability",
    "is_observability_initialized",
    "log_stream_event",
    "redact_for_telemetry",
    "span_agent_run",
    "span_tool_execution",
]
