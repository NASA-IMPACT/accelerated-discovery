"""Central Logfire initialization and span helpers. No import-time side effects."""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from akd.configs.project import TelemetrySettings

_initialized = False

# Keys that should never be sent in telemetry (API keys, tokens, prompts)
_REDACT_KEYS = frozenset(
    {
        "api_key",
        "token",
        "password",
        "secret",
        "authorization",
        "query",
        "content",
        "text",
        "message",
        "prompt",
        "input",
        "arguments",
    }
)
_MAX_PAYLOAD_CHARS = 500


def _is_test_run() -> bool:
    """Return True if we are running under pytest (telemetry should be disabled)."""
    return "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def redact_for_telemetry(
    data: dict[str, Any],
    *,
    redact_params: bool = True,
    max_value_chars: int = _MAX_PAYLOAD_CHARS,
) -> dict[str, Any]:
    """Return a copy of data safe for telemetry: sensitive keys redacted, large values truncated."""
    if not data:
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        key_lower = k.lower()
        if redact_params and any(s in key_lower for s in _REDACT_KEYS):
            out[k] = "[redacted]"
            continue
        if isinstance(v, str) and len(v) > max_value_chars:
            out[k] = v[:max_value_chars] + "..."
        elif isinstance(v, dict):
            out[k] = redact_for_telemetry(v, redact_params=redact_params, max_value_chars=max_value_chars)
        else:
            out[k] = v
    return out


def is_observability_initialized() -> bool:
    """Return True if Logfire was configured (init_observability succeeded)."""
    return _initialized


@contextmanager
def _noop_context():
    """No-op context manager when telemetry is disabled."""
    yield


@contextmanager
def span_agent_run(
    agent_name: str,
    run_id: str | None = None,
    *,
    params_safe: dict[str, Any] | None = None,
):
    """Context manager for an agent/tool run span. No-op when telemetry not initialized."""
    if not _initialized:
        yield
        return
    import logfire

    rid = run_id or str(uuid.uuid4())[:8]
    attrs: dict[str, Any] = {"agent": agent_name, "run_id": rid}
    if params_safe is not None:
        attrs["params_keys"] = list(params_safe.keys()) if isinstance(params_safe, dict) else []
    with logfire.span("akd.agent.run", **attrs):
        yield


@contextmanager
def span_tool_execution(
    tool_name: str,
    tool_call_id: str,
    *,
    run_id: str | None = None,
):
    """Context manager for a tool execution span. No-op when telemetry not initialized."""
    if not _initialized:
        yield
        return
    import logfire

    attrs: dict[str, Any] = {"tool_name": tool_name, "tool_call_id": tool_call_id}
    if run_id is not None:
        attrs["run_id"] = run_id
    with logfire.span("akd.tool.execute", **attrs):
        yield


def log_stream_event(
    event_type: str,
    *,
    run_id: str | None = None,
    source: str | None = None,
    is_lifecycle: bool = True,
) -> None:
    """Log a streaming lifecycle event. No-op when telemetry not initialized.

    Use for STARTING, COMPLETED, FAILED, TOOL_CALLING, TOOL_RESULT, HUMAN_INPUT_REQUIRED.
    Do not call for every STREAMING token (high volume). HUMAN_INPUT_REQUIRED is
    logged as lifecycle, not error.
    """
    if not _initialized:
        return
    import logfire

    attrs: dict[str, Any] = {"event_type": event_type, "is_lifecycle": is_lifecycle}
    if run_id is not None:
        attrs["run_id"] = run_id
    if source is not None:
        attrs["source"] = source
    logfire.info("akd.stream.event", **attrs)


def init_observability(
    settings: TelemetrySettings | None = None,
) -> bool:
    """Initialize Logfire once. Safe to call from script entrypoints.

    If telemetry is disabled or logfire is not installed, returns False and does nothing.
    Idempotent: only the first successful call configures Logfire.

    Args:
        settings: Telemetry settings. Defaults to get_project_settings().telemetry.

    Returns:
        True if Logfire was configured, False otherwise.
    """
    global _initialized
    if _initialized:
        return True

    if settings is None:
        from akd.configs.project import get_project_settings

        settings = get_project_settings().telemetry

    if not settings.enabled:
        return False
    if _is_test_run():
        return False

    try:
        import logfire
    except ImportError:
        return False

    logfire.configure(
        service_name=settings.service_name,
        environment=settings.environment,
        send_to_logfire=settings.send_to_cloud,
        trace_sample_rate=settings.sample_rate,
    )
    _initialized = True
    return True
