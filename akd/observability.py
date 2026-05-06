from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import logfire

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "token",
    "password",
    "secret",
    "openai_api_key",
    "serper_api_key",
    "logfire_token",
}

_INITIALIZED = False


def identity_tags() -> dict[str, str]:
    return {
        "team_id": os.getenv("AKD_TEAM_ID", "unknown"),
        "team_name": os.getenv("AKD_TEAM_NAME", "unknown"),
        "seat_id": os.getenv("AKD_SEAT_ID", "unknown"),
        "operator_id": os.getenv("AKD_OPERATOR_ID", "unknown"),
        "workspace_id": os.getenv("AKD_WORKSPACE_ID", "unknown"),
        "cost_center": os.getenv("AKD_COST_CENTER", "unknown"),
    }


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_observability_enabled() -> bool:
    return _env_flag("LOGFIRE_ENABLED", default=False)


def _scrub_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return scrub_payload(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_scrub_value(v) for v in value]
    return value


def scrub_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in SENSITIVE_KEYS:
            redacted[key] = "[REDACTED]"
            continue
        redacted[key] = _scrub_value(value)
    return redacted


def run_tags(
    *,
    workflow_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    parent_run_id: str | None = None,
    control_layer: str | None = None,
    provider_runtime: str | None = None,
    repo: str | None = None,
    agent_name: str | None = None,
    tool_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    tags: dict[str, Any] = dict(identity_tags())
    if workflow_id:
        tags["workflow_id"] = workflow_id
    if run_id:
        tags["run_id"] = run_id
    if session_id:
        tags["session_id"] = session_id
    if request_id:
        tags["request_id"] = request_id
    if parent_run_id:
        tags["parent_run_id"] = parent_run_id
    if control_layer:
        tags["control_layer"] = control_layer
    if provider_runtime:
        tags["provider_runtime"] = provider_runtime
    if repo:
        tags["repo"] = repo
    if agent_name:
        tags["agent_name"] = agent_name
    if tool_name:
        tags["tool_name"] = tool_name
    if model:
        tags["model"] = model
    return tags


def init_observability(service_name: str = "accelerated-discovery") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    if not is_observability_enabled():
        _INITIALIZED = True
        return

    token = os.getenv("LOGFIRE_TOKEN")
    configure_kwargs: dict[str, Any] = {
        "service_name": service_name,
        "environment": os.getenv("LOGFIRE_ENV", os.getenv("ENV", "local")),
    }
    if token:
        configure_kwargs["token"] = token
    logfire.configure(**configure_kwargs)

    if _env_flag("LOGFIRE_INSTRUMENT_HTTPX", default=True):
        logfire.instrument_httpx(capture_headers=False)

    _INITIALIZED = True
