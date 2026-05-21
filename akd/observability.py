"""Minimal Logfire setup for LiteLLM auto-instrumentation."""

from __future__ import annotations

import os
import sys

import logfire

_INITIALIZED = False


def _enabled() -> bool:
    return os.getenv("LOGFIRE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _warn_instrument(name: str, exc: Exception) -> None:
    msg = str(exc).strip().split("\n")[0]
    print(f"warning: logfire.{name} skipped ({msg}).", file=sys.stderr)


def init_observability(service_name: str = "accelerated-discovery") -> None:
    """Configure Logfire and instrument LiteLLM (call before agent imports)."""
    global _INITIALIZED
    if _INITIALIZED or not _enabled():
        _INITIALIZED = True
        return

    token = os.getenv("LOGFIRE_TOKEN")
    kwargs: dict[str, object] = {
        "service_name": service_name,
        "environment": os.getenv("LOGFIRE_ENV", os.getenv("ENV", "local")),
        "inspect_arguments": False,
    }
    if token:
        kwargs["token"] = token
    logfire.configure(**kwargs)

    try:
        logfire.instrument_litellm()
    except Exception as exc:
        _warn_instrument("instrument_litellm", exc)

    if _env_flag("LOGFIRE_INSTRUMENT_HTTPX", default=True):
        try:
            logfire.instrument_httpx(capture_headers=False)
        except Exception as exc:
            _warn_instrument("instrument_httpx", exc)

    _INITIALIZED = True
