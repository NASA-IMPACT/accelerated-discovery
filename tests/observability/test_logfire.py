"""Tests for observability (Logfire) integration: disabled in tests, redaction."""

import pytest

from akd.observability import (
    init_observability,
    is_observability_initialized,
    redact_for_telemetry,
    span_agent_run,
    span_tool_execution,
)


class TestInitObservability:
    """Telemetry is disabled during pytest to avoid flakiness and test data leakage."""

    def test_init_returns_false_under_pytest(self):
        """When running under pytest, init_observability does not enable Logfire."""
        result = init_observability()
        assert result is False
        assert is_observability_initialized() is False

    def test_spans_are_no_op_when_not_initialized(self):
        """Span context managers are no-op when telemetry not initialized."""
        with span_agent_run("TestAgent", run_id="r1"):
            pass
        with span_tool_execution("test_tool", "tc1"):
            pass


class TestRedactForTelemetry:
    """Redaction prevents PII/sensitive data from being sent to telemetry."""

    def test_redacts_sensitive_keys(self):
        out = redact_for_telemetry(
            {"api_key": "secret123", "query": "user prompt", "count": 5},
            redact_params=True,
        )
        assert out["api_key"] == "[redacted]"
        assert out["query"] == "[redacted]"
        assert out["count"] == 5

    def test_truncates_large_values(self):
        long_str = "x" * 600
        out = redact_for_telemetry({"key": long_str}, redact_params=False)
        assert len(out["key"]) < 600
        assert out["key"].endswith("...")

    def test_nested_redaction(self):
        out = redact_for_telemetry(
            {"nested": {"password": "pwd", "ok": 1}},
            redact_params=True,
        )
        assert out["nested"]["password"] == "[redacted]"
        assert out["nested"]["ok"] == 1

    def test_empty_dict(self):
        assert redact_for_telemetry({}) == {}
