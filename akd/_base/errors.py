"""Unified exception hierarchy for the AKD framework."""

from __future__ import annotations

from pydantic import BaseModel


class AKDError(Exception):
    """Base exception for all AKD errors."""

    pass


class AgentError(AKDError):
    """Base exception for agent-related errors."""

    pass


class UsageLimitExceeded(AgentError):
    """Raised when agent usage exceeds specified limits."""

    pass


class MaxToolIterationsExceeded(UsageLimitExceeded):
    """Raised when agent exceeds max tool iterations in ReAct loop."""

    pass


class UnexpectedModelBehavior(AgentError):
    """Raised when model produces unexpected output (empty, malformed)."""

    pass


class ToolError(AKDError):
    """Base exception for tool-related errors."""

    pass


class SchemaValidationError(AKDError):
    """Raised when schema validation fails."""

    pass


class GuardrailError(AKDError):
    """Base exception for guardrail violations.

    Attributes:
        output: The full GuardrailOutput object with all details (provider, risk_results, extra, etc.)
        detected_risks: Shortcut to output.detected_risks for convenience.
        risk_results: Shortcut to output.risk_results for per-risk details.
        provider: Which guardrail provider triggered the error.

    Note: output typed as BaseModel to avoid circular imports with GuardrailOutput.
    """

    def __init__(
        self,
        message: str,
        output: BaseModel | None = None,
    ) -> None:
        super().__init__(message)
        self.output = output
        # Convenience accessors
        self.detected_risks = getattr(output, "detected_risks", []) if output else []
        self.risk_results = getattr(output, "risk_results", {}) if output else {}
        self.provider = getattr(output, "provider", None) if output else None


class InputGuardrailTriggered(GuardrailError):
    """Raised when input validation fails and fail_on_risk=True."""

    pass


class OutputGuardrailTriggered(GuardrailError):
    """Raised when output validation fails and fail_on_risk=True."""

    pass
