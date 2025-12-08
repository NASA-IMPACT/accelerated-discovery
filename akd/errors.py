from __future__ import annotations


class AKDError(Exception):
    """Base exception for agent-related errors."""

    pass


class SchemaValidationError(AKDError):
    """Raised when schema validation fails."""

    pass


class GuardrailError(AKDError):
    """Base exception for guardrail violations.

    Note: detected_risks is typed as Any to avoid circular imports.
    At runtime, it will be list[GuardrailResult] from akd.guardrails.
    """

    def __init__(
        self,
        message: str,
        detected_risks: list | None = None,
    ) -> None:
        super().__init__(message)
        self.detected_risks = detected_risks or []


class InputGuardrailTriggered(GuardrailError):
    """Raised when input validation fails and fail_on_risk=True."""

    pass


class OutputGuardrailTriggered(GuardrailError):
    """Raised when output validation fails and fail_on_risk=True."""

    pass
