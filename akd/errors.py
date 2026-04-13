"""AKD exceptions - re-exports from akd._base.errors for backward compatibility."""

from akd._base.errors import (
    AgentError,
    AKDError,
    GuardrailError,
    InputGuardrailTriggered,
    MaxToolCallsExceeded,
    MaxToolIterationsExceeded,
    OutputGuardrailTriggered,
    SchemaValidationError,
    ToolError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

__all__ = [
    "AKDError",
    "AgentError",
    "GuardrailError",
    "InputGuardrailTriggered",
    "MaxToolCallsExceeded",
    "MaxToolIterationsExceeded",
    "OutputGuardrailTriggered",
    "SchemaValidationError",
    "ToolError",
    "UnexpectedModelBehavior",
    "UsageLimitExceeded",
]
