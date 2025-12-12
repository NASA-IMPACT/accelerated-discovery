"""Base IO schemas and protocol for guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, computed_field

from akd._base import InputSchema, OutputSchema
from akd.guardrails.categories._base import RiskCategory


class GuardrailInput(InputSchema):
    """
    Unified input for guardrail validation.

    Simple: content to check + optional context.
    For multi-turn, put the response to check in content, prior conversation in context.
    """

    content: str = Field(..., description="Content to check for risks")
    context: str | None = Field(None, description="Optional context (prior conversation, RAG docs)")
    risk_categories: Sequence[RiskCategory] = Field(
        default_factory=list,
        description="Risk categories to check (empty = provider defaults)",
    )


class GuardrailOutput(OutputSchema):
    """
    Unified output from guardrail check.

    Only detected risks are listed. Category metadata (description, severity)
    comes from the RiskCategory enum itself.
    """

    __response_field__ = "summary"

    detected_risks: Sequence[RiskCategory] = Field(
        default_factory=list,
        description="Risk categories that were detected",
    )
    risk_results: dict[RiskCategory, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-risk evaluation details (criteria, score, metadata)",
    )

    provider: str | None = Field(
        None,
        description="Name of the guardrail provider (e.g., GraniteGuardianTool, RiskAgent)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific data (raw response, confidence, etc.)",
    )

    @computed_field
    @property
    def passed(self) -> bool:
        """True if no risks detected."""
        return len(self.detected_risks) == 0

    @property
    def summary(self) -> str:
        """Summary of guardrail check results."""
        if self.passed:
            return "No risks detected"
        return f"Risks detected: {', '.join(str(r) for r in self.detected_risks)}"


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocol for guardrail implementations (Granite, Atlas, etc.).

    Any class implementing check() can be used as a guardrail.
    """

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Run guardrail check and return unified output."""
        ...

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Async version of check."""
        ...


__all__ = [
    "GuardrailInput",
    "GuardrailOutput",
    "GuardrailProtocol",
]
