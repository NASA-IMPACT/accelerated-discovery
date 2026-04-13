"""Base risk category types and IO schemas for guardrails."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, computed_field


class RiskMetadata(BaseModel):
    """Metadata for a risk category. Extensible for YAML fields."""

    description: str = Field(default="", description="Risk description")
    severity: Literal["low", "normal", "high"] = Field(default="normal")
    name: str | None = Field(default=None, description="Display name")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (url, concern, tag, etc.)")


class RiskCategory(StrEnum):
    """
    Base class for typed risk categories with inline metadata.

    Subclasses define members as tuples: (value, RiskMetadata)

    Example:
        class MyRisks(RiskCategory):
            HARM = ("harm", RiskMetadata(description="General harmful content", severity="high"))
            VIOLENCE = ("violence", RiskMetadata(description="Violence-related", severity="high"))
            SPAM = ("spam", RiskMetadata())  # uses defaults
    """

    metadata: RiskMetadata

    def __new__(cls, value: str, metadata: RiskMetadata | None = None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.metadata = metadata or RiskMetadata()
        return obj

    @property
    def description(self) -> str:
        """Convenience accessor for metadata.description."""
        return self.metadata.description

    @property
    def severity(self) -> str:
        """Convenience accessor for metadata.severity."""
        return self.metadata.severity

    @classmethod
    def from_string(cls, value: str) -> RiskCategory:
        """Convert a string to a RiskCategory enum member."""
        try:
            return cls(value)
        except ValueError:
            normalized = value.lower().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == normalized:
                    return member
            raise ValueError(f"'{value}' is not a valid {cls.__name__}")


# --- IO Schemas ---


class GuardrailInput(BaseModel):
    """
    Unified input for all guardrail providers.

    Supports both single-turn (user message only) and multi-turn
    (user + assistant) scenarios.
    """

    user_message: str = Field(..., description="User input text")
    assistant_message: str | None = Field(None, description="Assistant response to check")
    context: str | None = Field(None, description="Additional context (e.g., RAG retrieved docs)")
    risk_categories: list[str] = Field(
        default_factory=list,
        description="Risk category IDs to check (empty = provider defaults)",
    )

    @classmethod
    def single_turn(cls, user_message: str, risk_categories: list[str] | None = None) -> GuardrailInput:
        """Create input for single-turn (user message only) check."""
        return cls(user_message=user_message, risk_categories=risk_categories or [])

    @classmethod
    def multi_turn(
        cls,
        user_message: str,
        assistant_message: str,
        risk_categories: list[str] | None = None,
        context: str | None = None,
    ) -> GuardrailInput:
        """Create input for multi-turn (user + assistant) check."""
        return cls(
            user_message=user_message,
            assistant_message=assistant_message,
            risk_categories=risk_categories or [],
            context=context,
        )


class GuardrailCheckResult(BaseModel):
    """Result of a single risk category check."""

    risk_id: str = Field(..., description="Risk category ID that was checked")
    is_risky: bool = Field(..., description="Whether risk was detected")
    confidence: float | None = Field(None, description="Confidence score 0-1")
    explanation: str | None = Field(None, description="Human-readable explanation")
    metadata: dict[str, Any] | None = Field(None, description="Provider-specific data")


class GuardrailOutput(BaseModel):
    """
    Unified output from any guardrail provider.

    Contains results for each requested risk category, plus transparency
    about all risks detected by the underlying model.
    """

    results: list[GuardrailCheckResult] = Field(default_factory=list)

    # For multi-harm models: all risks detected (before filtering to requested)
    all_detected_risks: list[str] = Field(
        default_factory=list,
        description="All risks detected by model (unfiltered, for transparency)",
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific additional data",
    )

    @computed_field
    @property
    def passed(self) -> bool:
        """True if no risks detected in filtered results."""
        return not any(r.is_risky for r in self.results)

    @computed_field
    @property
    def triggered_risks(self) -> list[str]:
        """Risk IDs where risk was detected (filtered to requested risks)."""
        return [r.risk_id for r in self.results if r.is_risky]


# --- Provider Protocol ---


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocol for guardrail implementations (Granite, Atlas, etc.).

    Any class implementing check() can be used as a guardrail.
    """

    def check(self, input: GuardrailInput) -> GuardrailOutput:
        """Run guardrail check and return unified output."""
        ...

    async def acheck(self, input: GuardrailInput) -> GuardrailOutput:
        """Async version of check."""
        ...
