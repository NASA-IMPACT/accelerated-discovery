"""Base IO schemas and protocol for guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, get_args, get_origin, runtime_checkable

from pydantic import Field, computed_field

from akd._base import InputSchema, OutputSchema
from akd.guardrails.categories._base import RiskCategory


class RiskCategoryValidationMixin:
    """Mixin for guardrail tools to validate input risk category types.

    Extracts the expected category type from the config's `risk_categories` field
    annotation (e.g., `list[GraniteRiskCategory]` → `GraniteRiskCategory`) and
    validates that input categories match.

    Tools using this mixin should have:
    - `self.config` with a `risk_categories` field
    - Optionally `self.config.validate_categories: bool` to enable/disable validation
    """

    config: Any  # Type stub - provided by BaseTool/BaseAgent at runtime

    def _get_supported_category_type(self) -> type[RiskCategory]:
        """Extract the category type from config.risk_categories field annotation.

        Returns:
            The specific category type (e.g., GraniteRiskCategory) or RiskCategory
            if generic (meaning any category is allowed).
        """
        if not hasattr(self, "config"):
            return RiskCategory

        # Get field info from config model class (not instance)
        config_class = type(self.config)
        field_info = config_class.model_fields.get("risk_categories")
        if not field_info:
            return RiskCategory

        annotation = field_info.annotation
        # Handle list[GraniteRiskCategory] -> GraniteRiskCategory
        if get_origin(annotation) is list:
            args = get_args(annotation)
            if args and args[0] is not RiskCategory:
                return args[0]

        return RiskCategory  # Generic = any allowed

    def _validate_category_types(
        self,
        categories: Sequence[RiskCategory],
    ) -> None:
        """Validate that all categories match the config's declared type.

        Args:
            categories: The categories to validate.

        Raises:
            TypeError: If any category is not an instance of the supported type.
        """
        # Check if validation is enabled (default: True if not specified)
        if not getattr(self.config, "validate_categories", True):
            return

        supported_type = self._get_supported_category_type()

        for cat in categories:
            if not isinstance(cat, supported_type):
                raise TypeError(
                    f"{self.__class__.__name__} only supports {supported_type.__name__}, "
                    f"got {type(cat).__name__}({cat.value})",
                )


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
    "RiskCategoryValidationMixin",
]
