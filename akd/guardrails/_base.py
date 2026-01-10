"""Base IO schemas and protocol for guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, get_args, get_origin, runtime_checkable

if TYPE_CHECKING:
    from akd.guardrails.providers.composite import CompositeGuardrail

from pydantic import Field, computed_field

from akd._base import InputSchema, OutputSchema
from akd.errors import GuardrailError
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
    For multi-turn, use `from_multi_turn()` classmethod.
    """

    content: str = Field(..., description="Content to check for risks")
    context: str | None = Field(None, description="Optional context (prior conversation, RAG docs)")
    source_context: str | None = Field(
        None,
        description="Context about the source (agent/tool) that produced the content: its purpose, expected behavior, and constraints.",
    )
    risk_categories: Sequence[RiskCategory] = Field(
        default_factory=list,
        description="Risk categories to check (empty = provider defaults)",
    )

    @classmethod
    def from_multi_turn(
        cls,
        inputs: list[str],
        outputs: list[str],
        risk_categories: Sequence[RiskCategory] | None = None,
        additional_context: str | None = None,
    ) -> "GuardrailInput":
        """Create from multi-turn conversation.

        Args:
            inputs: List of user messages (chronological order).
            outputs: List of model responses (aligned with inputs by index).
            risk_categories: Risk categories to check.
            additional_context: Extra info (agent name/description, process info) prepended to context.

        Returns:
            GuardrailInput with last output as content, prior turns + additional_context as context.
        """
        if len(inputs) != len(outputs):
            raise ValueError(f"inputs and outputs must have same length, got {len(inputs)} and {len(outputs)}")

        if not inputs:
            raise ValueError("inputs cannot be empty")

        # Format turns as context
        if len(inputs) > 1:
            prior_turns = "\n\n".join(
                f"Turn {i + 1}:\nUser: {inp}\nModel: {outp}"
                for i, (inp, outp) in enumerate(zip(inputs[:-1], outputs[:-1]))
            )
            context = f"{prior_turns}\n\nTurn {len(inputs)}:\nUser: {inputs[-1]}"
        else:
            context = inputs[0]

        # Prepend additional context if provided
        if additional_context:
            context = f"{additional_context}\n\n{context}"

        return cls(
            content=outputs[-1],
            context=context,
            risk_categories=list(risk_categories or []),
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


class GuardrailOperatorMixin:
    """Mixin to add &, |, and >> operators to guardrail implementations.

    Operators:
        g1 & g2  → CompositeGuardrail with mode=ALL (both must pass)
        g1 | g2  → CompositeGuardrail with mode=ANY (at least one passes)
        g1 >> g2 → CompositeGuardrail with mode=FAIL_FAST (sequential, stop on first fail)

    Example:
        granite = GraniteGuardianTool()
        risk = RiskAgent()

        combined = granite & risk          # Both must pass
        combined = granite | risk          # Either passes
        combined = granite >> risk         # Sequential, stop on first fail
        combined = (granite >> risk) | fallback  # Nested composition
    """

    def _validate_other_guardrail(self, other: GuardrailProtocol) -> None:
        """Validate that other is a GuardrailProtocol instance."""
        if not isinstance(other, GuardrailProtocol):
            raise GuardrailError(f"{other} does not follow GuardrailProtocol")

    def __and__(self, other: GuardrailProtocol) -> "CompositeGuardrail":
        """g1 & g2 → ALL mode (both must pass, parallel)."""
        from akd.guardrails.providers.composite import (
            CompositeGuardrail,
            CompositeGuardrailMode,
        )

        self._validate_other_guardrail(other)

        # Flatten nested composites of same mode
        if isinstance(self, CompositeGuardrail) and self.mode == CompositeGuardrailMode.ALL:
            return CompositeGuardrail(
                guardrails=[*self.guardrails, other],
                mode=CompositeGuardrailMode.ALL,
            )
        return CompositeGuardrail(
            guardrails=[self, other],
            mode=CompositeGuardrailMode.ALL,
        )

    def __or__(self, other: GuardrailProtocol) -> "CompositeGuardrail":
        """g1 | g2 → ANY mode (at least one must pass, parallel)."""
        from akd.guardrails.providers.composite import (
            CompositeGuardrail,
            CompositeGuardrailMode,
        )

        self._validate_other_guardrail(other)

        if isinstance(self, CompositeGuardrail) and self.mode == CompositeGuardrailMode.ANY:
            return CompositeGuardrail(
                guardrails=[*self.guardrails, other],
                mode=CompositeGuardrailMode.ANY,
            )
        return CompositeGuardrail(
            guardrails=[self, other],
            mode=CompositeGuardrailMode.ANY,
        )

    def __rshift__(self, other: GuardrailProtocol) -> "CompositeGuardrail":
        """g1 >> g2 → FAIL_FAST mode (sequential, stop on first fail)."""
        from akd.guardrails.providers.composite import (
            CompositeGuardrail,
            CompositeGuardrailMode,
        )

        self._validate_other_guardrail(other)

        if isinstance(self, CompositeGuardrail) and self.mode == CompositeGuardrailMode.FAIL_FAST:
            return CompositeGuardrail(
                guardrails=[*self.guardrails, other],
                mode=CompositeGuardrailMode.FAIL_FAST,
            )
        return CompositeGuardrail(
            guardrails=[self, other],
            mode=CompositeGuardrailMode.FAIL_FAST,
        )


__all__ = [
    "GuardrailInput",
    "GuardrailOutput",
    "GuardrailProtocol",
    "GuardrailOperatorMixin",
    "RiskCategoryValidationMixin",
]
