"""
Unified Guardrails Package.

This package provides a unified interface for guardrails validation using
various providers (GraniteGuardianTool, RiskAgent, etc.).

Example:
    from akd.guardrails import GuardrailInput, GuardrailOutput, GraniteRiskCategory
    from akd.guardrails.providers import GraniteGuardianTool

    tool = GraniteGuardianTool()
    output = tool.check(GuardrailInput(content="How do I hack a system?"))
    print(output.passed, output.detected_risks)
"""

# Backward compatibility - re-export from legacy guardrails.py
import akd.guardrails_legacy as _legacy

# Base IO schemas and protocol
from akd.guardrails._base import GuardrailInput, GuardrailOutput, GuardrailProtocol

# Categories
from akd.guardrails.categories import (
    AtlasRiskCategory,
    GraniteRiskCategory,
    RiskCategory,
    RiskMetadata,
    ScienceRiskCategory,
)
from akd.guardrails.categories.granite import GraniteHarmCategory

# Composite guardrail
from akd.guardrails.providers.composite import CompositeGuardrail

# Legacy backward compat
add_guardrails = _legacy.add_guardrails
apply_guardrails = _legacy.apply_guardrails
GuardrailResult = _legacy.GuardrailResult
GuardrailsMixin = _legacy.GuardrailsMixin

__all__ = [
    # IO schemas and protocol
    "GuardrailInput",
    "GuardrailOutput",
    "GuardrailProtocol",
    # Composite
    "CompositeGuardrail",
    # Categories
    "RiskCategory",
    "RiskMetadata",
    "GraniteRiskCategory",
    "GraniteHarmCategory",
    "AtlasRiskCategory",
    "ScienceRiskCategory",
    # Legacy (backward compat)
    "add_guardrails",
    "apply_guardrails",
    "GuardrailResult",
    "GuardrailsMixin",
]
