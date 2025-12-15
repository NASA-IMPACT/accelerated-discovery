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

# New decorator API
from akd.guardrails.decorators import apply_guardrails, guardrail

# Composite guardrail
from akd.guardrails.providers.composite import (
    CompositeGuardrail,
    CompositeGuardrailMode,
)

__all__ = [
    # IO schemas and protocol
    "GuardrailInput",
    "GuardrailOutput",
    "GuardrailProtocol",
    # Composite
    "CompositeGuardrail",
    "CompositeGuardrailMode",
    # Categories
    "RiskCategory",
    "RiskMetadata",
    "GraniteRiskCategory",
    "GraniteHarmCategory",
    "AtlasRiskCategory",
    "ScienceRiskCategory",
    # Decorator API
    "guardrail",
    "apply_guardrails",
]
