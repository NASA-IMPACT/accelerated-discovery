"""
Unified Guardrails Package.

This package provides a unified interface for guardrails validation using
various providers (GraniteGuardianTool, RiskAgent, etc.).

Example:
    from akd.guardrails import RiskCategory, RiskMetadata, GraniteRiskCategory
"""

# Backward compatibility - re-export from legacy guardrails.py
# Import the module to avoid circular import, then re-export
import akd.guardrails_legacy as _legacy

# Categories - always available
from akd.guardrails.categories import (
    AtlasRiskCategory,
    GraniteRiskCategory,
    RiskCategory,
    RiskMetadata,
    ScienceRiskCategory,
)

add_guardrails = _legacy.add_guardrails
apply_guardrails = _legacy.apply_guardrails
GuardrailResult = _legacy.GuardrailResult
GuardrailsMixin = _legacy.GuardrailsMixin

__all__ = [
    # Categories
    "RiskCategory",
    "RiskMetadata",
    "GraniteRiskCategory",
    "AtlasRiskCategory",
    "ScienceRiskCategory",
    # Legacy (backward compat)
    "add_guardrails",
    "apply_guardrails",
    "GuardrailResult",
    "GuardrailsMixin",
]
