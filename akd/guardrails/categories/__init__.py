"""Risk category types and IO schemas for guardrails."""

from akd.guardrails.categories._base import (
    GuardrailCheckResult,
    GuardrailInput,
    GuardrailOutput,
    GuardrailProtocol,
    RiskCategory,
    RiskMetadata,
)
from akd.guardrails.categories.granite import GraniteHarmCategory, GraniteRiskCategory

# YAML-based categories - None if files don't exist
try:
    from akd.guardrails.categories.atlas import AtlasRiskCategory, ScienceRiskCategory
except Exception:
    AtlasRiskCategory = None  # type: ignore
    ScienceRiskCategory = None  # type: ignore

__all__ = [
    # Risk types
    "RiskCategory",
    "RiskMetadata",
    "GraniteRiskCategory",
    "GraniteHarmCategory",
    "AtlasRiskCategory",
    "ScienceRiskCategory",
    # IO schemas
    "GuardrailInput",
    "GuardrailOutput",
    "GuardrailCheckResult",
    # Protocol
    "GuardrailProtocol",
]
