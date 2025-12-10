"""Risk category types for guardrails validation."""

from akd.guardrails.categories._base import RiskCategory, RiskMetadata
from akd.guardrails.categories.granite import GraniteRiskCategory

# YAML-based categories - None if files don't exist
try:
    from akd.guardrails.categories.atlas import AtlasRiskCategory, ScienceRiskCategory
except Exception:
    AtlasRiskCategory = None  # type: ignore
    ScienceRiskCategory = None  # type: ignore

__all__ = [
    "RiskCategory",
    "RiskMetadata",
    "GraniteRiskCategory",
    "AtlasRiskCategory",
    "ScienceRiskCategory",
]
