"""Dynamic risk categories loaded from YAML files."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from akd.guardrails.categories._base import RiskCategory, RiskMetadata


@cache
def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load YAML file with caching."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_risk_category_from_yaml(
    enum_name: str,
    yaml_path: Path,
) -> type[RiskCategory]:
    """
    Dynamically build a RiskCategory enum from YAML at import time.

    YAML format:
        risks:
          - id: "atlas-harmful-content"
            name: "Harmful Content"
            description: "Content that could cause harm"
            url: "https://..."
            concern: "..."
            tag: "harmful-content"
            type: "output"
    """
    data = _load_yaml(yaml_path)

    # Build enum members: {ENUM_KEY: (value, RiskMetadata)}
    members: dict[str, tuple[str, RiskMetadata]] = {}
    for risk in data.get("risks", []):
        risk_id = risk.get("id", "")
        # Convert to enum key: "atlas-harmful-content" -> "HARMFUL_CONTENT"
        enum_key = risk_id.upper().replace("-", "_").replace("ATLAS_", "")

        # Core fields
        description = risk.get("description", "")
        name = risk.get("name")
        severity = risk.get("severity", "normal")

        # Put everything else in extra
        extra = {k: v for k, v in risk.items() if k not in ("id", "description", "name", "severity") and v}

        metadata = RiskMetadata(
            description=description,
            name=name,
            severity=severity,
            extra=extra,
        )

        members[enum_key] = (risk_id, metadata)

    # Dynamically create enum class inheriting from RiskCategory
    # The functional API creates a new enum type with our custom __new__
    new_enum: type[RiskCategory] = RiskCategory(enum_name, members)  # type: ignore[assignment]
    return new_enum


# Build at module load time
_YAML_DIR = Path(__file__).parent

# Load if files exist, otherwise set to None for graceful degradation
_atlas_yaml = _YAML_DIR / "risk_atlas_data.yaml"
_science_yaml = _YAML_DIR / "science_lit_risks.yaml"

AtlasRiskCategory: type[RiskCategory] | None = None
ScienceRiskCategory: type[RiskCategory] | None = None

if _atlas_yaml.exists():
    AtlasRiskCategory = build_risk_category_from_yaml("AtlasRiskCategory", _atlas_yaml)

if _science_yaml.exists():
    ScienceRiskCategory = build_risk_category_from_yaml("ScienceRiskCategory", _science_yaml)
