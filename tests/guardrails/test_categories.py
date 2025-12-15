"""Tests for risk category enums."""

import pytest

from akd.guardrails.categories import (
    AtlasRiskCategory,
    GraniteRiskCategory,
    RiskCategory,
    ScienceRiskCategory,
)


class TestRiskCategories:
    """Test risk category enums are properly loaded."""

    def test_granite_risk_category_exists(self):
        """GraniteRiskCategory should exist and have members."""
        assert GraniteRiskCategory is not None
        assert issubclass(GraniteRiskCategory, RiskCategory)
        assert hasattr(GraniteRiskCategory, "JAILBREAK")

    def test_atlas_risk_category_loaded(self):
        """AtlasRiskCategory should be loaded from YAML."""
        assert AtlasRiskCategory is not None
        assert issubclass(AtlasRiskCategory, RiskCategory)

    def test_science_risk_category_loaded(self):
        """ScienceRiskCategory should be loaded from YAML."""
        assert ScienceRiskCategory is not None
        assert issubclass(ScienceRiskCategory, RiskCategory)
        assert hasattr(ScienceRiskCategory, "HALLUCINATION_IDENTIFICATION")

    @pytest.mark.parametrize(
        "category_cls",
        [
            pytest.param(GraniteRiskCategory, id="granite"),
            pytest.param(AtlasRiskCategory, id="atlas"),
            pytest.param(ScienceRiskCategory, id="science"),
        ],
    )
    def test_category_inherits_risk_category(self, category_cls):
        """All category enums should inherit from RiskCategory."""
        assert category_cls is not None
        assert issubclass(category_cls, RiskCategory)
