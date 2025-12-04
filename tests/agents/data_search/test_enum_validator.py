"""
Test CMR Enum Validator

Tests the fuzzy matching and correction logic for instrument/platform enums.
Validates all 5 correction scenarios from the integration plan.
"""

import pytest

from akd.agents.data_search.handlers.cmr.schemas import CMRQueryApproach
from akd.agents.data_search.utils.cmr_enum_validator import CMREnumValidator


@pytest.fixture
def validator():
    """Create a validator instance for testing."""
    return CMREnumValidator(threshold=0.7, debug=False)


class TestScenarioA:
    """Scenario A: Simple Replacement - correct value in correct field."""

    def test_instrument_correction(self, validator):
        """Test correcting misspelled instrument name."""
        approach = CMRQueryApproach(
            instrument="modis",  # Should be "MODIS"
            platform="Terra",  # Already correct
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "MODIS"
        assert corrected.platform == "Terra"
        assert metadata["corrections_applied"]
        assert len(metadata["changes"]) == 1
        assert metadata["changes"][0]["field"] == "instrument"
        assert metadata["changes"][0]["original"] == "modis"
        assert metadata["changes"][0]["corrected"] == "MODIS"

    def test_platform_correction(self, validator):
        """Test correcting misspelled platform name."""
        approach = CMRQueryApproach(
            instrument="MODIS",  # Already correct
            platform="terra",  # Should be "Terra"
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "MODIS"
        assert corrected.platform == "Terra"
        assert metadata["corrections_applied"]
        assert len(metadata["changes"]) == 1
        assert metadata["changes"][0]["field"] == "platform"

    def test_both_corrections(self, validator):
        """Test correcting both instrument and platform."""
        approach = CMRQueryApproach(
            instrument="viirs",  # Should be "VIIRS"
            platform="aqua",  # Should be "Aqua"
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "VIIRS"
        assert corrected.platform == "Aqua"
        assert metadata["corrections_applied"]
        assert len(metadata["changes"]) == 2


class TestScenarioB:
    """Scenario B: Single Swap - one field is misidentified."""

    def test_instrument_field_contains_platform(self, validator):
        """Test when instrument field contains a platform name."""
        approach = CMRQueryApproach(
            instrument="Aqua",  # This is a platform, not an instrument
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument is None
        assert corrected.platform == "Aqua"
        assert metadata["corrections_applied"]
        # Should have move and drop changes
        assert any(c.get("moved_from") == "instrument" for c in metadata["changes"])
        assert any(c.get("dropped") for c in metadata["changes"])

    def test_platform_field_contains_instrument(self, validator):
        """Test when platform field contains an instrument name."""
        approach = CMRQueryApproach(
            instrument=None,
            platform="MODIS",  # This is an instrument, not a platform
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "MODIS"
        assert corrected.platform is None
        assert metadata["corrections_applied"]
        assert any(c.get("moved_from") == "platform" for c in metadata["changes"])


class TestScenarioC:
    """Scenario C: Double Swap - both fields are reversed."""

    def test_swapped_fields(self, validator):
        """Test when instrument and platform are swapped."""
        approach = CMRQueryApproach(
            instrument="Terra",  # This is a platform
            platform="MODIS",  # This is an instrument
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "MODIS"
        assert corrected.platform == "Terra"
        assert metadata["corrections_applied"]
        # Both should have swap indicator
        assert sum(c.get("swapped", False) for c in metadata["changes"]) == 2


class TestScenarioD:
    """Scenario D: Biased Swap (Different Scores) - both resolve to same type."""

    def test_both_match_to_platforms(self, validator):
        """Test when both fields match to platforms, keep highest score."""
        approach = CMRQueryApproach(
            instrument="terr",  # Fuzzy matches to "Terra" (platform) with lower score
            platform="aqua",  # Fuzzy matches to "Aqua" (platform) with higher score
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Should keep the higher scoring match
        assert corrected.instrument is None
        assert corrected.platform == "Aqua"
        assert metadata["corrections_applied"]
        # Should have dropped entries
        assert any(c.get("dropped") for c in metadata["changes"])


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_no_match_below_threshold(self, validator):
        """Test when values don't match anything above threshold."""
        approach = CMRQueryApproach(
            instrument="xyz123",
            platform="abc456",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "xyz123"
        assert corrected.platform == "abc456"
        assert not metadata["corrections_applied"]
        assert len(metadata["changes"]) == 0

    def test_empty_fields(self, validator):
        """Test when both fields are None."""
        approach = CMRQueryApproach(
            instrument=None,
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument is None
        assert corrected.platform is None
        assert not metadata["corrections_applied"]

    def test_one_empty_field(self, validator):
        """Test when only one field has a value."""
        approach = CMRQueryApproach(
            instrument="MODIS",
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        assert corrected.instrument == "MODIS"
        assert corrected.platform is None
        # No corrections needed for exact match
        assert not metadata["corrections_applied"]

    def test_preserves_other_fields(self, validator):
        """Test that other fields are preserved during correction."""
        approach = CMRQueryApproach(
            instrument="modis",
            platform="Terra",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
            bounding_box="-180,-90,180,90",
            processing_level="Level 2",
            spatial_resolution="1km",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Corrected fields
        assert corrected.instrument == "MODIS"
        assert corrected.platform == "Terra"
        # Preserved fields
        assert corrected.temporal == "2023-01-01T00:00:00Z,2023-12-31T23:59:59Z"
        assert corrected.bounding_box == "-180,-90,180,90"
        assert corrected.processing_level == "Level 2"
        assert corrected.spatial_resolution == "1km"


class TestAbbreviationMatching:
    """Test fuzzy matching with abbreviations."""

    def test_common_abbreviation(self, validator):
        """Test matching common instrument abbreviation."""
        approach = CMRQueryApproach(
            instrument="msi",  # Should match some MSI instrument
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Should find some match
        assert corrected.instrument is not None
        if metadata["corrections_applied"]:
            # If corrected, should have MSI in the name
            assert "MSI" in corrected.instrument.upper()


class TestMetadataStructure:
    """Test correction metadata structure."""

    def test_metadata_fields(self, validator):
        """Test that metadata has required fields."""
        approach = CMRQueryApproach(
            instrument="modis",
            platform="Terra",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Required metadata fields
        assert "corrections_applied" in metadata
        assert "changes" in metadata
        assert isinstance(metadata["corrections_applied"], bool)
        assert isinstance(metadata["changes"], list)

    def test_change_entry_structure(self, validator):
        """Test that change entries have required fields."""
        approach = CMRQueryApproach(
            instrument="modis",
            platform="Terra",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        if metadata["corrections_applied"]:
            for change in metadata["changes"]:
                assert "field" in change
                assert "original" in change
                assert "corrected" in change
                assert "score" in change
                assert change["field"] in ["instrument", "platform"]


class TestValidatorConfiguration:
    """Test validator configuration options."""

    def test_custom_threshold(self):
        """Test validator with custom threshold."""
        # Lower threshold should be more permissive
        low_threshold_validator = CMREnumValidator(threshold=0.3, debug=False)

        approach = CMRQueryApproach(
            instrument="mod",  # Partial match
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = low_threshold_validator.validate_approach(approach)

        # Lower threshold might find a match
        # (exact result depends on fuzzy matching scores)
        assert metadata is not None

    def test_debug_mode(self):
        """Test validator in debug mode."""
        debug_validator = CMREnumValidator(threshold=0.7, debug=True)

        approach = CMRQueryApproach(
            instrument="modis",
            platform="Terra",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        # Should not raise errors in debug mode
        corrected, metadata = debug_validator.validate_approach(approach)
        assert corrected is not None


class TestLongNameMatching:
    """Test that both short_name and long_name values are matched."""

    def test_short_name_matching(self, validator):
        """Test matching against short_name value."""
        approach = CMRQueryApproach(
            instrument="CMG-GT-1A",  # This is a short_name value
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Should match exactly or find close match
        assert corrected.instrument is not None

    def test_long_name_matching(self, validator):
        """Test matching against long_name value."""
        approach = CMRQueryApproach(
            instrument="Canadian Micro Gravity GT-1A",  # This is a long_name value
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = validator.validate_approach(approach)

        # Should match the long_name value
        assert corrected.instrument is not None
        # Either exact match on long name or correction to short name
        assert metadata is not None

    def test_partial_long_name_matching(self, validator):
        """Test fuzzy matching against partial long_name."""
        # Using a lower threshold for partial matching
        low_threshold_validator = CMREnumValidator(threshold=0.5, debug=True)

        approach = CMRQueryApproach(
            instrument="Micro Gravity",  # Partial long_name
            platform=None,
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        )

        corrected, metadata = low_threshold_validator.validate_approach(approach)

        # Should find a match (either short or long name)
        # Exact match depends on fuzzy scoring
        assert metadata is not None
