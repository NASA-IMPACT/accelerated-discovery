"""
CMR Enum Validator

Validates and corrects instrument/platform values in CMR query approaches
using fuzzy matching against official CMR enumerations.

Implements the enum integration plan with 5 correction scenarios:
A. Simple Replacement - correct value in correct field
B. Single Swap - one field is misidentified
C. Double Swap - both fields are reversed
D. Biased Swap (Different Scores) - both resolve to same type, keep highest score
E. Biased Swap (Same Scores) - both resolve to same type, keep original field match
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

from .improved_fuzzy_matcher import load_cmr_data, rank_instrument_platform


class CMREnumValidator:
    """
    Validates and corrects CMR instrument/platform values using fuzzy matching.

    After the LLM extracts known parameters, this validator ensures that
    instrument and platform values match CMR's controlled vocabulary.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        debug: bool = False,
        instruments_path: Optional[Path] = None,
        platforms_path: Optional[Path] = None,
    ):
        """
        Initialize the CMR enum validator.

        Args:
            threshold: Minimum similarity score for fuzzy matches (0.0-1.0)
            debug: Enable debug logging
            instruments_path: Path to CMR instruments JSON (optional)
            platforms_path: Path to CMR platforms JSON (optional)
        """
        self.threshold = threshold
        self.debug = debug

        # Default to utils/cmr_enums directory
        if instruments_path is None or platforms_path is None:
            utils_dir = Path(__file__).parent
            enums_dir = utils_dir / "cmr_enums"
            instruments_path = instruments_path or enums_dir / "cmr_instruments.json"
            platforms_path = platforms_path or enums_dir / "cmr_platforms.json"

        # Load CMR enumerations
        self.instruments, self.platforms = load_cmr_data(
            instruments_path,
            platforms_path,
        )

        if self.debug:
            logger.debug(
                f"Loaded {len(self.instruments)} instruments, {len(self.platforms)} platforms",
            )

    def validate_approach(self, approach) -> Tuple[any, Dict]:
        """
        Validate and correct instrument/platform values in a query approach.

        Args:
            approach: CMRQueryApproach object to validate

        Returns:
            Tuple of (corrected_approach, corrections_metadata)

        corrections_metadata format:
        {
            'corrections_applied': bool,
            'changes': [
                {'field': 'instrument', 'original': 'modis', 'corrected': 'MODIS', 'score': 0.95},
                ...
            ]
        }
        """
        # Extract current values
        original_instrument = approach.instrument
        original_platform = approach.platform

        # Skip validation if both are None/empty
        if not original_instrument and not original_platform:
            return approach, {"corrections_applied": False, "changes": []}

        # Perform fuzzy matching
        inst_match = None
        plat_match = None

        if original_instrument:
            inst_match = self._fuzzy_match_single("instrument", original_instrument)

        if original_platform:
            plat_match = self._fuzzy_match_single("platform", original_platform)

        # Apply corrections based on matches
        corrected_approach, metadata = self._apply_corrections(
            approach,
            original_instrument,
            original_platform,
            inst_match,
            plat_match,
        )

        # Log corrections if any were made
        if metadata["corrections_applied"] and self.debug:
            for change in metadata["changes"]:
                logger.info(
                    f"Corrected {change['field']}: '{change['original']}' → '{change['corrected']}' (score={change['score']:.2f})",
                )

        return corrected_approach, metadata

    def _fuzzy_match_single(
        self,
        field_type: str,
        value: str,
    ) -> Optional[Dict]:
        """
        Fuzzy match a single value against CMR enums.

        Args:
            field_type: 'instrument' or 'platform' (for context only)
            value: Value to match

        Returns:
            Best match dict {'name': str, 'kind': str, 'score': float} or None
        """
        if not value or not value.strip():
            return None

        # Use improved fuzzy matcher (searches both instruments and platforms)
        matches = rank_instrument_platform(
            value,
            self.instruments,
            self.platforms,
            k_each=1,  # Top 1 match per category
            min_score=self.threshold,
        )

        # Return top match if any found
        return matches[0] if matches else None

    def _apply_corrections(
        self,
        approach,
        original_instrument: Optional[str],
        original_platform: Optional[str],
        inst_match: Optional[Dict],
        plat_match: Optional[Dict],
    ) -> Tuple[any, Dict]:
        """
        Apply corrections based on fuzzy match results.

        Implements 5 correction scenarios from the integration plan:
        A. Simple Replacement
        B. Single Swap
        C. Double Swap
        D. Biased Swap (Different Scores)
        E. Biased Swap (Same Scores)

        Args:
            approach: Original CMRQueryApproach
            original_instrument: Original instrument value
            original_platform: Original platform value
            inst_match: Fuzzy match result for instrument field
            plat_match: Fuzzy match result for platform field

        Returns:
            Tuple of (corrected_approach, metadata)
        """
        changes = []
        new_instrument = original_instrument
        new_platform = original_platform

        # No corrections needed if no matches found
        if not inst_match and not plat_match:
            return approach, {"corrections_applied": False, "changes": []}

        # Extract match details
        inst_match_kind = inst_match["kind"] if inst_match else None
        inst_match_name = inst_match["name"] if inst_match else None
        inst_match_score = inst_match["score"] if inst_match else 0.0

        plat_match_kind = plat_match["kind"] if plat_match else None
        plat_match_name = plat_match["name"] if plat_match else None
        plat_match_score = plat_match["score"] if plat_match else 0.0

        # Scenario A: Simple Replacement (both match correct types)
        if (
            inst_match
            and inst_match_kind == "instrument"
            and plat_match
            and plat_match_kind == "platform"
        ):
            if original_instrument != inst_match_name:
                new_instrument = inst_match_name
                changes.append(
                    {
                        "field": "instrument",
                        "original": original_instrument,
                        "corrected": inst_match_name,
                        "score": inst_match_score,
                    },
                )

            if original_platform != plat_match_name:
                new_platform = plat_match_name
                changes.append(
                    {
                        "field": "platform",
                        "original": original_platform,
                        "corrected": plat_match_name,
                        "score": plat_match_score,
                    },
                )

        # Scenario C: Double Swap (both are reversed)
        elif (
            inst_match
            and inst_match_kind == "platform"
            and plat_match
            and plat_match_kind == "instrument"
        ):
            new_instrument = plat_match_name
            new_platform = inst_match_name

            changes.append(
                {
                    "field": "instrument",
                    "original": original_instrument,
                    "corrected": plat_match_name,
                    "score": plat_match_score,
                    "swapped": True,
                },
            )
            changes.append(
                {
                    "field": "platform",
                    "original": original_platform,
                    "corrected": inst_match_name,
                    "score": inst_match_score,
                    "swapped": True,
                },
            )

        # Scenario B: Single Swap (instrument field is actually platform)
        elif inst_match and inst_match_kind == "platform" and not plat_match:
            new_instrument = None
            new_platform = inst_match_name

            changes.append(
                {
                    "field": "platform",
                    "original": original_instrument,
                    "corrected": inst_match_name,
                    "score": inst_match_score,
                    "moved_from": "instrument",
                },
            )
            changes.append(
                {
                    "field": "instrument",
                    "original": original_instrument,
                    "corrected": None,
                    "score": 0.0,
                    "dropped": True,
                },
            )

        # Scenario B: Single Swap (platform field is actually instrument)
        elif plat_match and plat_match_kind == "instrument" and not inst_match:
            new_platform = None
            new_instrument = plat_match_name

            changes.append(
                {
                    "field": "instrument",
                    "original": original_platform,
                    "corrected": plat_match_name,
                    "score": plat_match_score,
                    "moved_from": "platform",
                },
            )
            changes.append(
                {
                    "field": "platform",
                    "original": original_platform,
                    "corrected": None,
                    "score": 0.0,
                    "dropped": True,
                },
            )

        # Scenario D/E: Biased Swap (both resolve to same type)
        elif inst_match and plat_match and inst_match_kind == plat_match_kind:
            # Both match to instruments
            if inst_match_kind == "instrument":
                # Compare scores
                if inst_match_score > plat_match_score:
                    # Keep instrument field match (higher score)
                    new_instrument = inst_match_name
                    new_platform = None
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": inst_match_name,
                            "score": inst_match_score,
                        },
                    )
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": None,
                            "score": plat_match_score,
                            "dropped": True,
                            "reason": "lower_score",
                        },
                    )
                elif plat_match_score > inst_match_score:
                    # Keep platform field match (higher score), move to instrument
                    new_instrument = plat_match_name
                    new_platform = None
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_platform,
                            "corrected": plat_match_name,
                            "score": plat_match_score,
                            "moved_from": "platform",
                        },
                    )
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": None,
                            "score": 0.0,
                            "dropped": True,
                        },
                    )
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": None,
                            "score": inst_match_score,
                            "dropped": True,
                            "reason": "lower_score",
                        },
                    )
                else:
                    # Scenario E: Same scores - keep the match from the original field
                    new_instrument = inst_match_name
                    new_platform = None
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": inst_match_name,
                            "score": inst_match_score,
                        },
                    )
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": None,
                            "score": plat_match_score,
                            "dropped": True,
                            "reason": "same_score_keep_original_field",
                        },
                    )

            # Both match to platforms
            else:  # plat_match_kind == "platform"
                # Compare scores
                if plat_match_score > inst_match_score:
                    # Keep platform field match (higher score)
                    new_platform = plat_match_name
                    new_instrument = None
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": plat_match_name,
                            "score": plat_match_score,
                        },
                    )
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": None,
                            "score": inst_match_score,
                            "dropped": True,
                            "reason": "lower_score",
                        },
                    )
                elif inst_match_score > plat_match_score:
                    # Keep instrument field match (higher score), move to platform
                    new_platform = inst_match_name
                    new_instrument = None
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_instrument,
                            "corrected": inst_match_name,
                            "score": inst_match_score,
                            "moved_from": "instrument",
                        },
                    )
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": None,
                            "score": 0.0,
                            "dropped": True,
                        },
                    )
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": None,
                            "score": plat_match_score,
                            "dropped": True,
                            "reason": "lower_score",
                        },
                    )
                else:
                    # Scenario E: Same scores - keep the match from the original field
                    new_platform = plat_match_name
                    new_instrument = None
                    changes.append(
                        {
                            "field": "platform",
                            "original": original_platform,
                            "corrected": plat_match_name,
                            "score": plat_match_score,
                        },
                    )
                    changes.append(
                        {
                            "field": "instrument",
                            "original": original_instrument,
                            "corrected": None,
                            "score": inst_match_score,
                            "dropped": True,
                            "reason": "same_score_keep_original_field",
                        },
                    )

        # Handle single field matches with simple replacement
        elif inst_match and inst_match_kind == "instrument" and not plat_match:
            if original_instrument != inst_match_name:
                new_instrument = inst_match_name
                changes.append(
                    {
                        "field": "instrument",
                        "original": original_instrument,
                        "corrected": inst_match_name,
                        "score": inst_match_score,
                    },
                )

        elif plat_match and plat_match_kind == "platform" and not inst_match:
            if original_platform != plat_match_name:
                new_platform = plat_match_name
                changes.append(
                    {
                        "field": "platform",
                        "original": original_platform,
                        "corrected": plat_match_name,
                        "score": plat_match_score,
                    },
                )

        # Create corrected approach
        if changes:
            # Create new approach with corrected values
            approach_dict = approach.model_dump()
            approach_dict["instrument"] = new_instrument
            approach_dict["platform"] = new_platform

            corrected_approach = approach.__class__(**approach_dict)

            return corrected_approach, {
                "corrections_applied": True,
                "changes": changes,
            }
        else:
            return approach, {"corrections_applied": False, "changes": []}
