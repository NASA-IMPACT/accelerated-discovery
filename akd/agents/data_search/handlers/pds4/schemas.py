"""
PDS4 Handler Schemas

Defines input/output schemas for PDS4 handler components following
the unified parameter extraction pattern.

Unlike CMR's two-stage approach (Known Parameters → Searchable Parameters),
PDS4 uses unified parameter extraction where complete tool execution strategies
are generated in a single component call.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from akd._base import InputSchema
from akd.agents.data_search.components._base_ranking import (
    BaseApproachFilteringInputSchema,
    BaseApproachFilteringOutput,
    BaseFinalRankingInputSchema,
    BaseFinalRankingOutput,
)


# ============================================================================
# Enumerations (from PDS4 MCP Resources)
# ============================================================================

class PDS4TargetType(str, Enum):
    """PDS4 target types from resource://target_type"""
    PLANETARY_NEBULA = "Planetary Nebula"
    GALAXY = "Galaxy"
    CALIBRATOR = "Calibrator"
    TRANS_NEPTUNIAN_OBJECT = "Trans-Neptunian Object"
    PLANETARY_SYSTEM = "Planetary System"
    SATELLITE = "Satellite"
    CENTAUR = "Centaur"
    ASTROPHYSICAL = "Astrophysical"
    STAR_CLUSTER = "Star Cluster"
    LABORATORY_ANALOG = "Laboratory Analog"
    DUST = "Dust"
    ASTEROID = "Asteroid"
    COMET = "Comet"
    EQUIPMENT = "Equipment"
    STAR = "Star"
    RING = "Ring"
    DWARF_PLANET = "Dwarf Planet"
    CALIBRATION_FIELD = "Calibration Field"
    PLANET = "Planet"
    PLASMA_CLOUD = "Plasma Cloud"
    PLASMA_STREAM = "Plasma Stream"
    MAGNETIC_FIELD = "Magnetic Field"
    SUN = "Sun"


class PDS4InstrumentType(str, Enum):
    """PDS4 instrument types from resource://instrument_type"""
    ENERGETIC_PARTICLE_DETECTOR = "Energetic Particle Detector"
    PLASMA_ANALYZER = "Plasma Analyzer"
    REGOLITH_PROPERTIES = "Regolith Properties"
    SPECTROGRAPH = "Spectrograph"
    IMAGER = "Imager"
    ATMOSPHERIC_SCIENCES = "Atmospheric Sciences"
    SPECTROMETER = "Spectrometer"
    RADIO_RADAR = "Radio-Radar"
    ULTRAVIOLET_SPECTROMETER = "Ultraviolet Spectrometer"
    SMALL_BODIES_SCIENCES = "Small Bodies Sciences"
    DUST = "Dust"
    PARTICLE_DETECTOR = "Particle Detector"
    PHOTOMETER = "Photometer"
    POLARIMETER = "Polarimeter"
    PLASMA_WAVE_SPECTROMETER = "Plasma Wave Spectrometer"


class PDS4InstrumentHostType(str, Enum):
    """PDS4 instrument host types from resource://instrument_host_type"""
    ROVER = "Rover"
    LANDER = "Lander"
    SPACECRAFT = "Spacecraft"
    EARTH_BASED = "Earth-based"
    OBSERVATORY = "Observatory"
    INSTRUMENT_HOST = "Instrument_host"
    UNK = "Unk"


# ============================================================================
# Query Approach Schema (Core of Unified Approach)
# ============================================================================

class PDS4QueryApproach(BaseModel):
    """
    Complete PDS4 MCP tool execution approach with unified parameters.

    Represents a unified approach combining context parameters and
    search keywords. Unlike CMR's two-stage approach, this contains
    everything needed for execution.

    Example:
        PDS4QueryApproach(
            approach_index=0,
            approach_description="Investigation-first: Search for Mars rover missions",
            investigation_keywords=["mars rover", "curiosity", "msl"],
            target_keywords=["mars"],
            instrument_keywords=["spectrometer", "chemcam"],
        )
    """

    approach_index: int = Field(
        ...,
        description="0-based index of this approach (0-3)"
    )

    approach_description: str = Field(
        ...,
        description="Human-readable description of this approach"
    )

    # ---- Context Parameters (PDS4-specific) ----
    # These are keywords used in PDS4 MCP context searches

    investigation_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for investigation/mission search (e.g., ['mars odyssey', 'curiosity'])"
    )

    target_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for target search (e.g., ['mars', 'europa', 'moon'])"
    )

    instrument_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for instrument search (e.g., ['spectrometer', 'chemcam'])"
    )

    # ---- Search Parameters ----

    primary_keywords: List[str] = Field(
        default_factory=list,
        description="Primary search terms for bundle/collection queries"
    )

    temporal_context: Optional[str] = Field(
        None,
        description="Descriptive temporal period (e.g., '2012-2020', 'Apollo era')"
    )

    # ---- Execution Parameters ----

    investigation_search_limit: int = Field(
        10,
        description="Max investigations to retrieve in context discovery"
    )

    target_search_limit: int = Field(
        10,
        description="Max targets to retrieve in context discovery"
    )

    instrument_search_limit: int = Field(
        10,
        description="Max instruments to retrieve in context discovery"
    )

    collection_search_limit: int = Field(
        20,
        description="Max collections to retrieve"
    )

    bundle_search_limit: int = Field(
        10,
        description="Max bundles to retrieve"
    )

    # ---- Extracted URNs (populated during execution) ----

    # Complete URN lists (top 3 of each, used in combinations)
    investigation_urns: List[str] = Field(
        default_factory=list,
        description="All investigation URNs extracted and ranked (top 3, used in collection search combinations)"
    )

    target_urns: List[str] = Field(
        default_factory=list,
        description="All target URNs extracted and ranked (top 3, used in collection search combinations)"
    )

    instrument_urns: List[str] = Field(
        default_factory=list,
        description="All instrument URNs extracted and ranked (top 3, used in collection search combinations)"
    )

    # Legacy single URN fields (DEPRECATED - kept for backward compatibility, excluded from serialization)
    investigation_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top investigation URN - use investigation_urns[0] instead"
    )

    target_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top target URN - use target_urns[0] instead"
    )

    instrument_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top instrument URN - use instrument_urns[0] instead"
    )

    def get_context_search_params(self) -> Dict[str, Any]:
        """
        Get parameters for context discovery tools.

        Returns:
            Dict with keys: investigation_params, target_params, instrument_params
        """
        params = {}

        if self.investigation_keywords:
            params["investigation_params"] = {
                "keywords": " ".join(self.investigation_keywords),
                "limit": self.investigation_search_limit
            }

        if self.target_keywords:
            params["target_params"] = {
                "keywords": " ".join(self.target_keywords),
                "limit": self.target_search_limit
            }

        if self.instrument_keywords:
            params["instrument_params"] = {
                "keywords": " ".join(self.instrument_keywords),
                "limit": self.instrument_search_limit
            }

        return params

    def get_collection_search_params(
        self,
        investigation_urn: Optional[str] = None,
        target_urn: Optional[str] = None,
        instrument_urn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get parameters for collection search using extracted URNs."""
        return {
            "ref_lid_investigation": investigation_urn or "",
            "ref_lid_target": target_urn or "",
            "ref_lid_instrument": instrument_urn or "",
            "limit": self.collection_search_limit
        }

    def get_bundle_search_params(self) -> Dict[str, Any]:
        """Get parameters for supplementary bundle search."""
        query = " ".join(self.primary_keywords) if self.primary_keywords else ""
        return {
            "title_query": query,
            "limit": self.bundle_search_limit
        }


# ============================================================================
# Parameter Extraction Component Schemas
# ============================================================================

class PDS4ParameterExtractionInputSchema(InputSchema):
    """Input for unified parameter extraction component."""

    original_query: str = Field(..., description="Original user query")
    topic: str = Field(..., description="Topic from topic splitter")
    decomposition: str = Field(..., description="Scientific decomposition text")


class PDS4ParameterExtractionOutput(BaseModel):
    """Output from unified parameter extraction component."""

    query_approaches: List[PDS4QueryApproach] = Field(
        ...,
        description="1-4 complete tool execution approaches",
        min_length=1,
        max_length=4
    )

    reasoning: str = Field(
        ...,
        description="Explanation of approach generation decisions"
    )


# ============================================================================
# Approach Filtering Component Schemas
# ============================================================================

class PDS4ApproachCollectionFilteringInputSchema(BaseApproachFilteringInputSchema):
    """PDS4-specific input schema for filtering collections within a single approach."""

    # Approach-specific parameters
    strategy_description: str = Field(
        ...,
        description="Description of the tool approach that generated these collections",
    )

    # Keywords used in context searches
    investigation_keywords: List[str] = Field(
        default_factory=list,
        description="Investigation/mission keywords used in context search",
    )
    target_keywords: List[str] = Field(
        default_factory=list,
        description="Target keywords used in context search",
    )
    instrument_keywords: List[str] = Field(
        default_factory=list,
        description="Instrument keywords used in context search",
    )
    temporal_context: Optional[str] = Field(
        None,
        description="Temporal context/period for the search",
    )

    # URNs extracted from context searches (COMPLETE LISTS used in combinations)
    investigation_urns: List[str] = Field(
        default_factory=list,
        description="All investigation URNs used in collection searches (top 3, used in combinations)",
    )
    target_urns: List[str] = Field(
        default_factory=list,
        description="All target URNs used in collection searches (top 3, used in combinations)",
    )
    instrument_urns: List[str] = Field(
        default_factory=list,
        description="All instrument URNs used in collection searches (top 3, used in combinations)",
    )

    # Legacy single URN fields (DEPRECATED - use plural fields above)
    # These will be removed from serialization
    investigation_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top investigation URN - use investigation_urns instead",
    )
    target_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top target URN - use target_urns instead",
    )
    instrument_urn: Optional[str] = Field(
        None,
        description="[DEPRECATED] Top instrument URN - use instrument_urns instead",
    )

    # Use base class fields (data_items, max_items) and provide PDS4-specific aliases
    @property
    def collections(self) -> List[Dict[str, Any]]:
        """Alias for PDS4-specific naming."""
        return self.data_items


class PDS4ApproachCollectionFilteringOutput(BaseApproachFilteringOutput):
    """PDS4-specific output schema for per-approach filtering.

    Inherits selected_item_indexes and reasoning from base class.
    """
    pass


# ============================================================================
# Final Ranking Component Schemas
# ============================================================================

class PDS4FinalCollectionRankingInputSchema(BaseFinalRankingInputSchema):
    """PDS4-specific input schema for final cross-strategy ranking.

    Inherits core fields from BaseFinalRankingInputSchema:
    - original_query, topic_title, topic_context
    - decomposition_title, decomposition_justification
    - data_items (collections), max_items
    """
    # No additional PDS4-specific fields needed for final ranking
    pass


class PDS4FinalCollectionRankingOutput(BaseFinalRankingOutput):
    """PDS4-specific output schema for final ranking.

    Inherits ranked_item_indexes and reasoning from base class.
    """
    pass


# ============================================================================
# Internal Execution Schemas
# ============================================================================

class URNExtractionResult(BaseModel):
    """Result of URN extraction from context search."""

    urn: str = Field(..., description="Extracted URN identifier")
    title: str = Field(..., description="Product title")
    score: float = Field(..., description="Relevance score (0-1)")
    source_tool: str = Field(..., description="Tool that returned this URN")


class ToolExecutionResult(BaseModel):
    """Result from a single MCP tool execution."""

    tool_name: str
    params: Dict[str, Any]
    total_hits: int
    results: List[Dict[str, Any]]
    execution_time_ms: float
    error: Optional[str] = None
