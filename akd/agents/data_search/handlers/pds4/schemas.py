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
# Tool Strategy Schema (Core of Unified Approach)
# ============================================================================

class PDS4ToolStrategy(BaseModel):
    """
    Complete PDS4 MCP tool execution strategy with unified parameters.

    Represents a unified approach combining context parameters,
    search keywords, and tool orchestration logic. Unlike CMR's
    two-stage approach, this contains everything needed for execution.

    Example:
        PDS4ToolStrategy(
            strategy_index=0,
            strategy_description="Investigation-first: Search for Mars rover missions",
            mission_keywords=["mars rover", "curiosity", "msl"],
            target_context="Mars",
            target_type=PDS4TargetType.PLANET,
            instrument_keywords=["spectrometer", "chemcam"],
            tool_sequence=["search_investigations", "search_targets", "search_collections"]
        )
    """

    strategy_index: int = Field(
        ...,
        description="0-based index of this strategy (0-3)"
    )

    strategy_description: str = Field(
        ...,
        description="Human-readable description of this strategy's approach"
    )

    # ---- Context Parameters (PDS4-specific) ----

    target_context: Optional[str] = Field(
        None,
        description="Target celestial body (e.g., 'Mars', 'Europa', 'Moon')"
    )

    target_type: Optional[PDS4TargetType] = Field(
        None,
        description="Target classification for filtering"
    )

    mission_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for investigation search (e.g., ['mars rover', 'curiosity'])"
    )

    instrument_keywords: List[str] = Field(
        default_factory=list,
        description="Instrument types or names (e.g., ['spectrometer', 'chemcam'])"
    )

    platform_keywords: List[str] = Field(
        default_factory=list,
        description="Instrument host keywords (e.g., ['rover', 'orbiter'])"
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

    # ---- Tool Orchestration ----

    tool_sequence: List[str] = Field(
        default_factory=list,
        description="Ordered list of MCP tools to execute (e.g., ['search_investigations', 'search_collections'])"
    )

    expected_urn_types: List[str] = Field(
        default_factory=list,
        description="URN types to extract from context searches (e.g., ['investigation', 'target'])"
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

    collection_search_limit: int = Field(
        20,
        description="Max collections to retrieve"
    )

    bundle_search_limit: int = Field(
        10,
        description="Max bundles to retrieve"
    )

    def get_context_search_params(self) -> Dict[str, Any]:
        """
        Get parameters for context discovery tools.

        Returns:
            Dict with keys: investigation_params, target_params
        """
        params = {}

        if self.mission_keywords:
            params["investigation_params"] = {
                "keywords": " ".join(self.mission_keywords),
                "limit": self.investigation_search_limit
            }

        if self.target_context or self.target_type:
            params["target_params"] = {
                "keywords": self.target_context or "",
                "target_type": self.target_type.value if self.target_type else "",
                "limit": self.target_search_limit
            }

        return params

    def get_collection_search_params(
        self,
        investigation_urn: Optional[str] = None,
        target_urn: Optional[str] = None,
        instrument_urn: Optional[str] = None,
        instrument_host_urn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get parameters for collection search using extracted URNs."""
        return {
            "ref_lid_investigation": investigation_urn or "",
            "ref_lid_target": target_urn or "",
            "ref_lid_instrument": instrument_urn or "",
            "ref_lid_instrument_host": instrument_host_urn or "",
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

    tool_strategies: List[PDS4ToolStrategy] = Field(
        ...,
        description="1-4 complete tool execution strategies",
        min_length=1,
        max_length=4
    )

    reasoning: str = Field(
        ...,
        description="Explanation of strategy generation decisions"
    )


# ============================================================================
# Strategy Filtering Component Schemas
# ============================================================================

class PDS4StrategyCollectionFilteringInputSchema(BaseApproachFilteringInputSchema):
    """PDS4-specific input schema for filtering collections within a single strategy."""

    # Strategy-specific parameters
    strategy_description: str = Field(
        ...,
        description="Description of the tool strategy that generated these collections",
    )
    target_context: Optional[str] = Field(
        None,
        description="Target context (mission/investigation) used in strategy",
    )
    target_type: Optional[str] = Field(
        None,
        description="Specific target type filter used",
    )
    mission_keywords: List[str] = Field(
        default_factory=list,
        description="Mission/investigation keywords used in context search",
    )
    investigation_urn: Optional[str] = Field(
        None,
        description="Investigation URN extracted from context search",
    )
    target_urn: Optional[str] = Field(
        None,
        description="Target URN extracted from context search",
    )

    # Use base class fields (data_items, max_items) and provide PDS4-specific aliases
    @property
    def collections(self) -> List[Dict[str, Any]]:
        """Alias for PDS4-specific naming."""
        return self.data_items


class PDS4StrategyCollectionFilteringOutput(BaseApproachFilteringOutput):
    """PDS4-specific output schema for per-strategy filtering.

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
