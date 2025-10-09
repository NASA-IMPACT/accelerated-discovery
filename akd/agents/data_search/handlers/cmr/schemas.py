"""CMR-specific schemas for data search components."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents.data_search.components._base_parameters import (
    BaseKnownParametersInputSchema,
    BaseKnownParametersOutput,
    BaseSearchableParametersOutput,
)
from akd.agents.data_search.components._base_ranking import (
    BaseApproachFilteringInputSchema,
    BaseApproachFilteringOutput,
    BaseFinalRankingInputSchema,
    BaseFinalRankingOutput,
    FilteredRankedItem,
    FinalRankedItem,
)
from akd.agents.data_search.components.scientific_decomposition import (
    ScientificDecomposition,
)
from akd.agents.data_search.components.topic_splitting import Topic

# ============================================================================
# Known Parameters Schemas
# ============================================================================


class CMRQueryApproach(BaseModel):
    """Parameters for a single CMR query approach using known parameters only."""

    instrument: Optional[str] = Field(
        None,
        description="Single instrument name (e.g., MODIS, VIIRS)",
    )
    platform: Optional[str] = Field(
        None,
        description="Single platform/satellite name (e.g., Terra, Aqua)",
    )
    processing_level: Optional[str] = Field(
        None,
        description="Data processing level (e.g., Level 1B, Level 2)",
    )
    temporal: Optional[str] = Field(
        None,
        description="Temporal range in ISO format: YYYY-MM-DDTHH:mm:ssZ,YYYY-MM-DDTHH:mm:ssZ",
    )
    bounding_box: Optional[str] = Field(
        None,
        description="Spatial bounding box as comma-separated string: west,south,east,north",
    )
    temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution (e.g., daily, weekly, monthly)",
    )
    spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution (e.g., 250m, 1km, 30m)",
    )

    def get_mcp_parameters(self) -> dict[str, str]:
        """
        Get parameters that can be passed directly to CMR MCP API.

        Returns:
            Dictionary with only CMR-compatible search parameters
        """
        mcp_params = {}

        if self.instrument:
            mcp_params["instrument"] = self.instrument
        if self.platform:
            mcp_params["platform"] = self.platform
        if self.processing_level:
            mcp_params["processing_level"] = self.processing_level
        if self.temporal:
            mcp_params["temporal"] = self.temporal
        if self.bounding_box:
            mcp_params["bounding_box"] = self.bounding_box

        return mcp_params

    def get_filter_parameters(self) -> dict[str, str]:
        """
        Get parameters that must be used for post-search filtering.

        Returns:
            Dictionary with resolution and other filter-only parameters
        """
        filter_params = {}

        if self.temporal_resolution:
            filter_params["temporal_resolution"] = self.temporal_resolution
        if self.spatial_resolution:
            filter_params["spatial_resolution"] = self.spatial_resolution

        return filter_params


class CMRKnownParametersOutput(BaseKnownParametersOutput[CMRQueryApproach]):
    """CMR-specific output from known parameters component."""

    query_approaches: List[CMRQueryApproach] = Field(
        ...,
        description="List of CMR query approaches using known parameters (1-5 approaches)",
        min_items=1,
        max_items=5,
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these specific parameters were identified",
    )


class CMRKnownParametersInputSchema(BaseKnownParametersInputSchema):
    """CMR-specific input schema for known parameters extraction."""

    pass  # Uses base class fields


# ============================================================================
# Searchable Parameters Schemas
# ============================================================================


class CMRSearchableQuery(BaseModel):
    """Complete CMR query with known parameters + searchable parameters."""

    # Approach tracking
    approach_index: int = Field(
        ...,
        description="Index of source QueryApproach (0-based)",
    )

    # Known parameters (carried forward)
    instrument: Optional[str] = Field(None, description="Instrument name")
    platform: Optional[str] = Field(None, description="Platform/satellite name")
    processing_level: Optional[str] = Field(None, description="Data processing level")
    temporal: Optional[str] = Field(None, description="Temporal range in ISO format")
    bounding_box: Optional[str] = Field(None, description="Spatial bounding box")
    temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution",
    )
    spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution",
    )

    # Searchable parameters (generated)
    primary_keywords: List[str] = Field(
        default_factory=list,
        description="Primary search keywords for CMR metadata search",
        max_items=4,
    )
    alternative_keywords: List[str] = Field(
        default_factory=list,
        description="Alternative/synonym keywords",
        max_items=3,
    )
    combined_keyword_string: str = Field(
        default="",
        description="Combined keyword string for CMR search (empty string if no keywords needed)",
    )

    def get_mcp_parameters(self) -> dict[str, str]:
        """
        Get parameters that can be passed directly to CMR MCP API.

        Returns:
            Dictionary with only CMR-compatible search parameters
        """
        mcp_params = {}

        if self.instrument:
            mcp_params["instrument"] = self.instrument
        if self.platform:
            mcp_params["platform"] = self.platform
        if self.processing_level:
            mcp_params["processing_level"] = self.processing_level
        if self.temporal:
            mcp_params["temporal"] = self.temporal
        if self.bounding_box:
            mcp_params["bounding_box"] = self.bounding_box

        # Add searchable parameters only if they exist
        if self.combined_keyword_string and self.combined_keyword_string.strip():
            mcp_params["keyword"] = self.combined_keyword_string

        return mcp_params

    def get_filter_parameters(self) -> dict[str, str]:
        """
        Get parameters that must be used for post-search filtering.

        Returns:
            Dictionary with resolution and other filter-only parameters
        """
        filter_params = {}

        if self.temporal_resolution:
            filter_params["temporal_resolution"] = self.temporal_resolution
        if self.spatial_resolution:
            filter_params["spatial_resolution"] = self.spatial_resolution

        return filter_params


class CMRSearchableParametersOutput(BaseSearchableParametersOutput[CMRSearchableQuery]):
    """CMR-specific output from searchable parameters component."""

    searchable_queries: List[CMRSearchableQuery] = Field(
        ...,
        description="Complete CMR queries with known + searchable parameters (expanded from approaches)",
        min_items=1,
        max_items=25,  # Up to 5 approaches × 5 variations each
    )
    keyword_strategy: str = Field(
        ...,
        description="Explanation of keyword selection strategy",
    )


class CMRSearchableParametersInputSchema(InputSchema):
    """CMR-specific input schema for searchable parameters generation."""

    original_query: str = Field(..., description="Original research question")
    topic: Topic = Field(..., description="Topic being processed")
    decomposition: ScientificDecomposition = Field(
        ...,
        description="Scientific decomposition",
    )
    query_approaches: List[CMRQueryApproach] = Field(
        ...,
        description="Known parameter approaches",
    )


# ============================================================================
# Approach Filtering Schemas
# ============================================================================


class CMRApproachCollectionFilteringInputSchema(BaseApproachFilteringInputSchema):
    """CMR-specific input schema for filtering collections within a single approach."""

    # CMR-specific approach context
    approach_instrument: Optional[str] = Field(
        None,
        description="Instrument for this CMR query approach",
    )
    approach_platform: Optional[str] = Field(
        None,
        description="Platform for this CMR query approach",
    )
    approach_processing_level: Optional[str] = Field(
        None,
        description="Processing level for this CMR query approach",
    )
    approach_temporal_range: Optional[str] = Field(
        None,
        description="Temporal range for this CMR query approach",
    )
    approach_spatial_bounds: Optional[str] = Field(
        None,
        description="Spatial bounds for this CMR query approach",
    )
    approach_temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution for this CMR query approach",
    )
    approach_spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution for this CMR query approach",
    )
    approach_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords used in this CMR query approach",
    )

    # Override to use CMR-specific naming
    collections: List[Dict[str, Any]] = Field(
        ...,
        max_items=25,
        description="CMR collections to filter (from this approach)",
    )
    max_collections: int = Field(
        default=5,
        description="Maximum collections to select from this approach",
    )

    # Alias base field to our CMR-specific field
    @property
    def data_items(self) -> List[Dict[str, Any]]:
        """Alias for base class compatibility."""
        return self.collections

    @property
    def max_items(self) -> int:
        """Alias for base class compatibility."""
        return self.max_collections


class CMRFilteredRankedCollection(FilteredRankedItem):
    """CMR collection that passed filtering, with rank within approach."""

    collection_index: int = Field(
        ...,
        description="Index in the input collections list (0-based)",
    )

    # Alias base field
    @property
    def item_index(self) -> int:
        """Alias for base class compatibility."""
        return self.collection_index


class CMRApproachCollectionFilteringOutput(BaseApproachFilteringOutput):
    """CMR-specific output from per-approach filtering."""

    selected_collections: List[CMRFilteredRankedCollection] = Field(
        ...,
        max_items=5,
        description="Top CMR collections for this approach, ranked",
    )

    # Alias base field
    @property
    def selected_items(self) -> List[FilteredRankedItem]:
        """Alias for base class compatibility."""
        return self.selected_collections


# ============================================================================
# Final Ranking Schemas
# ============================================================================


class CMRFinalCollectionRankingInputSchema(BaseFinalRankingInputSchema):
    """CMR-specific input schema for final cross-approach ranking."""

    # Override to use CMR-specific naming
    collections: List[Dict[str, Any]] = Field(
        ...,
        max_items=25,
        description="Pre-filtered CMR collections from all approaches",
    )
    max_collections: int = Field(
        default=25,
        description="Maximum collections to return (ranked 1-N)",
    )

    # Alias base field
    @property
    def data_items(self) -> List[Dict[str, Any]]:
        """Alias for base class compatibility."""
        return self.collections

    @property
    def max_items(self) -> int:
        """Alias for base class compatibility."""
        return self.max_collections


class CMRFinalRankedCollection(FinalRankedItem):
    """CMR collection with final rank across all approaches."""

    collection_index: int = Field(
        ...,
        description="Index in the input collections list (0-based)",
    )

    # Alias base field
    @property
    def item_index(self) -> int:
        """Alias for base class compatibility."""
        return self.collection_index


class CMRFinalCollectionRankingOutput(BaseFinalRankingOutput):
    """CMR-specific output from final ranking."""

    ranked_collections: List[CMRFinalRankedCollection] = Field(
        ...,
        max_items=25,
        description="CMR collections ranked 1-25 (or fewer if less available)",
    )
    total_collections_ranked: int = Field(
        ...,
        description="Total number of collections in the ranking",
    )

    # Alias base field
    @property
    def ranked_items(self) -> List[FinalRankedItem]:
        """Alias for base class compatibility."""
        return self.ranked_collections

    @property
    def total_items_ranked(self) -> int:
        """Alias for base class compatibility."""
        return self.total_collections_ranked
