"""CMR-specific schemas for data search components."""

import warnings
from typing import Any, ClassVar, Dict, List, Optional

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
    FinalRankedItem,
)
from akd.agents.data_search.components._shared_parameters import SearchVariations
from akd.agents.data_search.components.scientific_decomposition import (
    ScientificDecomposition,
)
from akd.agents.data_search.components.topic_splitting import Topic

# Suppress Pydantic warning about property objects not being JSON serializable
# These are runtime-only aliases and don't need to be in the JSON schema
warnings.filterwarnings(
    "ignore",
    message="Default value .* is not JSON serializable",
    category=UserWarning,
)

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

        Note: processing_level is excluded because CMR's processing_level filter
        is too restrictive and filters out valid results. Use get_filter_parameters()
        to access processing_level for post-search filtering instead.

        Returns:
            Dictionary with only CMR-compatible search parameters
        """
        mcp_params = {}

        if self.instrument:
            mcp_params["instrument"] = self.instrument
        if self.platform:
            mcp_params["platform"] = self.platform
        # processing_level intentionally excluded - used in filtering instead
        if self.temporal:
            mcp_params["temporal"] = self.temporal
        if self.bounding_box:
            mcp_params["bounding_box"] = self.bounding_box

        return mcp_params

    def get_filter_parameters(self) -> dict[str, str]:
        """
        Get parameters that must be used for post-search filtering.

        Returns:
            Dictionary with resolution, processing level, and other filter-only parameters
        """
        filter_params = {}

        if self.processing_level:
            filter_params["processing_level"] = self.processing_level
        if self.temporal_resolution:
            filter_params["temporal_resolution"] = self.temporal_resolution
        if self.spatial_resolution:
            filter_params["spatial_resolution"] = self.spatial_resolution

        return filter_params


class CMRKnownParametersOutput(BaseKnownParametersOutput[CMRQueryApproach]):
    """CMR-specific output from known parameters component."""

    query_approaches: List[CMRQueryApproach] = Field(
        ...,
        description="List of CMR query approaches using known parameters (1-4 approaches, only create what's needed)",
        min_items=1,
        max_items=4,
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
    search_string: Optional[str] = Field(
        None,
        description="Optional search string to narrow results when approach parameters are too broad (CMR treats as AND - all words must match)",
    )

    # Enum validation metadata (added during augmentation)
    enum_corrections: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata about instrument/platform fuzzy matching corrections applied to this query's approach",
    )

    def get_mcp_parameters(self) -> dict[str, str]:
        """
        Get parameters that can be passed directly to CMR MCP API.

        Note: processing_level is excluded because CMR's processing_level filter
        is too restrictive and filters out valid results. Use get_filter_parameters()
        to access processing_level for post-search filtering instead.

        Returns:
            Dictionary with only CMR-compatible search parameters
        """
        mcp_params = {}

        if self.instrument:
            mcp_params["instrument"] = self.instrument
        if self.platform:
            mcp_params["platform"] = self.platform
        # processing_level intentionally excluded - used in filtering instead
        if self.temporal:
            mcp_params["temporal"] = self.temporal
        if self.bounding_box:
            mcp_params["bounding_box"] = self.bounding_box

        # Add search string only if it exists
        if self.search_string and self.search_string.strip():
            mcp_params["keyword"] = self.search_string

        return mcp_params

    def get_filter_parameters(self) -> dict[str, str]:
        """
        Get parameters that must be used for post-search filtering.

        Returns:
            Dictionary with resolution, processing level, and other filter-only parameters
        """
        filter_params = {}

        if self.processing_level:
            filter_params["processing_level"] = self.processing_level
        if self.temporal_resolution:
            filter_params["temporal_resolution"] = self.temporal_resolution
        if self.spatial_resolution:
            filter_params["spatial_resolution"] = self.spatial_resolution

        return filter_params


class CMRSearchableParametersOutput(BaseSearchableParametersOutput[CMRSearchableQuery]):
    """CMR-specific output from searchable parameters component."""

    # Calculate max searchable queries from actual schema constraints
    # metadata[0] = MinLen, metadata[1] = MaxLen
    MAX_SEARCHABLE_QUERIES: ClassVar[int] = (
        CMRKnownParametersOutput.model_fields["query_approaches"].metadata[1].max_length
        * SearchVariations.model_fields["search_strings"].metadata[1].max_length
    )

    searchable_queries: List[CMRSearchableQuery] = Field(
        ...,
        description="Complete CMR queries with known + searchable parameters (expanded from approaches)",
        min_items=1,
        max_items=MAX_SEARCHABLE_QUERIES,
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

    # Approach object (contains all known parameters)
    approach: CMRQueryApproach = Field(
        ...,
        description="The CMR query approach that generated these collections",
    )

    # Search string kept separate (may differ from approach due to query variations)
    approach_search_string: Optional[str] = Field(
        None,
        description="Search string used in the searchable queries for this approach",
    )

    # Use base class fields (data_items, max_items) and provide CMR-specific aliases
    @property
    def collections(self) -> List[Dict[str, Any]]:
        """Alias for CMR-specific naming."""
        return self.data_items

    @property
    def max_collections(self) -> int:
        """Alias for CMR-specific naming."""
        return self.max_items


class CMRApproachCollectionFilteringOutput(BaseApproachFilteringOutput):
    """CMR-specific output from per-approach filtering."""

    # Use base class fields - no CMR-specific overrides needed
    pass


# ============================================================================
# Final Ranking Schemas
# ============================================================================


class CMRFinalCollectionRankingInputSchema(BaseFinalRankingInputSchema):
    """CMR-specific input schema for final cross-approach ranking."""

    # Use base class fields (data_items, max_items) and provide CMR-specific aliases
    @property
    def collections(self) -> List[Dict[str, Any]]:
        """Alias for CMR-specific naming."""
        return self.data_items

    @property
    def max_collections(self) -> int:
        """Alias for CMR-specific naming."""
        return self.max_items


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

    # Use base class fields - no CMR-specific overrides needed
    pass
