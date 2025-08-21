"""
Base classes and shared utilities for data search agents.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from akd.agents._base import BaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema


class DataSearchAgentInputSchema(InputSchema):
    """Base input schema for data search agents."""

    query: str = Field(..., description="Natural language query for data discovery")
    temporal_range: Optional[str] = Field(
        None, 
        description="Optional temporal constraint (e.g., '2023-01-01,2023-12-31')"
    )
    spatial_bounds: Optional[str] = Field(
        None,
        description="Optional spatial constraint as 'west,south,east,north'"
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of data files/granules to return",
    )
    processing_level: Optional[str] = Field(
        None,
        description="Optional data processing level filter (L0, L1A, L1B, L2, L3, L4)"
    )


class DataSearchResult(BaseModel):
    """Individual data search result (granule/file)."""
    
    concept_id: str = Field(..., description="Unique identifier for the data file")
    title: str = Field(..., description="Human-readable title")
    download_urls: List[dict] = Field(
        default_factory=list,
        description="List of download URL objects with metadata"
    )
    temporal_extent: Optional[dict] = Field(
        None,
        description="Temporal coverage information"
    )
    spatial_extent: Optional[dict] = Field(
        None, 
        description="Spatial coverage information"
    )
    file_size_mb: Optional[float] = Field(
        None,
        description="File size in megabytes"
    )
    online_access: bool = Field(
        default=False,
        description="Whether file is immediately accessible online"
    )
    collection_info: Optional[dict] = Field(
        None,
        description="Parent collection metadata"
    )


class DataSearchAgentOutputSchema(OutputSchema):
    """Base output schema for data search agents."""

    granules: List[dict] = Field(..., description="List of discovered data files/granules")
    search_metadata: dict = Field(..., description="Search provenance and metadata")
    total_results: int = Field(..., description="Total number of results found")
    collections_searched: List[dict] = Field(
        default_factory=list,
        description="Collections that were searched"
    )


class DataSearchAgentConfig(BaseAgentConfig):
    """Base configuration for data search agents."""

    debug: bool = Field(default=False, description="Enable debug logging")
    max_collections_to_search: int = Field(
        default=10,
        description="Maximum number of collections to search for granules"
    )
    max_granules_per_collection: int = Field(
        default=100,
        description="Maximum granules to retrieve per collection"
    )
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel collection/granule searches"
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )


class BaseDataSearchAgent[
    TInput: DataSearchAgentInputSchema, 
    TOutput: DataSearchAgentOutputSchema
](BaseAgent[TInput, TOutput]):
    """
    Abstract base class for data search agents.
    
    Provides common functionality for all data search agents including:
    - Standard input/output schemas
    - Common configuration handling  
    - Shared utility methods for data discovery workflows
    - Consistent error handling and logging patterns
    """

    input_schema = DataSearchAgentInputSchema
    output_schema = DataSearchAgentOutputSchema
    config_schema = DataSearchAgentConfig

    def _validate_query(self, query: str) -> str:
        """Validate and clean the input query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()

    def _parse_temporal_range(self, temporal_range: Optional[str]) -> Optional[dict]:
        """Parse temporal range string into structured format."""
        if not temporal_range:
            return None
            
        try:
            if "," in temporal_range:
                start, end = temporal_range.split(",", 1)
                return {
                    "start": start.strip(),
                    "end": end.strip()
                }
            else:
                # Single date - assume point in time
                return {
                    "start": temporal_range.strip(),
                    "end": temporal_range.strip()
                }
        except Exception:
            # Return None for invalid formats - let downstream handle
            return None

    def _parse_spatial_bounds(self, spatial_bounds: Optional[str]) -> Optional[dict]:
        """Parse spatial bounds string into structured format."""
        if not spatial_bounds:
            return None
            
        try:
            coords = [float(x.strip()) for x in spatial_bounds.split(",")]
            if len(coords) == 4:
                return {
                    "west": coords[0],
                    "south": coords[1], 
                    "east": coords[2],
                    "north": coords[3]
                }
        except Exception:
            # Return None for invalid formats - let downstream handle
            pass
        return None

    def _format_search_summary(
        self,
        total_granules: int,
        total_collections: int,
        search_duration_ms: float = 0.0,
    ) -> str:
        """Format a standardized search summary."""
        return (
            f"Data search completed: {total_granules} granules found "
            f"across {total_collections} collections "
            f"(search time: {search_duration_ms:.0f}ms)"
        )

    async def get_response_async(
        self,
        *args,
        **kwargs,
    ) -> TOutput:
        """
        Obtains a response from the data search agent asynchronously.
        
        This is a placeholder - data search agents typically don't use
        LLM generation directly, but rather orchestrate tool calls.
        """
        raise NotImplementedError(
            "Data search agents should implement _arun method directly, "
            "not get_response_async"
        )