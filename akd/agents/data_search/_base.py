"""
Base classes and shared utilities for data search agents.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig


class DataSearchAgentInputSchema(InputSchema):
    """Base input schema for data search agents."""

    query: str = Field(..., description="Natural language query for data discovery")
    temporal_range: Optional[str] = Field(
        None,
        description="Optional temporal constraint (e.g., '2023-01-01,2023-12-31')",
    )
    spatial_bounds: Optional[str] = Field(
        None,
        description="Optional spatial constraint as 'west,south,east,north'",
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of data files/granules to return",
    )


class DataSearchResult(BaseModel):
    """Individual data search result (granule/file)."""

    concept_id: str = Field(..., description="Unique identifier for the data file")
    title: str = Field(..., description="Human-readable title")
    download_urls: List[dict] = Field(
        default_factory=list,
        description="List of download URL objects with metadata",
    )
    temporal_extent: Optional[dict] = Field(
        None,
        description="Temporal coverage information",
    )
    spatial_extent: Optional[dict] = Field(
        None,
        description="Spatial coverage information",
    )
    file_size_mb: Optional[float] = Field(
        None,
        description="File size in megabytes",
    )
    online_access: bool = Field(
        default=False,
        description="Whether file is immediately accessible online",
    )
    collection_info: Optional[dict] = Field(
        None,
        description="Parent collection metadata",
    )


# New Workflow Schemas
class TopicResult(BaseModel):
    """Complete search result for a single topic."""

    topic: Dict[str, Any] = Field(
        ...,
        description="The topic that was processed",
    )
    decomposition_results: List["DecompositionResult"] = Field(
        default_factory=list,
        description="Results for each scientific decomposition of this topic",
    )
    total_cmr_results: int = Field(
        default=0,
        description="Sum of total_results_from_cmr across all decompositions",
    )
    total_filtered_results: int = Field(
        default=0,
        description="Sum of total_results_after_filtering across all decompositions",
    )
    note: Optional[str] = Field(
        None,
        description="Note about data availability or alternative sources",
    )


class DecompositionResult(BaseModel):
    """Search result for a single scientific decomposition."""

    decomposition: Dict[str, Any] = Field(
        ...,
        description="The scientific decomposition that was processed",
    )
    repository: Optional[str] = Field(
        None,
        description="Repository used (CMR, PDS4, etc.) or external source name",
    )
    query_approaches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Repository-specific query approaches generated",
    )
    searchable_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete queries with known + searchable parameters (repository-specific)",
    )
    data_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Data results from repository (collections for CMR, bundles for PDS4, etc.)",
    )
    total_results_from_cmr: int = Field(
        default=0,
        description="Total collections returned by CMR across all queries (before dedup/filtering)",
    )
    total_results_after_filtering: int = Field(
        default=0,
        description="Total collections after dedup and filtering (len of data_results)",
    )
    note: Optional[str] = Field(
        None,
        description="Additional information (external sources, errors, processing notes)",
    )


class DataSearchAgentOutputSchema(OutputSchema):
    """Base output schema for data search agents."""

    topics: List[TopicResult] = Field(
        default_factory=list,
        description="Search results organized by topic and decomposition",
    )
    search_metadata: dict = Field(..., description="Search provenance and metadata")
    total_cmr_results: int = Field(
        default=0,
        description="Total collections from CMR across all topics (before filtering)",
    )
    total_filtered_results: int = Field(
        default=0,
        description="Total collections after filtering across all topics",
    )


class BaseDataSearchConfig(BaseAgentConfig):
    """Base configuration for data search agents."""

    # Universal component models (used for all queries)
    topic_splitting_model: str = Field(
        default="gpt-5-nano",
        description="Model for topic splitting",
    )
    scientific_decomposition_model: str = Field(
        default="gpt-5-mini",
        description="Model for scientific decomposition",
    )
    repository_routing_model: str = Field(
        default="gpt-5-mini",
        description="Model for repository routing",
    )

    # General settings
    debug: bool = Field(default=False, description="Enable debug logging")
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel searches",
    )

    # Execution modes
    single_path_mode: bool = Field(
        default=False,
        description="Execute only [0] branch at each selection point (topics, decompositions, approaches, queries)",
    )
    auto_save: bool = Field(
        default=False,
        description="Automatically save results to captured_data/ directory on successful completion",
    )
    capture_metadata: bool = Field(
        default=True,
        description="Include git info, prompt templates, and config in auto-saved output",
    )


class CollectionSynthesisResult(BaseModel):
    """Result from collection synthesis step."""

    selected_collections: List[Dict[str, Any]] = Field(
        ...,
        description="Collections selected for granule search",
    )
    total_collections_found: int = Field(
        default=0,
        description="Total number of collections found before filtering",
    )
    synthesis_reasoning: Optional[str] = Field(
        None,
        description="Reasoning for collection selection/ranking",
    )


class GranuleSynthesisResult(BaseModel):
    """Result from granule synthesis step."""

    granules: List[Dict[str, Any]] = Field(
        ...,
        description="Final list of data granules/files",
    )
    search_metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata about the search process",
    )
    total_granules_found: int = Field(
        default=0,
        description="Total number of granules found",
    )
    collections_processed: int = Field(
        default=0,
        description="Number of collections processed",
    )


class BaseDataSearchAgent[
    TInput: DataSearchAgentInputSchema,
    TOutput: DataSearchAgentOutputSchema,
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
    config_schema = BaseDataSearchConfig

    def _validate_query(self, query: str) -> str:
        """Validate and clean the input query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()

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
            "not get_response_async",
        )
