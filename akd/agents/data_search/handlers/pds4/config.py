"""Configuration for PDS4 repository handler."""

from pydantic import Field

from akd.tools.data_search._base import DataSearchToolConfig


class PDS4HandlerConfig(DataSearchToolConfig):
    """Configuration for PDS4 repository handler.

    Extends DataSearchToolConfig to inherit tool configuration attributes:
    - mcp_endpoint: MCP server endpoint URL
    - timeout_seconds: Request timeout
    - max_retries: Maximum retry attempts
    - retry_delay: Base delay between retries
    - page_size: Default page size
    """

    # Override MCP endpoint default for PDS4
    mcp_endpoint: str = Field(
        default="http://localhost:8080/mcp/pds4/mcp/",
        description="PDS4 MCP server endpoint URL",
    )

    # Search behavior
    bundle_search_page_size: int = Field(
        default=20,
        description="Page size for bundle searches (exploratory/faceted searches)",
    )
    collection_search_page_size: int = Field(
        default=20,
        description="Page size for collection searches (URN-filtered data retrieval)",
    )
    context_search_page_size: int = Field(
        default=10,
        description="Page size for context searches (investigations/targets)",
    )

    # URN combination strategy
    max_investigation_urns_per_approach: int = Field(
        default=4,
        description="Maximum number of investigation URNs to use per base approach",
    )
    max_target_urns_per_approach: int = Field(
        default=3,
        description="Maximum number of target URNs to use per base approach",
    )
    max_approach_combinations: int = Field(
        default=12,
        description="Maximum total URN combinations to generate per base approach",
    )

    # Ranking pipeline configuration
    collections_per_strategy: int = Field(
        default=5,
        description="Top N results to take from each tool strategy execution",
    )
    max_collections_per_strategy: int = Field(
        default=5,
        description="Maximum collections to select per strategy after filtering",
    )
    final_collection_count: int = Field(
        default=25,
        description="Maximum collections in final ranked output",
    )

    # Quality control
    min_collection_relevance_score: float = Field(
        default=0.3,
        description="Minimum collection relevance score to include",
    )
    min_context_relevance_score: float = Field(
        default=0.4,
        description="Minimum context (investigation/target) relevance score for URN extraction",
    )

    # Performance tuning
    context_search_timeout: float = Field(
        default=20.0,
        description="Timeout for context searches (investigations/targets) in seconds",
    )
    collection_search_timeout: float = Field(
        default=30.0,
        description="Timeout for collection searches in seconds",
    )
    bundle_search_timeout: float = Field(
        default=25.0,
        description="Timeout for bundle searches in seconds",
    )
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel execution of tool strategies",
    )

    # Component model configuration (3 models for 3 components)
    parameter_extraction_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for unified parameter extraction (tool strategies)",
    )
    strategy_filtering_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for per-strategy collection filtering",
    )
    final_ranking_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for final cross-strategy ranking",
    )

    # Advanced options
    max_strategies: int = Field(
        default=4,
        description="Maximum number of tool strategies to generate and execute",
    )
    enable_bundle_supplementary_search: bool = Field(
        default=True,
        description="Enable supplementary bundle title searches when appropriate",
    )
    facet_limit: int = Field(
        default=25,
        description="Number of facet values to return in bundle searches",
    )
