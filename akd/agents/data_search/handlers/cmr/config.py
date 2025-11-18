"""Configuration for CMR repository handler."""

from pydantic import Field

from akd.tools.data_search._base import DataSearchToolConfig


class CMRHandlerConfig(DataSearchToolConfig):
    """Configuration for CMR repository handler.

    Extends DataSearchToolConfig to inherit tool configuration attributes:
    - mcp_endpoint: MCP server endpoint URL
    - timeout_seconds: Request timeout
    - max_retries: Maximum retry attempts
    - retry_delay: Base delay between retries
    - page_size: Default page size
    """

    # Override MCP endpoint default for CMR
    mcp_endpoint: str = Field(
        default="http://localhost:8080/mcp/cmr/mcp/",
        description="CMR MCP server endpoint URL",
    )

    # Search behavior
    collection_search_page_size: int = Field(
        default=20,
        description="Page size for collection searches",
    )
    granule_search_page_size: int = Field(
        default=50,
        description="Page size for granule searches (kept for future use)",
    )

    # Ranking pipeline configuration
    collections_per_query: int = Field(
        default=5,
        description="Top N results to take from each CMR query",
    )
    max_collections_per_approach: int = Field(
        default=5,
        description="Maximum collections to select per query approach",
    )
    final_collection_count: int = Field(
        default=25,
        description="Maximum collections in final ranked output",
    )

    # Performance tuning
    collection_search_timeout: float = Field(
        default=30.0,
        description="Timeout for collection searches in seconds",
    )
    granule_search_timeout: float = Field(
        default=45.0,
        description="Timeout for granule searches in seconds",
    )

    # Query approach configuration
    include_keyword_only_approach: bool = Field(
        default=True,
        description="Add keyword-only approach (no instrument/platform) based on first LLM approach to catch collections with incomplete metadata",
    )
    retrieve_all_cmr_collections: bool = Field(
        default=True,
        description="Retrieve all available collections from CMR via pagination (not just first page). When True, paginates through all results. When False, only retrieves first page.",
    )
    skip_ranking_and_filtering: bool = Field(
        default=False,
        description="Skip LLM-based approach filtering and final ranking stages. Returns all deduplicated collections from all_collections_from_cmr. Use for baseline evaluations and raw CMR analysis.",
    )

    # Component model configuration
    known_parameters_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for known parameters extraction",
    )
    searchable_parameters_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for searchable parameters generation",
    )
    approach_filtering_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for per-approach collection filtering",
    )
    final_ranking_model: str = Field(
        default="gpt-5-mini",
        description="Model to use for final cross-approach ranking",
    )
