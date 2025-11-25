"""Configuration for CMR repository handler."""
from pydantic import Field

from akd.configs.mcp_config import get_mcp_endpoint
from akd.tools.data_search._base import DataSearchToolConfig


from akd.tools.reranker import LLMRerankerToolConfig


class CMRHandlerConfig(DataSearchToolConfig):
    """Configuration for CMR repository handler.

    Extends DataSearchToolConfig to inherit tool configuration attributes:
    - mcp_endpoint: MCP server endpoint URL (from environment or default)
    - timeout_seconds: Request timeout
    - max_retries: Maximum retry attempts
    - retry_delay: Base delay between retries
    - page_size: Default page size
    """

    # Override MCP endpoint default - uses environment variable if set
    mcp_endpoint: str = Field(
        default_factory=get_mcp_endpoint,
        description="CMR MCP server endpoint URL (set via MCP_ENDPOINT env var or uses default)",
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
        description="Maximum collections in final ranked output (only applied if apply_final_collection_limit=True)",
    )
    apply_final_collection_limit: bool = Field(
        default=True,
        description="Whether to apply final_collection_count limit to final ranked output. When False, returns all ranked collections for recall evaluation. Note: Per-approach limits (max_collections_per_approach) are still applied in legacy ranking.",
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

    # Reranker configuration
    use_llm_reranker: bool = Field(
        default=False,
        description="Use LLM-based reranker instead of old ranking system. When True, uses LLMRerankerTool with configurable criteria. When False, uses legacy LLM-based ranking components.",
    )
    llm_reranker_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for LLM reranker scoring (only used if custom_llm_reranker_config is None)",
    )
    llm_reranker_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM reranker (only used if custom_llm_reranker_config is None)",
    )
    custom_llm_reranker_config: LLMRerankerToolConfig | None = Field(
        default=None,
        description="Optional custom LLMRerankerToolConfig with custom scoring criteria, field descriptions, and weights. If provided, completely overrides default config. If None, uses default CMR criteria with llm_reranker_model and llm_reranker_temperature.",
    )
