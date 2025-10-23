from __future__ import annotations

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Literal

from loguru import logger
from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.tools.reranker import (
    RerankerTool,
    RerankerToolConfig,
    RerankerType,
    create_reranker,
)
from akd.utils import reciprocal_rank_fusion


class QueryFocusStrategy(str, Enum):
    """Query adaptation strategies based on rubric analysis."""

    REFINE_TOPIC_SPECIFICITY = "refine_topic_specificity"
    SEARCH_COMPREHENSIVE_REVIEWS = "search_comprehensive_reviews"
    TARGET_PEER_REVIEWED_SOURCES = "target_peer_reviewed_sources"
    SEARCH_METHODOLOGICAL_PAPERS = "search_methodological_papers"
    ADD_RECENT_YEAR_FILTERS = "add_recent_year_filters"
    ADJUST_QUERY_SCOPE = "adjust_query_scope"


class SearchToolConfig(BaseToolConfig):
    """
    Base configuration for search tools.
    This can be extended by specific search tool configurations.
    """

    max_results: int = Field(
        10,
        description="Maximum number of search results to return.",
    )
    timeout: int = Field(
        30,
        description="Timeout in seconds for search requests.",
    )
    reranker_type: RerankerType = Field(
        default="none",
        description=(
            "Type of reranker to use for per-query result reranking before RRF fusion. "
            "Options: 'cross_encoder' (semantic reranking), 'identity'/'no_op' (pass-through), "
            "'none'/'nope' (no reranking, returns original results)."
        ),
    )
    reranker_config: RerankerToolConfig | None = Field(
        default=None,
        description=(
            "Optional custom configuration for the reranker tool. If None, uses default RerankerToolConfig settings."
        ),
    )


class SearchToolInputSchema(InputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content.
    """

    queries: list[str] = Field(..., description="List of search queries.")
    category: Literal["general", "science", "technology"] | None = Field(
        "science",
        description="Category of the search queries.",
    )
    max_results: int | None = Field(
        None,
        description="Maximum number of search results to return.",
    )


class SearchToolOutputSchema(OutputSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content."""

    results: list[SearchResultItem] = Field(
        ...,
        description="List of search result items",
    )
    extra: dict | None = Field(
        default_factory=dict,
        description="Any additional information or metadata related to the search results",
    )


class SearchTool(BaseTool[SearchToolInputSchema, SearchToolOutputSchema]):
    """
    Base tool for performing searches using Reciprocal Rank Fusion (RRF).

    This base class implements a standardized search pattern where:
    1. Each query fetches max_results items (not divided across queries)
    2. Results are aggregated using Reciprocal Rank Fusion
    3. Final results are trimmed to max_results

    Subclasses must implement _arun_single_query() to define how to fetch
    results for a single query.

    Attributes:
        input_schema (SearchToolInputSchema): The schema for the input data.
        output_schema (SearchToolOutputSchema): The schema for the output data.
        config_schema (SearchToolConfig): Configuration schema for the tool.
        reranker (RerankerTool): Reranker instance for per-query result reranking.
    """

    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema
    config_schema = SearchToolConfig

    def _post_init(self) -> None:
        """
        Post-initialization hook to set up the reranker.

        Creates a reranker instance based on config.reranker_type using
        the factory pattern. This allows per-query reranking before RRF fusion.
        """
        super()._post_init()

        # Initialize reranker using factory
        # Attributes are already set from config by parent's _post_init
        self.reranker: RerankerTool = create_reranker(
            reranker_type=self.reranker_type,  # type: ignore
            config=self.reranker_config,  # type: ignore
            debug=self.debug,
        )

    def _process_results(
        self,
        results: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """
        Process and filter search results.

        Default implementation returns results as-is (identity function).
        Subclasses can override to add filtering, scoring, deduplication, etc.

        Args:
            results: List of search result items to process.

        Returns:
            Processed list of search result items.
        """
        return results

    @abstractmethod
    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> SearchToolOutputSchema:
        """
        Fetch search results for a single query.

        Subclasses must implement this method to define their specific
        search behavior (e.g., SearxNG, Serper, Semantic Scholar, vector stores).

        Args:
            query: The search query string.
            max_results: Maximum number of results to fetch for this query.
            **kwargs: Implementation-specific parameters such as:
                - category (str | None): Optional category filter for the search
                - client: HTTP client for web-based search tools
                - index: Vector index for local search tools
                - Any other tool-specific parameters

        Returns:
            SearchToolOutputSchema with results and any metadata for this query.
        """
        pass

    async def _rerank(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        """
        Rerank search results for a single query.

        Args:
            query: The search query used to obtain the results.
            results: List of search results to rerank.

        Returns:
            Reranked list of search results.
        """
        if not results:
            return results

        if self.debug:
            logger.debug(f"Reranking {len(results)} results for query: {query} | Reranker type: {self.reranker}")  # type: ignore
        reranker_input = self.reranker.input_schema(query=query, results=results)
        reranked_output = await self.reranker._arun(reranker_input)
        return reranked_output.results

    def _merge_extra_metadata(self, all_extra: list[dict]) -> dict:
        """
        Merge extra metadata from multiple query results.

        Default implementation merges all dicts using dictionary unpacking.
        Later values override earlier ones for conflicting keys.
        Subclasses can override for custom merging logic (e.g., summing credits).

        Args:
            all_extra: List of extra metadata dicts from each query result.

        Returns:
            Merged metadata dictionary.
        """
        merged = {}
        for extra in all_extra:
            if extra:
                merged = {**merged, **extra}
        return merged

    async def _arun(
        self,
        params: SearchToolInputSchema,
        max_results: int | None = None,
        **kwargs,  # noqa: ARG002
    ) -> SearchToolOutputSchema:
        """
        Runs the search tool using Reciprocal Rank Fusion (RRF) pattern with optional reranking.

        This method:
        1. Fetches max_results for EACH query (not divided)
        2. Reranks each query's results independently (if reranker is configured)
        3. Applies RRF to aggregate and rank results across all queries
        4. Merges metadata from all queries
        5. Trims final results to max_results
        6. Returns unified, ranked results

        Args:
            params: Input parameters including queries and category.
            max_results: Override for maximum results to return.
            **kwargs: Additional keyword arguments passed to _arun_single_query.

        Returns:
            SearchToolOutputSchema with fused and ranked results.
        """
        # Determine final max_results limit
        final_max_results = max_results or params.max_results or self.max_results

        # Extract remaining fields from params to pass through
        remaining_fields = params.model_dump(exclude={"queries", "max_results", "category"})
        merged_kwargs = {**remaining_fields, **kwargs}

        # Fetch max_results for EACH query in parallel
        # Each query execution handles its own resources (HTTP client, etc.)
        tasks = [
            self._arun_single_query(
                query,
                final_max_results,
                category=params.category,
                **merged_kwargs,
            )
            for query in params.queries
        ]
        outputs = await asyncio.gather(*tasks)

        # Extract results and metadata from each output
        results_per_query = [output.results for output in outputs]
        all_extra = [output.extra or {} for output in outputs]

        # Per-query reranking before RRF fusion
        # Rerank each query's results independently for better relevance
        reranked_results_per_query = [
            await self._rerank(query, results) for query, results in zip(params.queries, results_per_query)
        ]

        # Apply Reciprocal Rank Fusion to aggregate reranked results
        fused_results = reciprocal_rank_fusion(
            *reranked_results_per_query,
            key="url",
            normalize=True,
        )

        # Trim to max_results
        final_results = fused_results[:final_max_results]

        # Merge metadata from all queries
        merged_extra = self._merge_extra_metadata(all_extra)

        return self.output_schema(
            results=final_results,
            extra=merged_extra,
        )
