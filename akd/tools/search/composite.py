"""Composite search tool for combining results from multiple search tools."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from loguru import logger
from pydantic.fields import Field

from akd.structures import SearchResultItem
from akd.tools.search._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from akd.utils import reciprocal_rank_fusion


class CompositeSearchToolConfig(SearchToolConfig):
    """Configuration for composite search tools with fusion strategy selection."""

    fusion_strategy: Literal["direct_rrf_blackbox", "flatten_rerank_rrf"] = Field(
        default="direct_rrf_blackbox",
        description=(
            "Strategy for fusing results from multiple tools. "
            "'direct_rrf_blackbox': Direct RRF on tool outputs (treats tools as independent blackbox rankers). "
            "'flatten_rerank_rrf': Flatten all results, rerank globally with cross-encoder, then apply RRF (better semantic ranking)."
        ),
    )


class CompositeSearchTool(SearchTool):
    """
    Base class for search tools that combine results from multiple search tools.

    This class provides a standardized pattern for:
    1. Running multiple search tools in parallel with the same params
    2. Aggregating their results
    3. Applying fusion strategy (direct RRF or flatten+rerank+RRF)
    4. Returning unified results

    Supports two fusion strategies:
    - direct_rrf_blackbox: Treat each tool as independent blackbox ranker, apply RRF directly
    - flatten_rerank_rrf: Flatten results, rerank globally with cross-encoder, then apply RRF

    Example:
        >>> class CompositeCodeSearchTool(CompositeSearchTool):
        >>>     def __init__(self, config=None, debug=False):
        >>>         super().__init__(
        >>>             LocalRepoCodeSearchTool(debug=debug),
        >>>             GitHubCodeSearchTool(debug=debug),
        >>>             SDECodeSearchTool(debug=debug),
        >>>             config=config,
        >>>             debug=debug,
        >>>         )
    """

    config_schema = CompositeSearchToolConfig

    def __init__(
        self,
        *search_tools: SearchTool,
        config: CompositeSearchToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initialize composite search tool with multiple search tools.

        Args:
            *search_tools: Variable number of search tools to combine.
            config: Configuration for the composite tool.
            debug: Enable debug logging.
        """
        super().__init__(config, debug)
        if not search_tools:
            raise ValueError("CompositeSearchTool requires at least one search tool")
        self.search_tools = list(search_tools)

    async def _fusion_direct_rrf_blackbox(
        self,
        outputs_per_tool: list[SearchToolOutputSchema],
        params: SearchToolInputSchema,  # noqa: ARG002
    ) -> list[SearchResultItem]:
        """
        Approach 1: Treat each tool as independent blackbox ranker, apply RRF directly.

        Args:
            outputs_per_tool: List of outputs from each search tool.
            params: Input parameters with queries (unused in this approach).

        Returns:
            Fused and ranked list of search results.
        """
        # Collect results per tool (each tool already did per-query reranking and RRF)
        results_per_tool: list[list[SearchResultItem]] = [tool_output.results for tool_output in outputs_per_tool]

        # Apply RRF fusion across all tools (treats tools as blackbox rankers)
        fused_results = reciprocal_rank_fusion(
            *results_per_tool,
            keys=self.config.rrf_keys,
            normalize=True,
        )

        return fused_results

    async def _fusion_flatten_rerank_rrf(
        self,
        outputs_per_tool: list[SearchToolOutputSchema],
        params: SearchToolInputSchema,
    ) -> list[SearchResultItem]:
        """
        Approach 2: Flatten all results, rerank globally with cross-encoder, then apply RRF.

        Args:
            outputs_per_tool: List of outputs from each search tool.
            params: Input parameters with queries.

        Returns:
            Fused and ranked list of search results.
        """
        # Flatten all results from all tools
        all_results: list[SearchResultItem] = []
        for tool_output in outputs_per_tool:
            all_results.extend(tool_output.results)

        # Group by query for global reranking
        # results_per_query[i] = all results for params.queries[i] from all tools combined
        results_per_query: list[list[SearchResultItem]] = []
        for query in params.queries:
            query_results = [r for r in all_results if r.query == query]
            results_per_query.append(query_results)

        # Rerank each query's results using inherited reranker (e.g., cross-encoder)
        # This generates unified scores across all tools' results
        reranked_results_per_query = [
            await self._rerank(query, results) for query, results in zip(params.queries, results_per_query)
        ]

        # Apply RRF fusion across reranked queries
        fused_results = reciprocal_rank_fusion(
            *reranked_results_per_query,
            keys=self.config.rrf_keys,
            normalize=True,
        )

        return fused_results

    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> SearchToolOutputSchema:
        """
        This method is not used by CompositeSearchTool.
        CompositeSearchTool overrides _arun() to handle multi-tool aggregation.
        """
        raise NotImplementedError(
            "CompositeSearchTool does not use _arun_single_query. Multi-tool aggregation is handled in _arun().",
        )

    async def _arun(
        self,
        params: SearchToolInputSchema,
        max_results: int | None = None,
        **kwargs,
    ) -> SearchToolOutputSchema:
        """
        Run search across all tools in parallel and combine results using configured fusion strategy.

        This method:
        1. Runs all search tools in parallel with the same params
        2. Collects outputs from each tool
        3. Adds tool metadata to results
        4. Applies fusion strategy:
           - direct_rrf_blackbox: Direct RRF on tool outputs
           - flatten_rerank_rrf: Flatten, rerank globally, then RRF
        5. Returns unified, ranked results

        Args:
            params: Input parameters including queries and category.
            max_results: Override for maximum results to return.
            **kwargs: Additional keyword arguments (unused, for signature compatibility).

        Returns:
            SearchToolOutputSchema with fused and ranked results from all tools.
        """
        # Run all search tools in parallel using asyncio.gather
        # Each tool handles its own max_results and reranking
        tasks = [tool.arun(tool.input_schema(**params.model_dump())) for tool in self.search_tools]

        if self.debug:
            logger.debug(f"Running {len(tasks)} search tools in parallel")

        # Gather outputs from all tools (may include exceptions due to return_exceptions=True)
        outputs_per_tool_raw: Sequence[SearchToolOutputSchema | BaseException] = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        # Process outputs: filter exceptions and add tool metadata
        all_extra: list[dict] = []
        outputs_per_tool: list[SearchToolOutputSchema] = []

        for tool, output in zip(self.search_tools, outputs_per_tool_raw):
            # Handle exceptions from gather
            if isinstance(output, BaseException):
                logger.error(f"Error running tool {tool.__class__.__name__}: {output}")
                if self.debug:
                    logger.exception(output)
                continue

            # Type guard: at this point, output is SearchToolOutputSchema
            # Add tool metadata to each result
            for result in output.results:
                # Add tool name to result metadata (extra is always initialized via default_factory)
                result.extra["tool"] = tool.__class__.__name__

            outputs_per_tool.append(output)

            # Collect metadata
            if output.extra:
                all_extra.append(output.extra)

        # Apply fusion strategy based on configuration
        if self.fusion_strategy == "flatten_rerank_rrf":  # type: ignore
            fused_results = await self._fusion_flatten_rerank_rrf(outputs_per_tool, params)
        else:  # default: "direct_rrf_blackbox"
            fused_results = await self._fusion_direct_rrf_blackbox(outputs_per_tool, params)

        # Trim to max_results from params
        final_max_results = params.max_results or self.max_results
        final_results = fused_results[:final_max_results]

        # Merge metadata from all tools
        merged_extra = self._merge_extra_metadata(all_extra)

        return self.output_schema(
            results=final_results,
            extra=merged_extra,
        )
