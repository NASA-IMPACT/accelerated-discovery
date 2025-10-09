"""
CMR (Common Metadata Repository) Handler.

Implements repository-specific logic for searching NASA's CMR for Earth science data.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema, DecompositionResult
from akd.agents.data_search.components import ScientificDecomposition, Topic
from akd.tools.data_search import CMRCollectionSearchTool, CMRGranuleSearchTool
from akd.utils.logging import ContextualLogger, log_component_action
from akd.utils.serialization import safe_model_dump, safe_model_dump_list

from .._base import BaseHandler
from .components import (
    CMRApproachCollectionFilteringComponent,
    CMRFinalCollectionRankingComponent,
    CMRKnownParametersComponent,
    CMRSearchableParametersComponent,
)
from .config import CMRHandlerConfig
from .schemas import (
    CMRApproachCollectionFilteringInputSchema,
    CMRFinalCollectionRankingInputSchema,
    CMRSearchableQuery,
)


class CMRHandler(BaseHandler):
    """
    Handler for NASA's Common Metadata Repository (CMR).

    Implements the complete CMR-specific pipeline:
    1. Extract known parameters from decomposition
    2. Generate searchable query variations
    3. Execute CMR collection searches
    4. Filter and rank collections per approach
    5. Final cross-approach ranking
    6. Return collections as data_results (granule search kept for future)
    """

    def __init__(self, config: CMRHandlerConfig, debug: bool = False):
        """Initialize CMR handler with configuration."""
        super().__init__(config, debug)

        # Initialize CMR-specific tools
        tool_config_params = {
            "mcp_endpoint": config.mcp_endpoint,
            "timeout_seconds": config.collection_search_timeout,
            "debug": debug,
        }

        self.collection_search_tool = CMRCollectionSearchTool.from_params(
            page_size=config.collection_search_page_size,
            **tool_config_params,
        )

        granule_tool_config = tool_config_params.copy()
        granule_tool_config["timeout_seconds"] = config.granule_search_timeout

        self.granule_search_tool = CMRGranuleSearchTool.from_params(
            collection_concept_id="",  # Will be set per search
            page_size=config.granule_search_page_size,
            **granule_tool_config,
        )

        # Get CMR prompts directory
        cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Initialize CMR-specific components with model configurations and prompts_dir
        known_params_config = BaseAgentConfig(model_name=config.known_parameters_model)
        searchable_params_config = BaseAgentConfig(
            model_name=config.searchable_parameters_model,
        )

        self.known_parameters_component = CMRKnownParametersComponent(
            config=known_params_config,
            debug=debug,
            prompts_dir=cmr_prompts_dir,
        )
        self.searchable_parameters_component = CMRSearchableParametersComponent(
            config=searchable_params_config,
            debug=debug,
            prompts_dir=cmr_prompts_dir,
        )

        # Logger
        self.handler_logger = ContextualLogger("CMRHandler")

    async def process_decomposition(
        self,
        decomposition: ScientificDecomposition,
        topic: Topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> DecompositionResult:
        """
        Process a scientific decomposition through CMR pipeline.

        Pipeline:
        1. Known Parameters → Extract hard filters
        2. Searchable Parameters → Generate query variations
        3. Collection Search → Execute CMR queries
        4. Collection Ranking → Filter and rank results
        5. Return collections as data_results

        Granule search is kept for future use but not currently executed.
        """
        self.handler_logger.info(f"Processing decomposition: {decomposition.title}")

        # Step 1: Known Parameters
        log_component_action(
            "KnownParameters",
            "STARTED",
            {"decomposition": decomposition.title},
        )
        known_params_output = await self.known_parameters_component.process(
            original_query,
            topic,
            decomposition,
        )
        self.handler_logger.info(
            f"Generated {len(known_params_output.query_approaches)} query approaches",
        )

        # Step 2: Searchable Parameters
        log_component_action(
            "SearchableParameters",
            "STARTED",
            {"approaches": len(known_params_output.query_approaches)},
        )
        searchable_output = await self.searchable_parameters_component.process(
            original_query,
            topic,
            decomposition,
            known_params_output.query_approaches,
        )
        self.handler_logger.info(
            f"Generated {len(searchable_output.searchable_queries)} searchable queries",
        )

        # Step 3: Query Execution
        log_component_action(
            "QueryExecution",
            "STARTED",
            {"queries": len(searchable_output.searchable_queries)},
        )
        approach_collections = await self._execute_searchable_queries(
            searchable_output.searchable_queries,
            params,
        )
        total_collections = sum(len(c) for c in approach_collections.values())
        self.handler_logger.info(
            f"Found {total_collections} collections across {len(approach_collections)} approaches",
        )

        # Step 4: Collection Ranking & Filtering
        log_component_action(
            "CollectionRanking",
            "STARTED",
            {
                "approaches": len(approach_collections),
                "total_collections": total_collections,
            },
        )
        ranked_collections = await self._rank_collections(
            approach_collections,
            original_query,
            topic,
            decomposition,
            known_params_output.query_approaches,
        )
        self.handler_logger.info(f"Ranked to {len(ranked_collections)} top collections")

        # Step 5: Granule Search (KEPT FOR FUTURE USE - NOT CURRENTLY CALLED)
        # granules = await self._search_granules_for_collections(
        #     ranked_collections,
        #     params,
        # )

        return DecompositionResult(
            decomposition=safe_model_dump(decomposition),
            repository="CMR",
            query_approaches=safe_model_dump_list(known_params_output.query_approaches),
            searchable_queries=safe_model_dump_list(
                searchable_output.searchable_queries,
            ),
            data_results=ranked_collections,  # Collections for now
            total_results_found=total_collections,
            note=None,
        )

    async def _execute_searchable_queries(
        self,
        searchable_queries: List[CMRSearchableQuery],
        params: DataSearchAgentInputSchema,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Execute searchable queries and return collections grouped by approach.

        Args:
            searchable_queries: Queries to execute (each tagged with approach_index)
            params: Search parameters

        Returns:
            Dictionary mapping approach_index to list of collections
        """
        # Group queries by their source approach
        approach_queries = {}  # approach_index -> [queries]
        for query in searchable_queries:
            approach_idx = query.approach_index
            if approach_idx not in approach_queries:
                approach_queries[approach_idx] = []
            approach_queries[approach_idx].append(query)

        # Execute and collect results per approach
        approach_collections = {}  # approach_index -> [collections]

        for approach_idx, queries in approach_queries.items():
            all_collections = []

            for query in queries:
                # Get CMR-compatible parameters using helper method
                search_params = query.get_mcp_parameters()

                # Override with explicit input parameters
                if params.temporal_range:
                    search_params["temporal"] = params.temporal_range
                if params.spatial_bounds:
                    search_params["bounding_box"] = params.spatial_bounds

                # Add pagination
                search_params["page_size"] = self.config.collection_search_page_size

                try:
                    # Execute search
                    tool_input = self.collection_search_tool.input_schema(
                        **search_params,
                    )
                    result = await self.collection_search_tool.arun(tool_input)

                    # Extract and limit collections per query
                    if hasattr(result, "collections") and result.collections:
                        limited = result.collections[
                            : self.config.collections_per_query
                        ]
                        all_collections.extend(limited)

                except Exception as e:
                    self.handler_logger.warning(f"Query execution failed: {e}")

            approach_collections[approach_idx] = all_collections

        return approach_collections

    def _deduplicate_approach_collections(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Deduplicate collections within each approach, preserving first occurrence.
        If a collection appears in multiple approaches, keep it only in the first.
        """
        global_seen_ids = set()  # Track across all approaches
        deduplicated_by_approach = {}

        # Process approaches in order (0, 1, 2, ...)
        for approach_idx in sorted(approach_collections.keys()):
            collections = approach_collections[approach_idx]
            deduplicated = []

            for collection in collections:
                concept_id = collection.get("concept_id")
                if concept_id and concept_id not in global_seen_ids:
                    global_seen_ids.add(concept_id)
                    deduplicated.append(collection)

            deduplicated_by_approach[approach_idx] = deduplicated

            self.handler_logger.debug(
                f"Approach {approach_idx}: {len(collections)} → "
                f"{len(deduplicated)} after deduplication",
            )

        return deduplicated_by_approach

    async def _filter_and_rank_by_approach(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Filter and rank collections within each approach in parallel.
        """
        if not approach_collections:
            return {}

        # Get CMR prompts directory
        cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Create filtering tasks for each approach
        filtering_tasks = []

        for approach_idx in sorted(approach_collections.keys()):
            collections = approach_collections[approach_idx]

            if not collections:
                continue

            # Get the corresponding QueryApproach
            if approach_idx >= len(query_approaches):
                self.handler_logger.warning(
                    f"No QueryApproach for index {approach_idx}",
                )
                continue

            approach = query_approaches[approach_idx]

            filter_input = CMRApproachCollectionFilteringInputSchema(
                original_query=original_query,
                topic_title=topic.title,
                topic_context=topic.functional_context,
                decomposition_title=decomp.title,
                decomposition_justification=decomp.scientific_justification,
                approach_instrument=approach.instrument,
                approach_platform=approach.platform,
                approach_processing_level=approach.processing_level,
                approach_temporal_range=approach.temporal,
                approach_spatial_bounds=approach.bounding_box,
                approach_temporal_resolution=approach.temporal_resolution,
                approach_spatial_resolution=approach.spatial_resolution,
                approach_keywords=[],  # Keywords are in searchable queries
                collections=collections,
                max_collections=self.config.max_collections_per_approach,
            )

            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.approach_filtering_model,
            )
            filtering_component = CMRApproachCollectionFilteringComponent(
                config=component_config,
                prompts_dir=cmr_prompts_dir,
            )

            task = filtering_component.arun(filter_input)
            filtering_tasks.append((approach_idx, collections, task))

        # Execute in parallel
        if len(filtering_tasks) > 1:
            results = await asyncio.gather(
                *[task for _, _, task in filtering_tasks],
                return_exceptions=True,
            )
        else:
            results = []
            for _, _, task in filtering_tasks:
                result = await task
                results.append(result)

        # Map results back to collections
        filtered_by_approach = {}

        for (approach_idx, collections, _), result in zip(filtering_tasks, results):
            if isinstance(result, Exception):
                self.handler_logger.error(
                    f"Approach {approach_idx} filtering failed: {result}",
                )
                continue

            # Extract selected collections
            selected = [
                collections[fc.collection_index]
                for fc in result.selected_collections
                if 0 <= fc.collection_index < len(collections)
            ]

            filtered_by_approach[approach_idx] = selected

            self.handler_logger.info(
                f"Approach {approach_idx}: {len(collections)} → "
                f"{len(selected)} collections selected",
            )

        return filtered_by_approach

    async def _rank_collections(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank collections using approach-aware pipeline.

        Pipeline:
        1. Per-approach deduplication
        2. Per-approach filtering and ranking (parallel)
        3. Final cross-approach ranking
        """
        if not approach_collections:
            return []

        # Stage 1: Per-approach deduplication
        deduplicated = self._deduplicate_approach_collections(approach_collections)

        total_before = sum(len(c) for c in approach_collections.values())
        total_after = sum(len(c) for c in deduplicated.values())
        self.handler_logger.info(
            f"Deduplication across {len(deduplicated)} approaches: "
            f"{total_before} → {total_after} collections",
        )

        # Stage 2: Per-approach filtering and ranking (parallel)
        filtered_by_approach = await self._filter_and_rank_by_approach(
            deduplicated,
            original_query,
            topic,
            decomp,
            query_approaches,
        )

        # Flatten all approach results into single list
        all_filtered = []
        for approach_idx in sorted(filtered_by_approach.keys()):
            all_filtered.extend(filtered_by_approach[approach_idx])

        self.handler_logger.info(
            f"After approach filtering: {len(all_filtered)} total collections "
            f"from {len(filtered_by_approach)} approaches",
        )

        if not all_filtered:
            self.handler_logger.warning("No collections passed approach filtering")
            return []

        # Get CMR prompts directory
        cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Stage 3: Final cross-approach ranking
        final_input = CMRFinalCollectionRankingInputSchema(
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomp.title,
            decomposition_justification=decomp.scientific_justification,
            collections=all_filtered,
            max_collections=min(len(all_filtered), self.config.final_collection_count),
        )

        try:
            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.final_ranking_model,
            )
            final_ranking_component = CMRFinalCollectionRankingComponent(
                config=component_config,
                prompts_dir=cmr_prompts_dir,
            )

            final_result = await final_ranking_component.arun(final_input)

            # Map indices to full collection objects and sort by rank
            final_ranked = [
                all_filtered[rc.collection_index]
                for rc in sorted(
                    final_result.ranked_collections,
                    key=lambda x: x.final_rank,
                )
                if 0 <= rc.collection_index < len(all_filtered)
            ]

            self.handler_logger.info(
                f"Final ranking complete: {len(final_ranked)} collections ranked",
            )

            return final_ranked

        except Exception as e:
            self.handler_logger.error(f"Final ranking failed: {e}")
            # Fallback: return up to final_collection_count
            return all_filtered[: self.config.final_collection_count]

    async def _search_granules_for_collections(
        self,
        collections: List[Dict[str, Any]],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """
        Search for granules in the provided collections.

        KEPT FOR FUTURE USE - Currently not called in process_decomposition.

        To enable: Uncomment granule search call in process_decomposition
        and include granules in data_results.
        """
        all_granules = []

        for collection in collections:
            concept_id = collection.get("concept_id")
            if not concept_id:
                continue

            # Build granule search parameters
            granule_params = {
                "collection_concept_id": concept_id,
                "page_size": self.config.granule_search_page_size,
            }

            # Add temporal/spatial constraints from input params
            if params.temporal_range:
                granule_params["temporal"] = params.temporal_range
            if params.spatial_bounds:
                granule_params["bounding_box"] = params.spatial_bounds

            try:
                # Execute granule search
                granule_search_params = self.granule_search_tool.input_schema(
                    **granule_params,
                )
                result = await self.granule_search_tool.arun(granule_search_params)

                # Extract granules
                if hasattr(result, "results") and result.results.get("granules"):
                    all_granules.extend(result.results["granules"])

            except Exception as e:
                self.handler_logger.warning(
                    f"Granule search failed for collection {concept_id}: {e}",
                )

        return all_granules
