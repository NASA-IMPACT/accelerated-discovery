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
        self.cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Store component configs for creating per-call instances
        # This prevents race conditions when processing multiple decompositions in parallel
        self.known_params_config = BaseAgentConfig(
            model_name=config.known_parameters_model,
        )
        self.searchable_params_config = BaseAgentConfig(
            model_name=config.searchable_parameters_model,
        )

    @property
    def known_parameters_component(self):
        """
        Create and return a fresh CMRKnownParametersComponent instance.

        This property creates a new component instance each time it's accessed to avoid
        race conditions in parallel execution. Use for testing/demos only.

        Returns:
            Fresh CMRKnownParametersComponent instance
        """
        return CMRKnownParametersComponent(
            config=self.known_params_config,
            debug=self.debug,
            prompts_dir=self.cmr_prompts_dir,
        )

    @property
    def searchable_parameters_component(self):
        """
        Create and return a fresh CMRSearchableParametersComponent instance.

        This property creates a new component instance each time it's accessed to avoid
        race conditions in parallel execution. Use for testing/demos only.

        Returns:
            Fresh CMRSearchableParametersComponent instance
        """
        return CMRSearchableParametersComponent(
            config=self.searchable_params_config,
            debug=self.debug,
            prompts_dir=self.cmr_prompts_dir,
        )

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
        # Create fresh component instances for this decomposition to avoid race conditions
        # (multiple decompositions may be processing in parallel)
        known_parameters_component = CMRKnownParametersComponent(
            config=self.known_params_config,
            debug=self.debug,
            prompts_dir=self.cmr_prompts_dir,
        )
        searchable_parameters_component = CMRSearchableParametersComponent(
            config=self.searchable_params_config,
            debug=self.debug,
            prompts_dir=self.cmr_prompts_dir,
        )

        # Step 1: Known Parameters
        known_params_output = await known_parameters_component.process(
            original_query,
            topic,
            decomposition,
        )

        # Step 2: Searchable Parameters
        searchable_output = await searchable_parameters_component.process(
            original_query,
            topic,
            decomposition,
            known_params_output.query_approaches,
        )

        # Step 3: Query Execution
        approach_collections = await self._execute_searchable_queries(
            searchable_output.searchable_queries,
            params,
        )
        total_collections = sum(len(c) for c in approach_collections.values())

        # Step 4: Collection Ranking & Filtering
        ranked_collections = await self._rank_collections(
            approach_collections,
            original_query,
            topic,
            decomposition,
            known_params_output.query_approaches,
        )

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

    async def _execute_single_query(
        self,
        query: CMRSearchableQuery,
        params: DataSearchAgentInputSchema,
        approach_idx: int,
    ) -> Dict[str, Any]:
        """
        Execute a single CMR query and return results with approach index.

        Args:
            query: Query to execute
            params: Search parameters
            approach_idx: Index of the approach this query belongs to

        Returns:
            Dictionary with approach_idx and collections
        """
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
            tool_input = self.collection_search_tool.input_schema(**search_params)
            result = await self.collection_search_tool.arun(tool_input)

            # Extract and limit collections per query
            collections = []
            if hasattr(result, "collections") and result.collections:
                collections = result.collections[: self.config.collections_per_query]

            return {"approach_idx": approach_idx, "collections": collections}

        except Exception:
            # Return empty result on failure
            return {"approach_idx": approach_idx, "collections": []}

    async def _execute_searchable_queries(
        self,
        searchable_queries: List[CMRSearchableQuery],
        params: DataSearchAgentInputSchema,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Execute searchable queries in parallel and return collections grouped by approach.

        Args:
            searchable_queries: Queries to execute (each tagged with approach_index)
            params: Search parameters

        Returns:
            Dictionary mapping approach_index to list of collections
        """
        # Create parallel tasks for all queries across all approaches
        query_tasks = []
        for query in searchable_queries:
            task = self._execute_single_query(query, params, query.approach_index)
            query_tasks.append(task)

        # Execute all queries in parallel
        results = await asyncio.gather(*query_tasks, return_exceptions=True)

        # Group results by approach
        approach_collections = {}  # approach_index -> [collections]

        for result in results:
            if isinstance(result, Exception):
                # Skip failed queries
                continue

            approach_idx = result["approach_idx"]
            collections = result["collections"]

            if approach_idx not in approach_collections:
                approach_collections[approach_idx] = []
            approach_collections[approach_idx].extend(collections)

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
                continue

            approach = query_approaches[approach_idx]

            filter_input = CMRApproachCollectionFilteringInputSchema(
                original_query=original_query,
                topic_title=topic.title,
                topic_context=topic.functional_context,
                decomposition_title=decomp.title,
                decomposition_justification=decomp.scientific_justification,
                approach=approach,  # Pass whole approach object
                approach_keywords=[],  # Keywords are in searchable queries
                data_items=collections,  # Use base class field name
                max_items=self.config.max_collections_per_approach,  # Use base class field name
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
                continue

            # Extract selected collections
            selected = [
                collections[fc.collection_index]
                for fc in result.selected_items  # Use base field name
                if 0 <= fc.collection_index < len(collections)
            ]

            filtered_by_approach[approach_idx] = selected

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

        if not all_filtered:
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
            data_items=all_filtered,  # Use base class field name
            max_items=min(
                len(all_filtered),
                self.config.final_collection_count,
            ),  # Use base class field name
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
                    final_result.ranked_items,  # Use base field name
                    key=lambda x: x.final_rank,
                )
                if 0 <= rc.collection_index < len(all_filtered)
            ]

            return final_ranked

        except Exception:
            # Fallback: return up to final_collection_count
            return all_filtered[: self.config.final_collection_count]

    async def _search_granules_for_single_collection(
        self,
        collection: Dict[str, Any],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """
        Search for granules in a single collection.

        Args:
            collection: Collection to search
            params: Search parameters

        Returns:
            List of granules found (empty list on failure)
        """
        concept_id = collection.get("concept_id")
        if not concept_id:
            return []

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
                return result.results["granules"]

            return []

        except Exception:
            return []

    async def _search_granules_for_collections(
        self,
        collections: List[Dict[str, Any]],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """
        Search for granules in the provided collections in parallel.

        KEPT FOR FUTURE USE - Currently not called in process_decomposition.

        To enable: Uncomment granule search call in process_decomposition
        and include granules in data_results.
        """
        # Create parallel tasks for all collections
        granule_tasks = [
            self._search_granules_for_single_collection(collection, params)
            for collection in collections
            if collection.get("concept_id")
        ]

        # Execute all granule searches in parallel
        granule_results = await asyncio.gather(*granule_tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        all_granules = []
        for result in granule_results:
            if isinstance(result, Exception):
                # Skip failed granule searches
                continue
            all_granules.extend(result)

        return all_granules
