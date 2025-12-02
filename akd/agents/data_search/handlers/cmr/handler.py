"""
CMR (Common Metadata Repository) Handler.

Implements repository-specific logic for searching NASA's CMR for Earth science data.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from akd.utils import safe_model_dump
from loguru import logger

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema, DecompositionResult
from akd.agents.data_search.components import ScientificDecomposition, Topic
from akd.agents.data_search.utils.cmr_enum_validator import CMREnumValidator
from akd.tools.data_search import CMRCollectionSearchTool, CMRGranuleSearchTool
from akd.tools.reranker import LLMRerankerToolConfig

from .._base import BaseHandler
from .components import (
    CMRApproachCollectionFilteringComponent,
    CMRFinalCollectionRankingComponent,
    CMRKnownParametersComponent,
    CMRSearchableParametersComponent,
)
from .config import CMRHandlerConfig
from .llm_reranker_adapter import LLMRerankerAdapter
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

    def __init__(
        self,
        config: CMRHandlerConfig,
        debug: bool = False,
        single_path_mode: bool = False,
    ):
        """Initialize CMR handler with configuration."""
        super().__init__(config, debug)
        self.single_path_mode = single_path_mode

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

        # Initialize CMR enum validator for instrument/platform validation
        self.enum_validator = CMREnumValidator(
            threshold=0.7,
            debug=debug,
        )

        # Initialize LLM reranker adapter if configured
        self.reranker_adapter = None
        if config.use_llm_reranker:
            # Use custom config if provided, otherwise create with default CMR config
            if config.custom_llm_reranker_config:
                reranker_config = config.custom_llm_reranker_config
                self.reranker_adapter = LLMRerankerAdapter(
                    config=reranker_config,
                    debug=debug,
                )
                if debug:
                    logger.info(
                        f"Initialized LLM reranker adapter with custom config (model: {reranker_config.model_name}, "
                        f"{len(reranker_config.scoring_criteria)} criteria)",
                    )
            else:
                self.reranker_adapter = LLMRerankerAdapter.from_default_cmr_config(
                    model_name=config.llm_reranker_model,
                    temperature=config.llm_reranker_temperature,
                    debug=debug,
                )
                if debug:
                    logger.info(f"Initialized LLM reranker adapter with default CMR config (model: {config.llm_reranker_model})")


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


    def _maybe_add_keyword_only_approach(self, llm_approaches):
        """
        Optionally add a keyword-only approach by cloning first approach.

        The keyword-only approach removes instrument/platform filters to cast
        a wider net for collections that may not have proper metadata tagging.
        It preserves all other filters (temporal, spatial, processing_level, etc.).

        This is Option 1: Keyword-only counts as an EXTRA approach (max+1).
        LLM generates 1-4, we add keyword-only → 2-5 total.

        Args:
            llm_approaches: Approaches generated by LLM (unmodified)

        Returns:
            Modified approach list with keyword-only approach prepended
        """
        if not self.config.include_keyword_only_approach:
            return llm_approaches

        if len(llm_approaches) == 0:
            return llm_approaches

        # Clone the first approach (most relevant from LLM)
        keyword_only = llm_approaches[0].model_copy(deep=True)
        keyword_only.instrument = None
        keyword_only.platform = None

        if self.debug:
            logger.debug(
                f"Added keyword-only approach (no instrument/platform filters) based on approach[0]. "
                f"Total approaches: {len(llm_approaches) + 1} (1 keyword-only + {len(llm_approaches)} LLM)",
            )

        # Prepend keyword-only approach (position doesn't affect results due to parallel processing)
        return [keyword_only] + llm_approaches

    async def process_decomposition(
        self,
        decomposition: ScientificDecomposition,
        topic: Topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
        run_id: str,
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
            run_id=run_id,
        )
        searchable_parameters_component = CMRSearchableParametersComponent(
            config=self.searchable_params_config,
            debug=self.debug,
            prompts_dir=self.cmr_prompts_dir,
            run_id=run_id,
        )

        # Step 1: Known Parameters
        known_params_output = await known_parameters_component.process(
            original_query,
            topic,
            decomposition,
        )

        # Step 1.5: Validate and correct instrument/platform enums
        validated_approaches = []
        all_corrections = []

        for approach in known_params_output.query_approaches:
            corrected_approach, corrections_metadata = self.enum_validator.validate_approach(
                approach,
            )
            validated_approaches.append(corrected_approach)
            all_corrections.append(corrections_metadata)

        # Log summary of corrections
        if self.debug:
            total_corrections = sum(1 for c in all_corrections if c["corrections_applied"])
            if total_corrections > 0:
                logger.info(
                    f"Applied enum corrections to {total_corrections}/{len(validated_approaches)} approaches",
                )

        # Replace original approaches with validated ones
        known_params_output.query_approaches = validated_approaches

        # Step 1.6: Optionally add keyword-only approach
        known_params_output.query_approaches = self._maybe_add_keyword_only_approach(
            known_params_output.query_approaches,
        )

        # Select approaches based on execution mode
        if self.single_path_mode and known_params_output.query_approaches:
            approaches_to_use = [known_params_output.query_approaches[0]]
            if self.debug:
                print(
                    f"Single-path mode: Using approach[0], generated {len(known_params_output.query_approaches)} total",
                )
        else:
            approaches_to_use = known_params_output.query_approaches

        # Step 2: Searchable Parameters
        searchable_output = await searchable_parameters_component.process(
            original_query,
            topic,
            decomposition,
            approaches_to_use,
        )

        # Select queries based on execution mode
        if self.single_path_mode and searchable_output.searchable_queries:
            # In single-path mode, prefer a query with search string over one without
            queries_with_search_strings = [q for q in searchable_output.searchable_queries if q.search_string]

            if queries_with_search_strings:
                queries_to_execute = [queries_with_search_strings[0]]
                if self.debug:
                    print(
                        f"Single-path mode: Executing query with search string, "
                        f"generated {len(searchable_output.searchable_queries)} total",
                    )
            else:
                queries_to_execute = [searchable_output.searchable_queries[0]]
                if self.debug:
                    print(
                        f"Single-path mode: Executing query[0] (no search string), "
                        f"generated {len(searchable_output.searchable_queries)} total",
                    )
        else:
            queries_to_execute = searchable_output.searchable_queries

        # Step 3: Query Execution
        (
            approach_collections,
            approach_collections_by_query,
            query_execution_logs,
            all_collections_unranked,
        ) = await self._execute_searchable_queries(
            queries_to_execute,
            params,
        )
        total_collections = sum(len(c) for c in approach_collections.values())

        # Step 4: Collection Ranking & Filtering
        # IMPORTANT: Pass approaches_to_use (includes keyword-only) not the original approaches
        # because searchable queries are tagged with indices from approaches_to_use
        ranked_collections, ranking_metadata = await self._rank_collections(
            approach_collections,
            approach_collections_by_query,
            original_query,
            topic,
            decomposition,
            approaches_to_use,
            run_id,
            # searchable_output.searchable_queries,  # Re-enable this line
        )

        print(
            f"DEBUG: process_decomposition - ranked_collections has {len(ranked_collections)} items",
        )
        print(f"DEBUG: process_decomposition - total_collections = {total_collections}")

        # Augment queries with execution metadata and enum corrections
        augmented_queries, total_cmr = self._augment_queries_with_execution_metadata(
            searchable_output.searchable_queries,
            query_execution_logs,
            all_corrections,
        )

        # Step 5: Granule Search (KEPT FOR FUTURE USE - NOT CURRENTLY CALLED)
        # granules = await self._search_granules_for_collections(
        #     ranked_collections,
        #     params,
        # )

        result = DecompositionResult(
            decomposition=decomposition.model_dump(exclude_none=True),
            repository="CMR",
            query_approaches=[
                a.model_dump(exclude_none=True) for a in approaches_to_use
            ],  # Use actual approaches (includes keyword-only)
            searchable_queries=augmented_queries,  # Now includes execution metadata
            all_collections_from_cmr=all_collections_unranked,  # NEW: ALL retrieved collections
            data_results=ranked_collections,  # Final ranked collections (subset of all_collections)
            total_results_from_cmr=total_cmr,
            total_results_after_filtering=len(ranked_collections),
            enum_corrections=all_corrections,  # Instrument/platform corrections metadata
            ranking_fallbacks=ranking_metadata if ranking_metadata else None,  # Fallback metadata
            note=None,
        )

        print(
            f"DEBUG: DecompositionResult created with data_results={len(result.data_results)} items",
        )
        return result

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

        # Add pagination
        search_params["page_size"] = self.config.collection_search_page_size

        try:
            print(f"DEBUG: CMR Search Parameters: {search_params}")

            # Retrieve collections - paginate if config enabled
            all_collections_retrieved = []
            total_hits = 0

            if self.config.retrieve_all_cmr_collections:
                # Paginate through all results
                page_num = 1
                while True:
                    search_params["page_num"] = page_num
                    tool_input = self.collection_search_tool.input_schema(
                        **search_params,
                    )
                    result = await self.collection_search_tool.arun(tool_input)

                    # Update total_hits from CMR response
                    if hasattr(result, "total_hits"):
                        total_hits = result.total_hits

                    # Get collections from this page
                    page_collections = (
                        result.collections if hasattr(result, "collections") and result.collections else []
                    )

                    if not page_collections:
                        # No more results
                        break

                    all_collections_retrieved.extend(page_collections)

                    print(
                        f"DEBUG: Page {page_num} retrieved {len(page_collections)} collections (total so far: {len(all_collections_retrieved)}/{total_hits})",
                    )

                    # Check if we've retrieved all available
                    if len(all_collections_retrieved) >= total_hits:
                        break

                    page_num += 1

                print(
                    f"DEBUG: Pagination complete - retrieved {len(all_collections_retrieved)} of {total_hits} total collections",
                )
            else:
                # Single page retrieval (legacy behavior)
                tool_input = self.collection_search_tool.input_schema(**search_params)
                result = await self.collection_search_tool.arun(tool_input)

                if hasattr(result, "collections") and result.collections:
                    all_collections_retrieved = result.collections
                if hasattr(result, "total_hits"):
                    total_hits = result.total_hits

            # Limit collections for ranking pipeline
            collections_for_ranking = all_collections_retrieved[: self.config.collections_per_query]

            print(
                f"DEBUG: Using {len(collections_for_ranking)} collections for ranking (from {len(all_collections_retrieved)} retrieved) for approach {approach_idx}",
            )
            if collections_for_ranking:
                print(
                    f"DEBUG: First collection title: {collections_for_ranking[0].get('entry_title', 'N/A')[:80]}",
                )

            return {
                "approach_idx": approach_idx,
                "all_collections": all_collections_retrieved,  # NEW: All retrieved collections
                "collections": collections_for_ranking,  # Collections for ranking (limited)
                "query_object": query,
                "execution_metadata": {
                    "mcp_parameters_sent": search_params,
                    "cmr_collections_returned": len(all_collections_retrieved),
                    "cmr_total_hits": total_hits,
                },
            }

        except Exception as e:
            print(f"DEBUG: CMR search failed for approach {approach_idx}: {e}")
            # Return empty result on failure
            return {
                "approach_idx": approach_idx,
                "all_collections": [],
                "collections": [],
                "query_object": query,
                "execution_metadata": {
                    "mcp_parameters_sent": search_params,
                    "cmr_collections_returned": 0,
                    "cmr_total_hits": 0,
                },
            }

    async def _execute_searchable_queries(
        self,
        searchable_queries: List[CMRSearchableQuery],
        params: DataSearchAgentInputSchema,
    ) -> tuple[
        Dict[int, List[Dict[str, Any]]],
        Dict[int, List[List[Dict[str, Any]]]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Execute searchable queries in parallel and return collections grouped by approach.

        Args:
            searchable_queries: Queries to execute (each tagged with approach_index)
            params: Search parameters

        Returns:
            Tuple of (approach_collections, approach_collections_by_query, execution_logs, all_collections_unranked)
            - approach_collections: Dictionary mapping approach_index to flat list of collections (limited for ranking)
            - approach_collections_by_query: Dictionary mapping approach_index to list of query result lists
            - execution_logs: List of execution metadata for each query
            - all_collections_unranked: List of ALL collections retrieved from CMR (before any limits)
        """
        # Create parallel tasks for all queries across all approaches
        query_tasks = []
        for query in searchable_queries:
            task = self._execute_single_query(query, params, query.approach_index)
            query_tasks.append(task)

        # Execute all queries in parallel
        results = await asyncio.gather(*query_tasks, return_exceptions=True)

        # Group results by approach and collect execution logs
        approach_collections = {}  # approach_index -> [collections for ranking - limited]
        approach_collections_by_query = {}  # approach_index -> [[query0], [query1], ...]
        all_collections_unranked = []  # ALL collections retrieved from CMR
        execution_logs = []

        for result in results:
            if isinstance(result, Exception):
                # Skip failed queries
                continue

            approach_idx = result["approach_idx"]
            collections = result["collections"]  # Limited for ranking
            all_collections = result.get("all_collections", [])  # ALL retrieved

            # Track ALL retrieved collections (NEW)
            all_collections_unranked.extend(all_collections)

            # Track flat list (for normal ranking flow - limited)
            if approach_idx not in approach_collections:
                approach_collections[approach_idx] = []
            approach_collections[approach_idx].extend(collections)

            # Track grouped list (for fallback round-robin)
            if approach_idx not in approach_collections_by_query:
                approach_collections_by_query[approach_idx] = []
            approach_collections_by_query[approach_idx].append(collections)

            # Collect execution metadata
            execution_logs.append(
                {
                    "query": result["query_object"],
                    "metadata": result["execution_metadata"],
                },
            )

        return (
            approach_collections,
            approach_collections_by_query,
            execution_logs,
            all_collections_unranked,
        )

    def _augment_queries_with_execution_metadata(
        self,
        searchable_queries: List[CMRSearchableQuery],
        execution_logs: List[Dict[str, Any]],
        enum_corrections: List[Dict[str, Any]] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Augment searchable query dicts with execution metadata and enum corrections.

        Args:
            searchable_queries: Original query objects
            execution_logs: Execution metadata from _execute_searchable_queries
            enum_corrections: List of correction metadata per approach (indexed by approach_index)

        Returns:
            Tuple of (augmented_queries, total_cmr_results)
            - augmented_queries: List of query dicts with execution metadata added
            - total_cmr_results: Sum of cmr_collections_returned across all queries
        """
        augmented = []
        total_cmr = 0

        for query in searchable_queries:
            query_dict = query.model_dump(exclude_none=True)

            # Find matching execution log (match by query object identity)
            matching_log = next(
                (log for log in execution_logs if log["query"] == query),
                None,
            )

            if matching_log:
                metadata = matching_log["metadata"]
                query_dict["mcp_parameters_sent"] = metadata["mcp_parameters_sent"]
                query_dict["cmr_collections_returned"] = metadata["cmr_collections_returned"]
                total_cmr += metadata["cmr_collections_returned"]

            # Add enum corrections for this query's approach
            if enum_corrections and 0 <= query.approach_index < len(enum_corrections):
                query_dict["enum_corrections"] = enum_corrections[query.approach_index]

            augmented.append(query_dict)

        return augmented, total_cmr

    def _round_robin_select(
        self,
        grouped_lists: List[List[Dict[str, Any]]],
        max_items: int,
    ) -> List[Dict[str, Any]]:
        """
        Select items round-robin from grouped lists.

        Takes first item from each group, then second from each, etc.

        Args:
            grouped_lists: List of lists (e.g., [[query0_colls], [query1_colls], [query2_colls]])
            max_items: Maximum items to select

        Returns:
            Round-robin selected items

        Example:
            Input: [[A1, A2, A3], [B1, B2], [C1, C2, C3]], max_items=7
            Output: [A1, B1, C1, A2, B2, C2, A3]
        """
        result = []
        max_depth = max(len(group) for group in grouped_lists) if grouped_lists else 0

        for depth in range(max_depth):
            for group in grouped_lists:
                if len(result) >= max_items:
                    return result
                if depth < len(group):
                    result.append(group[depth])

        return result

    def _deduplicate_approach_collections_grouped(
        self,
        approach_collections_by_query: Dict[int, List[List[Dict[str, Any]]]],
    ) -> Dict[int, List[List[Dict[str, Any]]]]:
        """
        Deduplicate collections within each approach, preserving query grouping.

        Global dedup: If collection appears in multiple approaches, keep only in first.
        Within-approach dedup: If collection appears in multiple queries, keep only in first query.

        Args:
            approach_collections_by_query: {approach_idx: [[query0_colls], [query1_colls], ...]}

        Returns:
            Deduplicated structure with same format
        """
        global_seen_ids = set()
        deduplicated = {}

        for approach_idx in sorted(approach_collections_by_query.keys()):
            query_groups = approach_collections_by_query[approach_idx]
            deduplicated_groups = []
            approach_seen_ids = set()

            for query_collections in query_groups:
                deduplicated_query = []

                for collection in query_collections:
                    concept_id = collection.get("concept_id")

                    # Skip if seen globally or within this approach
                    if concept_id and concept_id not in global_seen_ids and concept_id not in approach_seen_ids:
                        global_seen_ids.add(concept_id)
                        approach_seen_ids.add(concept_id)
                        deduplicated_query.append(collection)

                # Keep empty lists to preserve query indices
                deduplicated_groups.append(deduplicated_query)

            deduplicated[approach_idx] = deduplicated_groups

        return deduplicated

    async def _filter_and_rank_by_approach(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        approach_collections_by_query: Dict[int, List[List[Dict[str, Any]]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[Any],
        run_id: str = None,
    ) -> tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Filter and rank collections within each approach in parallel.

        If LLM filtering fails for an approach, falls back to round-robin selection
        from query results to ensure no data is lost.

        Returns:
            Tuple of (filtered_by_approach, approach_fallbacks)
            - filtered_by_approach: Collections selected per approach
            - approach_fallbacks: List of fallback metadata for failed approaches
        """
        if not approach_collections:
            print("DEBUG: No approach_collections to filter")
            return {}, []

        print(f"DEBUG: Filtering {len(approach_collections)} approaches")
        for idx, colls in approach_collections.items():
            print(f"DEBUG:   Approach {idx}: {len(colls)} collections")

        # Get CMR prompts directory
        cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Create filtering tasks for each approach
        filtering_tasks = []

        for approach_idx in sorted(approach_collections.keys()):
            collections = approach_collections[approach_idx]

            if not collections:
                print(f"DEBUG: Approach {approach_idx} has no collections, skipping")
                continue

            # Get the corresponding QueryApproach
            if approach_idx >= len(query_approaches):
                print(f"DEBUG: No query approach for index {approach_idx}, skipping")
                continue

            approach = query_approaches[approach_idx]

            print(
                f"DEBUG: Creating filter input for approach {approach_idx} with {len(collections)} collections",
            )

            filter_input = CMRApproachCollectionFilteringInputSchema(
                original_query=original_query,
                topic_title=topic.title,
                topic_context=topic.functional_context,
                decomposition_title=decomp.title,
                decomposition_justification=decomp.scientific_justification,
                approach=approach,  # Pass whole approach object
                approach_search_string=None,  # Search strings are in searchable queries
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
                run_id=run_id,
            )

            task = filtering_component.arun(filter_input)
            filtering_tasks.append((approach_idx, collections, task))

        print(f"DEBUG: Created {len(filtering_tasks)} filtering tasks")

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

        print(f"DEBUG: Got {len(results)} filtering results")

        # Map results back to collections
        filtered_by_approach = {}
        approach_fallbacks = []  # Track fallback metadata

        for (approach_idx, collections, _), result in zip(filtering_tasks, results):
            if isinstance(result, Exception):
                print(
                    f"DEBUG: Approach {approach_idx} filtering FAILED with exception: {result}",
                )
                print(f"DEBUG: Using round-robin fallback for approach {approach_idx}")

                # Get query-grouped collections for this approach
                query_groups = approach_collections_by_query.get(approach_idx, [])
                fallback = self._round_robin_select(
                    query_groups,
                    max_items=self.config.max_collections_per_approach,
                )
                filtered_by_approach[approach_idx] = fallback

                # Log fallback metadata
                approach_fallbacks.append(
                    {
                        "approach_index": approach_idx,
                        "exception_type": type(result).__name__,
                        "exception_message": str(result),
                        "collections_before_fallback": len(collections),
                        "collections_returned": len(fallback),
                        "fallback_method": "round_robin_from_queries",
                    },
                )

                print(
                    f"DEBUG: Fallback selected {len(fallback)} collections via round-robin from {len(query_groups)} queries",
                )
                continue

            print(f"DEBUG: Approach {approach_idx} result type: {type(result)}")
            print(
                f"DEBUG: Approach {approach_idx} selected {len(result.selected_item_indexes)} collections",
            )

            # Log the LLM's reasoning for filtering
            print(f"DEBUG: Approach {approach_idx} LLM reasoning:")
            print(f"  {result.reasoning}")

            # Log selected collection indexes
            if result.selected_item_indexes:
                print(
                    f"DEBUG: Selected collection indexes: {result.selected_item_indexes}",
                )
            else:
                print(
                    f"DEBUG: No collections selected by LLM for approach {approach_idx}",
                )

            # Extract selected collections using the list of indexes
            selected = [collections[idx] for idx in result.selected_item_indexes if 0 <= idx < len(collections)]

            print(
                f"DEBUG: Approach {approach_idx} extracted {len(selected)} selected collections",
            )
            filtered_by_approach[approach_idx] = selected

        print(
            f"DEBUG: Final filtered_by_approach has {len(filtered_by_approach)} approaches",
        )
        for idx, colls in filtered_by_approach.items():
            print(f"DEBUG:   Approach {idx}: {len(colls)} collections after filtering")

        return filtered_by_approach, approach_fallbacks

    async def _rank_collections_with_llm_reranker(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        query_approaches: List[Any],
        run_id: str = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Rank collections using LLM reranker adapter.

        Args:
            approach_collections: Collections grouped by approach index
            original_query: Original user query
            query_approaches: List of query approach objects
            run_id: Run ID for logging

        Returns:
            Tuple of (ranked_collections, ranking_metadata)
        """
        logger.info(
            f"[{run_id}] Using LLM reranker adapter instead of legacy ranking system",
        )

        # Deduplicate approach collections and enrich with query approach info
        deduplicated_flat = {}
        for approach_idx, collections in approach_collections.items():
            seen_ids = set()
            deduped = []
            for coll in collections:
                col1_copy = coll.copy()
                concept_id = col1_copy.get("concept_id")
                if concept_id and concept_id not in seen_ids:
                    seen_ids.add(concept_id)

                    # Enrich collection with query approach information
                    if query_approaches and approach_idx < len(query_approaches):
                        approach = query_approaches[approach_idx]
                        col1_copy["query_approach_info"] = approach.dict()
                    deduped.append(col1_copy)
            deduplicated_flat[approach_idx] = deduped

        # Flatten all approaches into single list
        all_collections = []
        for approach_idx in sorted(deduplicated_flat.keys()):
            all_collections.extend(deduplicated_flat[approach_idx])

        print(
            f"DEBUG: LLM reranker - processing {len(all_collections)} deduplicated collections",
        )

        # Use adapter to rank collections
        ranked_collections = await self.reranker_adapter.rank_collections(
            collections=all_collections,
            query=original_query,
        )

        # Apply final collection limit if configured
        if self.config.apply_final_collection_limit:
            ranked_collections = ranked_collections[: self.config.final_collection_count]
            logger.info(
                f"[{run_id}] Applied final_collection_count limit: {len(ranked_collections)} collections",
            )

        print(
            f"DEBUG: LLM reranker - returning {len(ranked_collections)} ranked collections",
        )

        # Get metadata from adapter
        adapter_metadata = self.reranker_adapter.get_reranker_metadata()
        ranking_metadata = {
            **adapter_metadata,
            "total_before_reranking": len(all_collections),
            "total_after_reranking": len(ranked_collections),
            "final_limit_applied": self.config.apply_final_collection_limit,
        }

        logger.info(
            f"[{run_id}] LLM reranker processed {len(all_collections)} → {len(ranked_collections)} collections",
        )

        return ranked_collections, ranking_metadata

    async def _rank_collections_legacy(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        approach_collections_by_query: Dict[int, List[List[Dict[str, Any]]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[Any],
        run_id: str = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Rank collections using legacy approach-aware pipeline.

        Pipeline:
        1. Per-approach deduplication (preserving query grouping)
        2. Per-approach filtering and ranking (parallel, with round-robin fallback)
        3. Final cross-approach ranking (with round-robin fallback)

        Args:
            approach_collections: Collections grouped by approach index (flat lists)
            approach_collections_by_query: Collections grouped by approach and query
            original_query: Original user query
            topic: Topic object
            decomp: Scientific decomposition
            query_approaches: List of query approach objects
            run_id: Run ID for logging

        Returns:
            Tuple of (ranked_collections, ranking_metadata)
        """
        # Stage 1: Per-approach deduplication (preserving query grouping)
        deduplicated_grouped = self._deduplicate_approach_collections_grouped(
            approach_collections_by_query,
        )

        # Flatten grouped structure for normal filtering flow
        deduplicated_flat = {}
        for approach_idx, query_groups in deduplicated_grouped.items():
            deduplicated_flat[approach_idx] = []
            for query_colls in query_groups:
                deduplicated_flat[approach_idx].extend(query_colls)

        print(
            f"DEBUG: After deduplication: {sum(len(c) for c in deduplicated_flat.values())} total collections",
        )

        # Stage 2: Per-approach filtering and ranking (parallel, with fallback)
        (
            filtered_by_approach,
            approach_fallbacks,
        ) = await self._filter_and_rank_by_approach(
            deduplicated_flat,
            deduplicated_grouped,
            original_query,
            topic,
            decomp,
            query_approaches,
            run_id,
        )

        # Flatten all approach results into single list
        all_filtered = []
        for approach_idx in sorted(filtered_by_approach.keys()):
            all_filtered.extend(filtered_by_approach[approach_idx])

        print(
            f"DEBUG: After per-approach filtering: {len(all_filtered)} collections remain",
        )

        if not all_filtered:
            print("DEBUG: No collections passed filtering, returning empty list")
            # Build metadata even when empty
            ranking_metadata = {
                "approach_filtering": approach_fallbacks,
                "final_ranking": {"used": False},
            }
            return [], ranking_metadata

        # Get CMR prompts directory
        cmr_prompts_dir = Path(__file__).parent / "prompts"

        # Initialize final ranking fallback tracker
        final_fallback = None

        # Stage 3: Final cross-approach ranking
        # Determine max_items based on apply_final_collection_limit config
        if self.config.apply_final_collection_limit:
            max_items_for_ranking = min(len(all_filtered), self.config.final_collection_count)
        else:
            max_items_for_ranking = len(all_filtered)

        final_input = CMRFinalCollectionRankingInputSchema(
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomp.title,
            decomposition_justification=decomp.scientific_justification,
            data_items=all_filtered,  # Use base class field name
            max_items=max_items_for_ranking,  # Use base class field name
        )

        try:
            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.final_ranking_model,
            )
            final_ranking_component = CMRFinalCollectionRankingComponent(
                config=component_config,
                prompts_dir=cmr_prompts_dir,
                run_id=run_id,
            )

            final_result = await final_ranking_component.arun(final_input)

            # Map ranked indexes to full collection objects (already in ranked order)
            final_ranked = [
                all_filtered[idx] for idx in final_result.ranked_item_indexes if 0 <= idx < len(all_filtered)
            ]

            print(f"DEBUG: Final ranking returned {len(final_ranked)} collections")

            # Build metadata (no final ranking fallback used)
            ranking_metadata = {
                "approach_filtering": approach_fallbacks,
                "final_ranking": {"used": False},
                "final_limit_applied": self.config.apply_final_collection_limit,
            }
            return final_ranked, ranking_metadata

        except Exception as e:
            print(f"DEBUG: Final ranking FAILED with exception: {e}")
            print("DEBUG: Using round-robin fallback across approaches")

            # Build list of approach results in sorted order
            approach_lists = [filtered_by_approach[idx] for idx in sorted(filtered_by_approach.keys())]

            # Apply limit to fallback if configured
            if self.config.apply_final_collection_limit:
                max_fallback_items = self.config.final_collection_count
            else:
                max_fallback_items = sum(len(lst) for lst in approach_lists)

            fallback = self._round_robin_select(
                approach_lists,
                max_items=max_fallback_items,
            )

            print(
                f"DEBUG: Fallback selected {len(fallback)} collections via round-robin from {len(approach_lists)} approaches",
            )

            # Log final ranking fallback
            final_fallback = {
                "used": True,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "collections_before_fallback": len(all_filtered),
                "collections_returned": len(fallback),
                "fallback_method": "round_robin_from_approaches",
                "final_limit_applied": self.config.apply_final_collection_limit,
            }

            # Build metadata (with final ranking fallback)
            ranking_metadata = {
                "approach_filtering": approach_fallbacks,
                "final_ranking": final_fallback,
            }
            return fallback, ranking_metadata

    async def _rank_collections(
        self,
        approach_collections: Dict[int, List[Dict[str, Any]]],
        approach_collections_by_query: Dict[int, List[List[Dict[str, Any]]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[Any] = None,
        run_id: str = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Rank collections using approach-aware pipeline.

        Routes to either LLM reranker or legacy ranking system based on configuration.

        Returns:
            Tuple of (ranked_collections, ranking_metadata)
            - ranked_collections: Final ranked list of collections
            - ranking_metadata: Dictionary with fallback information for both stages
        """
        print(
            f"DEBUG: _rank_collections called with {len(approach_collections)} approaches",
        )
        if not approach_collections:
            print("DEBUG: No approach_collections, returning empty list")
            return [], {}

        # Early exit if ranking/filtering disabled
        if self.config.skip_ranking_and_filtering:
            logger.info(
                f"[{run_id}] Skipping ranking/filtering (skip_ranking_and_filtering=True) - "
                f"using all deduplicated collections from CMR pagination",
            )

            # Deduplicate approach collections (keep this step)
            deduplicated_flat = {}
            for approach_idx, collections in approach_collections.items():
                seen_ids = set()
                deduped = []
                for coll in collections:
                    concept_id = coll.get("concept_id")
                    if concept_id and concept_id not in seen_ids:
                        seen_ids.add(concept_id)
                        deduped.append(coll)
                deduplicated_flat[approach_idx] = deduped

            # Flatten all approaches into single list (no ranking)
            all_collections = []
            for approach_idx in sorted(deduplicated_flat.keys()):
                all_collections.extend(deduplicated_flat[approach_idx])

            # Return ALL deduplicated collections (no limit)
            ranking_metadata = {
                "approach_filtering": {
                    "enabled": False,
                    "reason": "skip_ranking_and_filtering=True",
                },
                "final_ranking": {
                    "enabled": False,
                    "reason": "skip_ranking_and_filtering=True",
                },
                "total_before_filtering": len(all_collections),
                "total_after_filtering": len(all_collections),
                "total_after_ranking": len(all_collections),
            }

            logger.info(
                f"[{run_id}] Returning {len(all_collections)} deduplicated collections (unranked)",
            )

            return all_collections, ranking_metadata

        # Route to LLM reranker or legacy system
        if self.config.use_llm_reranker and self.reranker_adapter:
            return await self._rank_collections_with_llm_reranker(
                approach_collections,
                original_query,
                query_approaches,
                run_id,
            )
        else:
            return await self._rank_collections_legacy(
                approach_collections,
                approach_collections_by_query,
                original_query,
                topic,
                decomp,
                query_approaches,
                run_id,
            )

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
