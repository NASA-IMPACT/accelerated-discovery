"""
PDS4 (Planetary Data System) Handler.

Implements repository-specific logic for searching NASA's PDS for planetary science data.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema, DecompositionResult
from akd.agents.data_search.components import ScientificDecomposition, Topic
from akd.tools.data_search import (
    PDS4BundleSearchTool,
    PDS4CollectionSearchTool,
    PDS4InvestigationSearchTool,
    PDS4TargetSearchTool,
)
from akd.utils.serialization import safe_model_dump, safe_model_dump_list

from .._base import BaseHandler
from .components import (
    PDS4FinalCollectionRankingComponent,
    PDS4ParameterExtractionComponent,
    PDS4StrategyCollectionFilteringComponent,
)
from .config import PDS4HandlerConfig
from .schemas import (
    PDS4FinalCollectionRankingInputSchema,
    PDS4StrategyCollectionFilteringInputSchema,
    PDS4ToolStrategy,
)


class PDS4Handler(BaseHandler):
    """
    Handler for NASA's Planetary Data System (PDS4).

    Implements the complete PDS4-specific pipeline:
    1. Extract unified tool strategies from decomposition
    2. Execute context searches (investigations/targets) to get URNs
    3. Execute collection searches using URN references
    4. Filter and rank collections per strategy
    5. Final cross-strategy ranking
    6. Return collections as data_results
    """

    def __init__(
        self,
        config: PDS4HandlerConfig,
        debug: bool = False,
        single_path_mode: bool = False,
    ):
        """Initialize PDS4 handler with configuration."""
        super().__init__(config, debug)
        self.single_path_mode = single_path_mode

        # Initialize PDS4-specific tools
        tool_config_params = {
            "mcp_endpoint": config.mcp_endpoint,
            "timeout_seconds": config.context_search_timeout,
            "debug": debug,
        }

        # Context search tools
        self.investigation_search_tool = PDS4InvestigationSearchTool.from_params(
            **tool_config_params,
        )
        self.target_search_tool = PDS4TargetSearchTool.from_params(
            **tool_config_params,
        )

        # Collection search tool
        collection_tool_config = tool_config_params.copy()
        collection_tool_config["timeout_seconds"] = config.collection_search_timeout

        self.collection_search_tool = PDS4CollectionSearchTool.from_params(
            **collection_tool_config,
        )

        # Bundle search tool (supplementary/exploratory)
        bundle_tool_config = tool_config_params.copy()
        bundle_tool_config["timeout_seconds"] = config.bundle_search_timeout

        self.bundle_search_tool = PDS4BundleSearchTool.from_params(
            **bundle_tool_config,
        )

        # Get PDS4 prompts directory
        self.pds4_prompts_dir = Path(__file__).parent / "prompts"

        # Store component config for creating per-call instances
        # This prevents race conditions when processing multiple decompositions in parallel
        self.parameter_extraction_config = BaseAgentConfig(
            model_name=config.parameter_extraction_model,
        )

    @property
    def parameter_extraction_component(self):
        """
        Create and return a fresh PDS4ParameterExtractionComponent instance.

        This property creates a new component instance each time it's accessed to avoid
        race conditions in parallel execution. Use for testing/demos only.

        Returns:
            Fresh PDS4ParameterExtractionComponent instance
        """
        return PDS4ParameterExtractionComponent(
            config=self.parameter_extraction_config,
            debug=self.debug,
            prompts_dir=self.pds4_prompts_dir,
        )

    async def process_decomposition(
        self,
        decomposition: ScientificDecomposition,
        topic: Topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
        run_id: str,
    ) -> DecompositionResult:
        """
        Process a scientific decomposition through PDS4 pipeline.

        Pipeline:
        1. Unified Parameter Extraction → Generate tool strategies
        2. Context Search → Execute investigation/target searches to get URNs
        3. Collection Search → Execute URN-filtered collection searches
        4. Collection Ranking → Filter and rank results
        5. Return collections as data_results
        """
        # Create fresh component instance for this decomposition to avoid race conditions
        parameter_extraction_component = PDS4ParameterExtractionComponent(
            config=self.parameter_extraction_config,
            debug=self.debug,
            prompts_dir=self.pds4_prompts_dir,
            run_id=run_id,
        )

        # Step 1: Unified Parameter Extraction
        parameter_output = await parameter_extraction_component.process(
            original_query,
            topic,
            decomposition,
        )

        # Select strategies based on execution mode
        if self.single_path_mode and parameter_output.tool_strategies:
            strategies_to_use = [parameter_output.tool_strategies[0]]
            if self.debug:
                logger.info(
                    f"Single-path mode: Using strategy[0], "
                    f"generated {len(parameter_output.tool_strategies)} total",
                )
        else:
            strategies_to_use = parameter_output.tool_strategies[
                : self.config.max_strategies
            ]

        # Step 2 & 3: Execute tool strategies (context search → URN extraction → collection search)
        (
            strategy_collections,
            strategy_execution_logs,
        ) = await self._execute_tool_strategies(
            strategies_to_use,
            params,
        )
        total_collections = sum(len(c) for c in strategy_collections.values())

        # Step 4: Collection Ranking & Filtering
        ranked_collections = await self._rank_collections(
            strategy_collections,
            original_query,
            topic,
            decomposition,
            parameter_output.tool_strategies,
            run_id,
        )

        if self.debug:
            logger.info(
                f"Ranked {len(ranked_collections)} collections from {total_collections} total",
            )

        # Augment strategies with execution metadata
        augmented_strategies, total_pds4 = self._augment_strategies_with_execution_metadata(
            parameter_output.tool_strategies,
            strategy_execution_logs,
        )

        result = DecompositionResult(
            decomposition=safe_model_dump(decomposition),
            repository="PDS4",
            query_approaches=augmented_strategies,  # Store PDS4 tool strategies here
            searchable_queries=[],  # PDS4 doesn't use this field
            data_results=ranked_collections,
            total_results_from_repository=total_pds4,
            total_results_after_filtering=len(ranked_collections),
            note=None,
        )

        if self.debug:
            logger.info(
                f"DecompositionResult created with {len(result.data_results)} data_results",
            )

        return result

    async def _execute_context_search(
        self,
        strategy: PDS4ToolStrategy,
        search_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute context search (investigation or target) to get URN references.

        Args:
            strategy: Tool strategy with search parameters
            search_type: Either "investigation" or "target"

        Returns:
            List of context items (investigations or targets)
        """
        try:
            if search_type == "investigation":
                # Search investigations
                search_params = strategy.get_context_search_params()
                if not search_params.get("keywords"):
                    return []

                tool_input = self.investigation_search_tool.input_schema(
                    keywords=search_params["keywords"],
                    limit=self.config.context_search_page_size,
                )
                result = await self.investigation_search_tool.arun(tool_input)

                if hasattr(result, "investigations") and result.investigations:
                    return result.investigations[: self.config.context_search_page_size]

            elif search_type == "target":
                # Search targets
                search_params = strategy.get_context_search_params()
                if not search_params.get("keywords"):
                    return []

                tool_input = self.target_search_tool.input_schema(
                    keywords=search_params["keywords"],
                    target_type=strategy.target_type.value if strategy.target_type else "",
                    limit=self.config.context_search_page_size,
                )
                result = await self.target_search_tool.arun(tool_input)

                if hasattr(result, "targets") and result.targets:
                    return result.targets[: self.config.context_search_page_size]

            return []

        except Exception as e:
            if self.debug:
                logger.error(f"Context search ({search_type}) failed: {e}")
            return []

    def _extract_urns_from_context(
        self,
        context_items: List[Dict[str, Any]],
        context_type: str,
    ) -> List[str]:
        """
        Extract URN identifiers from context search results.

        Args:
            context_items: Results from investigation or target search
            context_type: Either "investigation" or "target"

        Returns:
            List of URN identifiers (e.g., "urn:nasa:pds:context:investigation:mission.mars2020")
        """
        urns = []
        for item in context_items:
            # PDS4 context products typically have lidvid or id fields
            urn = item.get("lidvid") or item.get("id")
            if urn:
                urns.append(urn)
        return urns

    async def _execute_single_strategy(
        self,
        strategy: PDS4ToolStrategy,
        params: DataSearchAgentInputSchema,
        strategy_idx: int,
    ) -> Dict[str, Any]:
        """
        Execute a single tool strategy and return results.

        Workflow:
        1. Execute context search (investigation/target) if needed
        2. Extract URN references
        3. Execute collection search with URN filters
        4. Return collections with strategy metadata

        Args:
            strategy: Tool strategy to execute
            params: Search parameters
            strategy_idx: Index of the strategy

        Returns:
            Dictionary with strategy_idx, collections, and execution metadata
        """
        execution_metadata = {
            "context_searches_executed": [],
            "urns_extracted": {},
            "collection_search_parameters": {},
            "collections_returned": 0,
        }

        investigation_urns = []
        target_urns = []

        try:
            # Step 1: Execute context searches if needed
            if strategy.target_context == "investigation":
                investigations = await self._execute_context_search(
                    strategy,
                    "investigation",
                )
                investigation_urns = self._extract_urns_from_context(
                    investigations,
                    "investigation",
                )
                execution_metadata["context_searches_executed"].append("investigation")
                execution_metadata["urns_extracted"]["investigation"] = investigation_urns

            elif strategy.target_context == "target":
                targets = await self._execute_context_search(strategy, "target")
                target_urns = self._extract_urns_from_context(targets, "target")
                execution_metadata["context_searches_executed"].append("target")
                execution_metadata["urns_extracted"]["target"] = target_urns

            # Store extracted URNs back in strategy for downstream use (filtering, ranking)
            if investigation_urns:
                strategy.investigation_urn = investigation_urns[0]
            if target_urns:
                strategy.target_urn = target_urns[0]

            # Step 2: Execute collection search with URN filters
            collection_params = {
                "ref_lid_investigation": investigation_urns[0] if investigation_urns else "",
                "ref_lid_target": target_urns[0] if target_urns else "",
                "ref_lid_instrument": "",
                "ref_lid_instrument_host": "",
                "limit": self.config.collection_search_page_size,
            }

            execution_metadata["collection_search_parameters"] = collection_params

            # Execute collection search
            tool_input = self.collection_search_tool.input_schema(**collection_params)
            result = await self.collection_search_tool.arun(tool_input)

            # Extract collections
            collections = []
            if hasattr(result, "collections") and result.collections:
                total_from_pds4 = len(result.collections)
                collections = result.collections[: self.config.collections_per_strategy]
                execution_metadata["collections_returned"] = total_from_pds4

            if self.debug:
                logger.info(
                    f"Strategy {strategy_idx} returned {len(collections)} collections",
                )

            return {
                "strategy_idx": strategy_idx,
                "collections": collections,
                "strategy_object": strategy,
                "execution_metadata": execution_metadata,
            }

        except Exception as e:
            if self.debug:
                logger.error(f"Strategy {strategy_idx} execution failed: {e}")

            return {
                "strategy_idx": strategy_idx,
                "collections": [],
                "strategy_object": strategy,
                "execution_metadata": execution_metadata,
            }

    async def _execute_tool_strategies(
        self,
        tool_strategies: List[PDS4ToolStrategy],
        params: DataSearchAgentInputSchema,
    ) -> tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Execute tool strategies in parallel and return collections grouped by strategy.

        Args:
            tool_strategies: Strategies to execute
            params: Search parameters

        Returns:
            Tuple of (strategy_collections, execution_logs)
            - strategy_collections: Dictionary mapping strategy_index to list of collections
            - execution_logs: List of execution metadata for each strategy
        """
        if self.config.enable_parallel_search:
            # Create parallel tasks for all strategies
            strategy_tasks = []
            for strategy in tool_strategies:
                task = self._execute_single_strategy(
                    strategy,
                    params,
                    strategy.strategy_index,
                )
                strategy_tasks.append(task)

            # Execute all strategies in parallel
            results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
        else:
            # Execute sequentially
            results = []
            for strategy in tool_strategies:
                result = await self._execute_single_strategy(
                    strategy,
                    params,
                    strategy.strategy_index,
                )
                results.append(result)

        # Group results by strategy and collect execution logs
        strategy_collections = {}  # strategy_index -> [collections]
        execution_logs = []

        for result in results:
            if isinstance(result, Exception):
                if self.debug:
                    logger.error(f"Strategy execution failed: {result}")
                continue

            strategy_idx = result["strategy_idx"]
            collections = result["collections"]

            if strategy_idx not in strategy_collections:
                strategy_collections[strategy_idx] = []
            strategy_collections[strategy_idx].extend(collections)

            # Collect execution metadata
            execution_logs.append(
                {
                    "strategy": result["strategy_object"],
                    "metadata": result["execution_metadata"],
                },
            )

        return strategy_collections, execution_logs

    def _augment_strategies_with_execution_metadata(
        self,
        tool_strategies: List[PDS4ToolStrategy],
        execution_logs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Augment tool strategy dicts with execution metadata.

        Args:
            tool_strategies: Original strategy objects
            execution_logs: Execution metadata from _execute_tool_strategies

        Returns:
            Tuple of (augmented_strategies, total_pds4_results)
            - augmented_strategies: List of strategy dicts with execution metadata added
            - total_pds4_results: Sum of collections_returned across all strategies
        """
        augmented = []
        total_pds4 = 0

        for strategy in tool_strategies:
            strategy_dict = safe_model_dump(strategy)

            # Find matching execution log
            matching_log = next(
                (log for log in execution_logs if log["strategy"] == strategy),
                None,
            )

            if matching_log:
                metadata = matching_log["metadata"]
                strategy_dict["execution_metadata"] = metadata
                total_pds4 += metadata["collections_returned"]

            augmented.append(strategy_dict)

        return augmented, total_pds4

    def _deduplicate_strategy_collections(
        self,
        strategy_collections: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Deduplicate collections within each strategy, preserving first occurrence.
        If a collection appears in multiple strategies, keep it only in the first.
        """
        global_seen_ids = set()  # Track across all strategies
        deduplicated_by_strategy = {}

        # Process strategies in order (0, 1, 2, ...)
        for strategy_idx in sorted(strategy_collections.keys()):
            collections = strategy_collections[strategy_idx]
            deduplicated = []

            for collection in collections:
                lidvid = collection.get("lidvid") or collection.get("id")
                if lidvid and lidvid not in global_seen_ids:
                    global_seen_ids.add(lidvid)
                    deduplicated.append(collection)

            deduplicated_by_strategy[strategy_idx] = deduplicated

        return deduplicated_by_strategy

    async def _filter_and_rank_by_strategy(
        self,
        strategy_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        tool_strategies: List[PDS4ToolStrategy],
        run_id: Optional[str] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Filter and rank collections within each strategy in parallel.
        """
        if not strategy_collections:
            if self.debug:
                logger.debug("No strategy_collections to filter")
            return {}

        if self.debug:
            logger.info(f"Filtering {len(strategy_collections)} strategies")
            for idx, colls in strategy_collections.items():
                logger.debug(f"  Strategy {idx}: {len(colls)} collections")

        # Create filtering tasks for each strategy
        filtering_tasks = []

        for strategy_idx in sorted(strategy_collections.keys()):
            collections = strategy_collections[strategy_idx]

            if not collections:
                if self.debug:
                    logger.debug(f"Strategy {strategy_idx} has no collections, skipping")
                continue

            # Get the corresponding ToolStrategy
            if strategy_idx >= len(tool_strategies):
                if self.debug:
                    logger.debug(f"No tool strategy for index {strategy_idx}, skipping")
                continue

            strategy = tool_strategies[strategy_idx]

            if self.debug:
                logger.debug(
                    f"Creating filter input for strategy {strategy_idx} with {len(collections)} collections",
                )

            filter_input = PDS4StrategyCollectionFilteringInputSchema(
                original_query=original_query,
                topic_title=topic.title,
                topic_context=topic.functional_context,
                decomposition_title=decomp.title,
                decomposition_justification=decomp.scientific_justification,
                strategy_description=strategy.strategy_description,
                target_context=strategy.target_context,
                target_type=strategy.target_type.value if strategy.target_type else None,
                mission_keywords=strategy.mission_keywords,
                investigation_urn=strategy.investigation_urn,
                target_urn=strategy.target_urn,
                data_items=collections,
                max_items=self.config.max_collections_per_strategy,
            )

            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.strategy_filtering_model,
            )
            filtering_component = PDS4StrategyCollectionFilteringComponent(
                config=component_config,
                prompts_dir=self.pds4_prompts_dir,
                run_id=run_id,
            )

            task = filtering_component.arun(filter_input)
            filtering_tasks.append((strategy_idx, collections, task))

        if self.debug:
            logger.debug(f"Created {len(filtering_tasks)} filtering tasks")

        # Execute in parallel
        if len(filtering_tasks) > 1 and self.config.enable_parallel_search:
            results = await asyncio.gather(
                *[task for _, _, task in filtering_tasks],
                return_exceptions=True,
            )
        else:
            results = []
            for _, _, task in filtering_tasks:
                result = await task
                results.append(result)

        if self.debug:
            logger.debug(f"Got {len(results)} filtering results")

        # Map results back to collections
        filtered_by_strategy = {}

        for (strategy_idx, collections, _), result in zip(filtering_tasks, results):
            if isinstance(result, Exception):
                if self.debug:
                    logger.error(f"Strategy {strategy_idx} filtering failed: {result}")
                continue

            if self.debug:
                logger.info(
                    f"Strategy {strategy_idx} selected {len(result.selected_item_indexes)} collections",
                )
                logger.debug(f"Strategy {strategy_idx} LLM reasoning: {result.reasoning}")

            # Extract selected collections using the list of indexes
            selected = [
                collections[idx]
                for idx in result.selected_item_indexes
                if 0 <= idx < len(collections)
            ]

            filtered_by_strategy[strategy_idx] = selected

        if self.debug:
            logger.info(
                f"Final filtered_by_strategy has {len(filtered_by_strategy)} strategies",
            )
            for idx, colls in filtered_by_strategy.items():
                logger.debug(
                    f"  Strategy {idx}: {len(colls)} collections after filtering",
                )

        return filtered_by_strategy

    async def _rank_collections(
        self,
        strategy_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        tool_strategies: List[PDS4ToolStrategy] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank collections using strategy-aware pipeline.

        Pipeline:
        1. Per-strategy deduplication
        2. Per-strategy filtering and ranking (parallel)
        3. Final cross-strategy ranking
        """
        if self.debug:
            logger.info(
                f"Ranking collections from {len(strategy_collections)} strategies",
            )

        if not strategy_collections:
            return []

        # Stage 1: Per-strategy deduplication
        deduplicated = self._deduplicate_strategy_collections(strategy_collections)

        if self.debug:
            total_after_dedup = sum(len(c) for c in deduplicated.values())
            logger.debug(
                f"After deduplication: {total_after_dedup} total collections",
            )

        # Stage 2: Per-strategy filtering and ranking (parallel)
        filtered_by_strategy = await self._filter_and_rank_by_strategy(
            deduplicated,
            original_query,
            topic,
            decomp,
            tool_strategies,
            run_id,
        )

        # Flatten all strategy results into single list
        all_filtered = []
        for strategy_idx in sorted(filtered_by_strategy.keys()):
            all_filtered.extend(filtered_by_strategy[strategy_idx])

        if self.debug:
            logger.info(
                f"After per-strategy filtering: {len(all_filtered)} collections remain",
            )

        if not all_filtered:
            return []

        # Stage 3: Final cross-strategy ranking
        final_input = PDS4FinalCollectionRankingInputSchema(
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomp.title,
            decomposition_justification=decomp.scientific_justification,
            data_items=all_filtered,
            max_items=min(
                len(all_filtered),
                self.config.final_collection_count,
            ),
        )

        try:
            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.final_ranking_model,
            )
            final_ranking_component = PDS4FinalCollectionRankingComponent(
                config=component_config,
                prompts_dir=self.pds4_prompts_dir,
                run_id=run_id,
            )

            final_result = await final_ranking_component.arun(final_input)

            # Map ranked indexes to full collection objects (already in ranked order)
            final_ranked = [
                all_filtered[idx]
                for idx in final_result.ranked_item_indexes
                if 0 <= idx < len(all_filtered)
            ]

            if self.debug:
                logger.info(f"Final ranking returned {len(final_ranked)} collections")

            return final_ranked

        except Exception as e:
            if self.debug:
                logger.error(f"Final ranking failed: {e}")

            # Fallback: return up to final_collection_count
            fallback = all_filtered[: self.config.final_collection_count]

            if self.debug:
                logger.warning(f"Returning fallback of {len(fallback)} collections")

            return fallback
