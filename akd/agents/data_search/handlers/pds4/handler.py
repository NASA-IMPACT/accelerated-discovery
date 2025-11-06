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
    PDS4ApproachCollectionFilteringComponent,
    PDS4FinalCollectionRankingComponent,
    PDS4ParameterExtractionComponent,
)
from .config import PDS4HandlerConfig
from .schemas import (
    PDS4ApproachCollectionFilteringInputSchema,
    PDS4FinalCollectionRankingInputSchema,
    PDS4QueryApproach,
)


class PDS4Handler(BaseHandler):
    """
    Handler for NASA's Planetary Data System (PDS4).

    Implements the complete PDS4-specific pipeline:
    1. Extract unified tool strategies from decomposition
    2. Execute context searches (investigations/targets) to get URNs
    3. Execute collection searches using URN references
    4. Filter and rank collections per approach
    5. Final cross-approach ranking
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
        if self.single_path_mode and parameter_output.query_approaches:
            approaches_to_use = [parameter_output.query_approaches[0]]
            if self.debug:
                logger.info(
                    f"Single-path mode: Using strategy[0], "
                    f"generated {len(parameter_output.query_approaches)} total",
                )
        else:
            approaches_to_use = parameter_output.query_approaches[
                : self.config.max_strategies
            ]

        # Step 2 & 3: Execute tool strategies (context search → URN extraction → collection search)
        (
            strategy_collections,
            strategy_execution_logs,
        ) = await self._execute_query_approaches(
            approaches_to_use,
            params,
        )
        total_collections = sum(len(c) for c in strategy_collections.values())

        # Step 4: Collection Ranking & Filtering
        ranked_collections = await self._rank_collections(
            strategy_collections,
            original_query,
            topic,
            decomposition,
            parameter_output.query_approaches,
            run_id,
        )

        if self.debug:
            logger.info(
                f"Ranked {len(ranked_collections)} collections from {total_collections} total",
            )

        # Augment strategies with execution metadata
        augmented_strategies, total_pds4 = self._augment_strategies_with_execution_metadata(
            parameter_output.query_approaches,
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
        approach: PDS4QueryApproach,
        search_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute context search (investigation or target) to get URN references.

        Args:
            approach: Query approach with search parameters
            search_type: Either "investigation" or "target"

        Returns:
            List of context items (investigations or targets)
        """
        try:
            if search_type == "investigation":
                # Search investigations
                search_params = approach.get_context_search_params()
                investigation_params = search_params.get("investigation_params")

                if not investigation_params or not investigation_params.get("keywords"):
                    return []

                tool_input = self.investigation_search_tool.input_schema(
                    keywords=investigation_params["keywords"],
                    limit=self.config.context_search_page_size,
                )
                result = await self.investigation_search_tool.arun(tool_input)

                if hasattr(result, "investigations") and result.investigations:
                    return result.investigations[: self.config.context_search_page_size]

            elif search_type == "target":
                # Search targets
                search_params = approach.get_context_search_params()
                target_params = search_params.get("target_params")

                if not target_params or not target_params.get("keywords"):
                    return []

                tool_input = self.target_search_tool.input_schema(
                    keywords=target_params["keywords"],
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
            # PDS4 context products use 'lid' field for identifiers
            # Try lid first (standard), then lidvid (versioned), then id (generic)
            urn = item.get("lid") or item.get("lidvid") or item.get("id")
            if urn:
                urns.append(urn)
        return urns

    def _filter_urns_by_type(
        self,
        urns: List[str],
        context_items: List[Dict[str, Any]],
        context_type: str,
    ) -> List[str]:
        """
        Filter URNs to exclude inappropriate types.

        For target URNs, excludes:
        - laboratory_analog: Lab samples, not celestial bodies
        - equipment: Instruments/hardware, not observation targets
        - calibrator: Calibration targets, not science targets

        For other context types (investigation, instrument), no filtering applied.

        Args:
            urns: List of URN identifiers to filter
            context_items: Original context search results (for metadata access)
            context_type: "investigation", "target", "instrument", etc.

        Returns:
            Filtered list of URN identifiers
        """
        if context_type != "target":
            # Only filter target URNs for now
            return urns

        # Target types to exclude
        excluded_types = {"laboratory_analog", "equipment", "calibrator"}

        # Build URN-to-item mapping for metadata lookup
        urn_to_item = {}
        for item in context_items:
            item_urn = item.get("lid") or item.get("lidvid") or item.get("id")
            if item_urn:
                urn_to_item[item_urn] = item

        filtered_urns = []
        for urn in urns:
            # Extract type from URN: "urn:nasa:pds:context:target:planet.mars" → "planet"
            urn_parts = urn.split(":")
            if len(urn_parts) >= 6:
                # Format: urn:nasa:pds:context:target:TYPE.name
                target_identifier = urn_parts[5]  # e.g., "planet.mars"
                target_type = target_identifier.split(".")[0]  # e.g., "planet"

                # Exclude unwanted types
                if target_type.lower() not in excluded_types:
                    filtered_urns.append(urn)
                elif self.debug:
                    logger.debug(f"Filtered out target URN (type={target_type}): {urn}")
            else:
                # Malformed URN, keep it (let downstream handle)
                filtered_urns.append(urn)

        return filtered_urns

    def _rank_urns_by_relevance(
        self,
        urns: List[str],
        context_items: List[Dict[str, Any]],
        keywords: List[str],
    ) -> List[str]:
        """
        Rank URNs by relevance to search keywords.

        Scoring system:
        - Exact keyword match in title: +10 points per keyword
        - Partial keyword match in title: +5 points per keyword
        - Keyword in description: +2 points per keyword

        Args:
            urns: List of URN identifiers to rank
            context_items: Original context search results with metadata
            keywords: Search keywords to match against

        Returns:
            URNs sorted by descending relevance score
        """
        if not urns or not keywords:
            return urns

        # Build URN-to-item mapping for metadata access
        urn_to_item = {}
        for item in context_items:
            item_urn = item.get("lid") or item.get("lidvid") or item.get("id")
            if item_urn:
                urn_to_item[item_urn] = item

        # Normalize keywords for matching
        normalized_keywords = [kw.lower().strip() for kw in keywords]

        # Score each URN
        urn_scores = []
        for urn in urns:
            item = urn_to_item.get(urn)
            if not item:
                # No metadata, assign neutral score
                urn_scores.append((urn, 0))
                continue

            score = 0

            # Get title and description
            title = item.get("title", "").lower()
            description = ""

            # Extract description based on context type
            if "investigation" in item:
                description = item.get("investigation", {}).get("description", "").lower()
            elif "target" in item:
                description = item.get("target", {}).get("description", "").lower()
            elif "instrument" in item:
                description = item.get("instrument", {}).get("description", "").lower()

            # Score based on keyword matches
            for keyword in normalized_keywords:
                if not keyword:
                    continue

                # Exact match in title
                if keyword == title:
                    score += 10
                # Partial match in title
                elif keyword in title:
                    score += 5

                # Match in description
                if keyword in description:
                    score += 2

            urn_scores.append((urn, score))

        # Sort by score (descending), then by original order (stable)
        urn_scores.sort(key=lambda x: x[1], reverse=True)

        # Return sorted URNs
        ranked_urns = [urn for urn, score in urn_scores]

        if self.debug:
            logger.debug(f"URN ranking scores: {[(urn.split(':')[-1], score) for urn, score in urn_scores[:5]]}")

        return ranked_urns

    def _generate_urn_combinations(
        self,
        investigation_urns: List[str],
        target_urns: List[str],
    ) -> List[Dict[str, str]]:
        """
        Generate URN combinations for collection searches.

        Creates all possible combinations of investigation and target URNs,
        respecting configuration limits.

        Args:
            investigation_urns: Ranked list of investigation URNs
            target_urns: Ranked list of target URNs

        Returns:
            List of parameter dictionaries for collection searches, each containing:
            - ref_lid_investigation: Investigation URN (or "")
            - ref_lid_target: Target URN (or "")
        """
        combinations = []

        # Limit URNs per type based on configuration
        inv_limited = investigation_urns[: self.config.max_investigation_urns_per_approach]
        tgt_limited = target_urns[: self.config.max_target_urns_per_approach]

        # If we have both investigations and targets, create all combinations
        if inv_limited and tgt_limited:
            for inv_urn in inv_limited:
                for tgt_urn in tgt_limited:
                    combinations.append(
                        {
                            "ref_lid_investigation": inv_urn,
                            "ref_lid_target": tgt_urn,
                        }
                    )
        # If only investigations (no targets), search by investigation alone
        elif inv_limited:
            for inv_urn in inv_limited:
                combinations.append(
                    {
                        "ref_lid_investigation": inv_urn,
                        "ref_lid_target": "",
                    }
                )
        # If only targets (no investigations), search by target alone
        elif tgt_limited:
            for tgt_urn in tgt_limited:
                combinations.append(
                    {
                        "ref_lid_investigation": "",
                        "ref_lid_target": tgt_urn,
                    }
                )
        # Fallback: no context URNs (shouldn't happen, but handle gracefully)
        else:
            combinations.append(
                {
                    "ref_lid_investigation": "",
                    "ref_lid_target": "",
                }
            )

        # Cap total combinations based on configuration
        combinations = combinations[: self.config.max_approach_combinations]

        if self.debug:
            logger.debug(
                f"Generated {len(combinations)} URN combinations "
                f"({len(inv_limited)} investigations × {len(tgt_limited)} targets)"
            )

        return combinations

    async def _execute_single_strategy(
        self,
        approach: PDS4QueryApproach,
        params: DataSearchAgentInputSchema,
        approach_idx: int,
    ) -> Dict[str, Any]:
        """
        Execute a single query approach and return results.

        Workflow:
        1. Execute context search (investigation/target) if needed
        2. Extract URN references
        3. Execute collection search with URN filters
        4. Return collections with approach metadata

        Args:
            approach: Query approach to execute
            params: Search parameters
            approach_idx: Index of the approach

        Returns:
            Dictionary with approach_idx, collections, and execution metadata
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
            investigations = []
            targets = []

            # Execute investigation search if approach has investigation keywords
            if approach.investigation_keywords:
                investigations = await self._execute_context_search(
                    approach,
                    "investigation",
                )
                investigation_urns_raw = self._extract_urns_from_context(
                    investigations,
                    "investigation",
                )
                # Rank by relevance to keywords
                investigation_urns = self._rank_urns_by_relevance(
                    investigation_urns_raw,
                    investigations,
                    approach.investigation_keywords,
                )
                execution_metadata["context_searches_executed"].append("investigation")
                execution_metadata["urns_extracted"]["investigation"] = investigation_urns

            # Execute target search if approach has target keywords
            if approach.target_keywords:
                targets = await self._execute_context_search(approach, "target")
                target_urns_raw = self._extract_urns_from_context(targets, "target")
                # Filter out inappropriate types (laboratory_analog, equipment, etc.)
                target_urns_filtered = self._filter_urns_by_type(
                    target_urns_raw,
                    targets,
                    "target",
                )
                # Rank by relevance to keywords
                target_urns = self._rank_urns_by_relevance(
                    target_urns_filtered,
                    targets,
                    approach.target_keywords,
                )
                execution_metadata["context_searches_executed"].append("target")
                execution_metadata["urns_extracted"]["target"] = target_urns

            # Store first URN back in approach for backward compatibility
            if investigation_urns:
                approach.investigation_urn = investigation_urns[0]
            if target_urns:
                approach.target_urn = target_urns[0]

            # Step 2: Generate URN combinations and execute collection searches
            urn_combinations = self._generate_urn_combinations(
                investigation_urns,
                target_urns,
            )

            # Track combination-specific metadata
            execution_metadata["urn_combinations_generated"] = len(urn_combinations)
            execution_metadata["urn_combinations"] = []

            # Execute collection search for each URN combination
            all_collections = []
            seen_lidvids = set()  # For deduplication

            for combo_idx, combo_params in enumerate(urn_combinations):
                # Add standard parameters
                collection_params = {
                    **combo_params,
                    "ref_lid_instrument": "",
                    "ref_lid_instrument_host": "",
                    "limit": self.config.collection_search_page_size,
                }

                # Store first combination parameters for backward compatibility
                if combo_idx == 0:
                    execution_metadata["collection_search_parameters"] = collection_params

                try:
                    # Execute collection search
                    tool_input = self.collection_search_tool.input_schema(**collection_params)
                    result = await self.collection_search_tool.arun(tool_input)

                    # Extract and deduplicate collections
                    combo_collections = []
                    if hasattr(result, "collections") and result.collections:
                        for collection in result.collections:
                            # Deduplicate by lidvid
                            lidvid = collection.get("lidvid") or collection.get("lid", "")
                            if lidvid and lidvid not in seen_lidvids:
                                seen_lidvids.add(lidvid)
                                combo_collections.append(collection)
                                all_collections.append(collection)

                    # Track this combination
                    execution_metadata["urn_combinations"].append({
                        "investigation_urn": combo_params.get("ref_lid_investigation", ""),
                        "target_urn": combo_params.get("ref_lid_target", ""),
                        "collections_returned": len(combo_collections),
                    })

                    if self.debug and combo_collections:
                        logger.debug(
                            f"  Combo {combo_idx+1}/{len(urn_combinations)}: "
                            f"{len(combo_collections)} collections"
                        )

                except Exception as e:
                    if self.debug:
                        logger.warning(f"  Combo {combo_idx+1} failed: {e}")
                    execution_metadata["urn_combinations"].append({
                        "investigation_urn": combo_params.get("ref_lid_investigation", ""),
                        "target_urn": combo_params.get("ref_lid_target", ""),
                        "collections_returned": 0,
                        "error": str(e),
                    })

            # Limit total collections per approach
            collections = all_collections[: self.config.collections_per_strategy]
            execution_metadata["collections_returned"] = len(all_collections)
            execution_metadata["collections_after_deduplication"] = len(collections)

            if self.debug:
                logger.info(
                    f"Approach {approach_idx} returned {len(collections)} collections",
                )

            return {
                "approach_idx": approach_idx,
                "collections": collections,
                "strategy_object": approach,
                "execution_metadata": execution_metadata,
            }

        except Exception as e:
            if self.debug:
                logger.error(f"Approach {approach_idx} execution failed: {e}")

            return {
                "approach_idx": approach_idx,
                "collections": [],
                "strategy_object": approach,
                "execution_metadata": execution_metadata,
            }

    async def _execute_query_approaches(
        self,
        query_approaches: List[PDS4QueryApproach],
        params: DataSearchAgentInputSchema,
    ) -> tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Execute query approaches in parallel and return collections grouped by approach.

        Args:
            query_approaches: Approaches to execute
            params: Search parameters

        Returns:
            Tuple of (strategy_collections, execution_logs)
            - strategy_collections: Dictionary mapping approach_index to list of collections
            - execution_logs: List of execution metadata for each approach
        """
        if self.config.enable_parallel_search:
            # Create parallel tasks for all approaches
            approach_tasks = []
            for approach in query_approaches:
                task = self._execute_single_strategy(
                    approach,
                    params,
                    approach.approach_index,
                )
                approach_tasks.append(task)

            # Execute all approaches in parallel
            results = await asyncio.gather(*approach_tasks, return_exceptions=True)
        else:
            # Execute sequentially
            results = []
            for approach in query_approaches:
                result = await self._execute_single_strategy(
                    approach,
                    params,
                    approach.approach_index,
                )
                results.append(result)

        # Group results by approach and collect execution logs
        strategy_collections = {}  # approach_index -> [collections]
        execution_logs = []

        for result in results:
            if isinstance(result, Exception):
                if self.debug:
                    logger.error(f"Strategy execution failed: {result}")
                continue

            approach_idx = result["approach_idx"]
            collections = result["collections"]

            if approach_idx not in strategy_collections:
                strategy_collections[approach_idx] = []
            strategy_collections[approach_idx].extend(collections)

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
        query_approaches: List[PDS4QueryApproach],
        execution_logs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Augment tool strategy dicts with execution metadata.

        Args:
            query_approaches: Original strategy objects
            execution_logs: Execution metadata from _execute_query_approaches

        Returns:
            Tuple of (augmented_strategies, total_pds4_results)
            - augmented_strategies: List of strategy dicts with execution metadata added
            - total_pds4_results: Sum of collections_returned across all strategies
        """
        augmented = []
        total_pds4 = 0

        for approach in query_approaches:
            approach_dict = safe_model_dump(approach)

            # Find matching execution log
            matching_log = next(
                (log for log in execution_logs if log["strategy"] == approach),
                None,
            )

            if matching_log:
                metadata = matching_log["metadata"]
                approach_dict["execution_metadata"] = metadata
                total_pds4 += metadata["collections_returned"]

            augmented.append(approach_dict)

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
        for approach_idx in sorted(strategy_collections.keys()):
            collections = strategy_collections[approach_idx]
            deduplicated = []

            for collection in collections:
                lidvid = collection.get("lidvid") or collection.get("id")
                if lidvid and lidvid not in global_seen_ids:
                    global_seen_ids.add(lidvid)
                    deduplicated.append(collection)

            deduplicated_by_strategy[approach_idx] = deduplicated

        return deduplicated_by_strategy

    async def _filter_and_rank_by_strategy(
        self,
        strategy_collections: Dict[int, List[Dict[str, Any]]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
        query_approaches: List[PDS4QueryApproach],
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

        for approach_idx in sorted(strategy_collections.keys()):
            collections = strategy_collections[approach_idx]

            if not collections:
                if self.debug:
                    logger.debug(f"Approach {approach_idx} has no collections, skipping")
                continue

            # Get the corresponding QueryApproach
            if approach_idx >= len(query_approaches):
                if self.debug:
                    logger.debug(f"No query approach for index {approach_idx}, skipping")
                continue

            approach = query_approaches[approach_idx]

            if self.debug:
                logger.debug(
                    f"Creating filter input for approach {approach_idx} with {len(collections)} collections",
                )

            filter_input = PDS4ApproachCollectionFilteringInputSchema(
                original_query=original_query,
                topic_title=topic.title,
                topic_context=topic.functional_context,
                decomposition_title=decomp.title,
                decomposition_justification=decomp.scientific_justification,
                strategy_description=approach.approach_description,
                investigation_keywords=approach.investigation_keywords,
                target_keywords=approach.target_keywords,
                instrument_keywords=approach.instrument_keywords,
                instrument_host_keywords=approach.instrument_host_keywords,
                temporal_context=approach.temporal_context,
                investigation_urn=approach.investigation_urn,
                target_urn=approach.target_urn,
                instrument_urn=approach.instrument_urn,
                instrument_host_urn=approach.instrument_host_urn,
                data_items=collections,
                max_items=self.config.max_collections_per_strategy,
            )

            # Initialize component with configured model and prompts_dir
            component_config = BaseAgentConfig(
                model_name=self.config.strategy_filtering_model,
            )
            filtering_component = PDS4ApproachCollectionFilteringComponent(
                config=component_config,
                prompts_dir=self.pds4_prompts_dir,
                run_id=run_id,
            )

            task = filtering_component.arun(filter_input)
            filtering_tasks.append((approach_idx, collections, task))

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

        for (approach_idx, collections, _), result in zip(filtering_tasks, results):
            if isinstance(result, Exception):
                if self.debug:
                    logger.error(f"Strategy {approach_idx} filtering failed: {result}")
                continue

            if self.debug:
                logger.info(
                    f"Strategy {approach_idx} selected {len(result.selected_item_indexes)} collections",
                )
                logger.debug(f"Strategy {approach_idx} LLM reasoning: {result.reasoning}")

            # Extract selected collections using the list of indexes
            selected = [
                collections[idx]
                for idx in result.selected_item_indexes
                if 0 <= idx < len(collections)
            ]

            filtered_by_strategy[approach_idx] = selected

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
        query_approaches: List[PDS4QueryApproach] = None,
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
            query_approaches,
            run_id,
        )

        # Flatten all strategy results into single list
        all_filtered = []
        for approach_idx in sorted(filtered_by_strategy.keys()):
            all_filtered.extend(filtered_by_strategy[approach_idx])

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
