"""
CMR Data Search Agent - Discover NASA Earth science data through natural language queries.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List

# Temporary compatibility class for legacy methods
from pydantic import BaseModel, Field, HttpUrl

from akd.tools.data_search import CMRCollectionSearchTool, CMRGranuleSearchTool
from akd.utils.logging import ContextualLogger, log_component_action, log_search_event
from akd.utils.serialization import safe_model_dump, safe_model_dump_list

from ._base import (  # New workflow schemas; Base schemas; Legacy schemas
    AngleSearchResult,
    BaseDataSearchAgent,
    CollectionSynthesisResult,
    DataSearchAgentConfig,
    DataSearchAgentInputSchema,
    DataSearchAgentOutputSchema,
    DecompositionResult,
    GranuleSynthesisResult,
    TopicResult,
)
from .components import (  # New workflow components
    KnownParametersComponent,
    RepositoryRouterComponent,
    ScientificDecomposition,
    ScientificDecompositionComponent,
    SearchableParametersComponent,
    SearchableQuery,
    Topic,
    TopicSplittingComponent,
)
from .components.collection_ranking import (
    CollectionRankingComponent,
    CollectionRankingInputSchema,
)
from .components.repository_router import NASARepositoryEnum


class ScientificAngle(BaseModel):
    """Legacy compatibility class - not used in new workflow."""

    title: str = Field(..., description="Title of the scientific angle")
    scientific_justification: str = Field(..., description="Scientific justification")


class CMRDataSearchAgentConfig(DataSearchAgentConfig):
    """Configuration for the CMR Data Search Agent."""

    # MCP server configuration
    mcp_endpoint: HttpUrl = Field(
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
        description="Page size for granule searches",
    )

    # Quality control
    min_collection_relevance_score: float = Field(
        default=0.3,
        description="Minimum collection relevance score to include",
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

    # Model configuration for pipeline components
    topic_splitting_model: str = Field(
        default="gpt-4o",
        description="Model to use for topic splitting",
    )
    scientific_decomposition_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for scientific decomposition",
    )
    repository_routing_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for repository routing decisions",
    )
    collection_ranking_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for collection ranking and selection",
    )
    cmr_query_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for CMR query generation",
    )

    # Legacy compatibility
    angle_generation_model: str = Field(
        default="gpt-4o",
        description="Legacy parameter - now maps to topic_splitting_model for backward compatibility",
    )


class CMRDataSearchAgent(BaseDataSearchAgent):
    """
    Advanced data search agent for NASA's Common Metadata Repository.

    This agent orchestrates the complete data discovery workflow:
    1. Decomposes natural language queries into CMR search parameters
    2. Searches for relevant collections using multiple parameter combinations
    3. Filters and ranks collections based on relevance and quality
    4. Searches for granules (data files) in selected collections
    5. Synthesizes results with download URLs and comprehensive metadata

    Example usage:
        agent = CMRDataSearchAgent()
        result = await agent.arun(DataSearchAgentInputSchema(
            query="Find MODIS sea surface temperature data from 2023 over the Pacific Ocean"
        ))
    """

    input_schema = DataSearchAgentInputSchema
    output_schema = DataSearchAgentOutputSchema
    config_schema = CMRDataSearchAgentConfig

    def __init__(
        self,
        config: CMRDataSearchAgentConfig | None = None,
        collection_search_tool: CMRCollectionSearchTool | None = None,
        granule_search_tool: CMRGranuleSearchTool | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the CMR Data Search Agent."""
        super().__init__(config=config or CMRDataSearchAgentConfig(), debug=debug)

        # Initialize tools with shared MCP endpoint configuration
        tool_config_params = {
            "mcp_endpoint": self.config.mcp_endpoint,
            "timeout_seconds": self.config.collection_search_timeout,
            "debug": debug,
        }

        self.collection_search_tool = (
            collection_search_tool
            or CMRCollectionSearchTool.from_params(
                page_size=self.config.collection_search_page_size,
                **tool_config_params,
            )
        )

        granule_tool_config = tool_config_params.copy()
        granule_tool_config["timeout_seconds"] = self.config.granule_search_timeout

        self.granule_search_tool = (
            granule_search_tool
            or CMRGranuleSearchTool.from_params(
                collection_concept_id="",  # Will be set per search
                page_size=self.config.granule_search_page_size,
                **granule_tool_config,
            )
        )

        # Initialize LLM-driven components with model-specific configurations
        from akd.agents._base import BaseAgentConfig

        # Create configs for each component with specific models
        topic_config = BaseAgentConfig(model_name=self.config.topic_splitting_model)
        router_config = BaseAgentConfig(model_name=self.config.repository_routing_model)
        decomp_config = BaseAgentConfig(
            model_name=self.config.scientific_decomposition_model,
        )
        known_params_config = BaseAgentConfig(model_name=self.config.cmr_query_model)
        searchable_params_config = BaseAgentConfig(
            model_name=self.config.cmr_query_model,
        )
        ranking_config = BaseAgentConfig(
            model_name=self.config.collection_ranking_model,
        )

        # New workflow components
        self.topic_splitting_component = TopicSplittingComponent(
            config=topic_config,
            debug=debug,
        )
        self.repository_router_component = RepositoryRouterComponent(
            config=router_config,
            debug=debug,
        )
        self.scientific_decomposition_component = ScientificDecompositionComponent(
            config=decomp_config,
            debug=debug,
        )
        self.known_parameters_component = KnownParametersComponent(
            config=known_params_config,
            debug=debug,
        )
        self.searchable_parameters_component = SearchableParametersComponent(
            config=searchable_params_config,
            debug=debug,
        )
        self.collection_ranking_component = CollectionRankingComponent(
            config=ranking_config,
            debug=debug,
        )

        # Track search state
        self.search_history: List[Dict[str, Any]] = []

        # Optional progress handler for frontend integration
        self.progress_handler = None
        self._progress_handler_ready = False

        # Contextual logger for this agent
        self.agent_logger = ContextualLogger("CMRDataSearchAgent")

    def set_progress_handler(self, progress_handler):
        """
        Set the progress handler for real-time frontend updates.

        Args:
            progress_handler: SearchProgressHandler instance for WebSocket communication
        """
        self.progress_handler = progress_handler
        self._progress_handler_ready = True
        self.agent_logger.debug("Progress handler set and marked as ready")

    async def _wait_for_progress_handler_ready(self, timeout: float = 5.0):
        """Wait for progress handler to be ready or timeout."""
        import asyncio

        if not self.progress_handler:
            return

        start_time = datetime.now()
        while not self._progress_handler_ready:
            if (datetime.now() - start_time).total_seconds() > timeout:
                self.agent_logger.warning(
                    "Progress handler readiness timeout - proceeding without waiting",
                )
                break
            await asyncio.sleep(0.1)

    async def _emit_progress_safely(self, method_name: str, *args, **kwargs):
        """Safely emit progress updates with error handling."""
        if not self.progress_handler or not hasattr(self.progress_handler, method_name):
            return

        try:
            method = getattr(self.progress_handler, method_name)
            await method(*args, **kwargs)
            self.agent_logger.debug(f"Progress update sent: {method_name}")
        except Exception as e:
            self.agent_logger.warning(f"Progress update failed for {method_name}: {e}")
            # Continue execution - progress failures shouldn't stop the search

    async def _arun(
        self,
        params: DataSearchAgentInputSchema,
        **kwargs: Any,
    ) -> DataSearchAgentOutputSchema:
        """
        Execute the topic-based CMR data search workflow.

        Args:
            params: Input parameters with natural language query
            **kwargs: Additional parameters

        Returns:
            DataSearchAgentOutputSchema with discovered data files and metadata
        """
        search_start_time = datetime.now()
        original_query = params.query

        # Create search-specific logger
        search_id = (
            getattr(self.progress_handler, "search_id", "unknown")
            if self.progress_handler
            else "unknown"
        )
        search_logger = ContextualLogger("CMRDataSearchAgent", search_id)

        log_search_event(
            search_id,
            "SEARCH_STARTED",
            {"query": original_query, "start_time": search_start_time.isoformat()},
        )
        search_logger.info(f"Starting topic-based data search: '{original_query}'")

        # Wait for progress handler to be ready before starting
        await self._wait_for_progress_handler_ready()

        # Emit progress update for search start
        await self._emit_progress_safely("on_search_started", original_query)

        try:
            # Step 1: Topic Splitting
            log_component_action("TopicSplitting", "STARTED", {"query": original_query})
            search_logger.info("Step 1: Identifying functional topics")

            topics_output = await self.topic_splitting_component.process(original_query)
            search_logger.info(
                f"Identified {len(topics_output.topics)} functional topics",
            )

            # Step 2: Repository Router (process each topic individually)
            log_component_action(
                "RepositoryRouter",
                "STARTED",
                {"topics_count": len(topics_output.topics)},
            )
            search_logger.info("Step 2: Routing topics to data sources")

            routes = []
            for topic in topics_output.topics:
                routing_output = await self.repository_router_component.process(
                    original_query,
                    topic,
                )
                routes.append(routing_output.route)

            # Step 3: Process each topic based on its own route
            topic_results = []
            for i, (topic, route) in enumerate(
                zip(topics_output.topics, routes),
                start=1,
            ):
                repos = [r.strip() for r in route.repositories]
                search_logger.info(f"Routing for topic {i}: {topic.title} → {repos}")

                # Check if CMR is in the repositories
                has_cmr = NASARepositoryEnum.CMR in route.repositories
                if has_cmr:
                    search_logger.info(f"Processing topic {i} via CMR")
                    topic_result = await self._process_single_topic(
                        topic,
                        original_query,
                        params,
                    )
                    topic_results.append(topic_result)
                else:
                    default_repo = repos[0] if repos else "Unknown"
                    note_text = (
                        "; ".join(route.rationales) if route.rationales else None
                    )
                    topic_results.append(
                        TopicResult(
                            topic=safe_model_dump(topic),
                            data_source=default_repo,
                            decomposition_results=[],
                            note=note_text
                            or f"Routed to {default_repo}; CMR not selected",
                        ),
                    )

            # Calculate totals
            total_granules = sum(
                sum(dr.total_granules_found for dr in tr.decomposition_results)
                for tr in topic_results
            )

            search_duration = (datetime.now() - search_start_time).total_seconds()

            log_component_action(
                "TopicProcessing",
                "COMPLETED",
                {
                    "topics_processed": len(topic_results),
                    "total_granules": total_granules,
                    "duration_seconds": search_duration,
                },
            )

            search_logger.info(
                f"Topic-based search completed: {total_granules} granules found across {len(topic_results)} topics in {search_duration:.1f}s",
            )

            # Build search metadata
            search_metadata = {
                "search_id": search_id,
                "original_query": original_query,
                "timestamp": search_start_time.isoformat(),
                "duration_seconds": search_duration,
                "topics_processed": len(topic_results),
                "workflow_version": "topic-decomposition-v1",
            }

            # Create topic-organized response
            final_response = DataSearchAgentOutputSchema(
                topics=topic_results,
                search_metadata=search_metadata,
                total_results=total_granules,
                # Empty legacy fields for schema compatibility
                angles=[],
                granules=[],
                collections_searched=[],
            )

            await self._emit_progress_safely(
                "on_search_completed",
                safe_model_dump(final_response),
            )

            return final_response

        except Exception as e:
            error_msg = f"Topic-based data search failed: {e}"
            log_search_event(search_id, "SEARCH_FAILED", {"error": str(e)})
            search_logger.error(error_msg)

            await self._emit_progress_safely("on_search_error", error_msg)

            return self._create_error_response(original_query, error_msg)

    async def _process_single_topic(
        self,
        topic: Topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> TopicResult:
        """
        Process a single topic through the complete pipeline.

        Args:
            topic: Topic to process
            original_query: Original research question
            params: Search parameters

        Returns:
            Complete topic result with all decompositions
        """
        search_logger = ContextualLogger("topic_processing")
        search_logger.info(f"Processing topic: {topic.title}")

        # Scientific Decomposition
        log_component_action(
            "ScientificDecomposition",
            "STARTED",
            {"topic": topic.title},
        )
        decomp_output = await self.scientific_decomposition_component.process(
            original_query,
            topic,
        )
        search_logger.info(
            f"Generated {len(decomp_output.decompositions)} decompositions for topic '{topic.title}'",
        )

        # Process each decomposition
        decomp_results = []
        for i, decomp in enumerate(decomp_output.decompositions):
            search_logger.info(
                f"Processing decomposition {i + 1}/{len(decomp_output.decompositions)}: {decomp.title}",
            )
            decomp_result = await self._process_single_decomposition(
                topic,
                decomp,
                original_query,
                params,
            )
            decomp_results.append(decomp_result)

        return TopicResult(
            topic=safe_model_dump(topic),
            data_source="CMR",
            decomposition_results=decomp_results,
        )

    async def _process_single_decomposition(
        self,
        topic: Topic,
        decomp: ScientificDecomposition,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> DecompositionResult:
        """
        Process a single decomposition through parameter generation, search, and ranking.

        Args:
            topic: Parent topic
            decomp: Scientific decomposition to process
            original_query: Original research question
            params: Search parameters

        Returns:
            Complete decomposition result
        """
        search_logger = ContextualLogger("decomposition_processing")
        search_logger.info(f"Processing decomposition: {decomp.title}")

        # Known Parameters
        log_component_action(
            "KnownParameters",
            "STARTED",
            {"decomposition": decomp.title},
        )
        known_params_output = await self.known_parameters_component.process(
            original_query,
            topic,
            decomp,
        )
        search_logger.info(
            f"Generated {len(known_params_output.query_approaches)} query approaches",
        )

        # Searchable Parameters
        log_component_action(
            "SearchableParameters",
            "STARTED",
            {"approaches": len(known_params_output.query_approaches)},
        )
        searchable_output = await self.searchable_parameters_component.process(
            original_query,
            topic,
            decomp,
            known_params_output.query_approaches,
        )
        search_logger.info(
            f"Generated {len(searchable_output.searchable_queries)} searchable queries",
        )

        # Query Execution
        log_component_action(
            "QueryExecution",
            "STARTED",
            {"queries": len(searchable_output.searchable_queries)},
        )
        collections = await self._execute_searchable_queries(
            searchable_output.searchable_queries,
            params,
        )
        search_logger.info(f"Found {len(collections)} collections from queries")

        # Collection Ranking & Filtering
        log_component_action(
            "CollectionRanking",
            "STARTED",
            {"collections": len(collections)},
        )
        ranked_collections = await self._rank_collections(
            collections,
            original_query,
            topic,
            decomp,
        )
        search_logger.info(f"Ranked to {len(ranked_collections)} top collections")

        # Granule Search
        log_component_action(
            "GranuleSearch",
            "STARTED",
            {"collections": len(ranked_collections)},
        )
        granules = await self._search_granules_for_collections(
            ranked_collections,
            params,
        )
        search_logger.info(f"Found {len(granules)} granules")

        return DecompositionResult(
            decomposition=safe_model_dump(decomp),
            query_approaches=safe_model_dump_list(known_params_output.query_approaches),
            searchable_queries=safe_model_dump_list(
                searchable_output.searchable_queries,
            ),
            collections=ranked_collections,
            granules=granules,
            total_collections_found=len(collections),
            total_granules_found=len(granules),
        )

    async def _execute_searchable_queries(
        self,
        searchable_queries: List[SearchableQuery],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """Execute searchable queries and return collections."""
        all_collections = []

        for query in searchable_queries:
            # Convert SearchableQuery to tool parameters
            search_params = {}

            # Add known parameters
            if query.instrument:
                search_params["instrument"] = query.instrument
            if query.platform:
                search_params["platform"] = query.platform
            if query.temporal:
                search_params["temporal"] = query.temporal
            if query.bounding_box:
                search_params["bounding_box"] = query.bounding_box

            # Add searchable parameters
            if query.combined_keyword_string:
                search_params["keyword"] = query.combined_keyword_string

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

                # Extract collections
                if hasattr(result, "collections") and result.collections:
                    all_collections.extend(result.collections)

            except Exception as e:
                search_logger = ContextualLogger("query_execution")
                search_logger.warning(f"Query execution failed: {e}")

        return all_collections

    async def _rank_collections(
        self,
        collections: List[Dict[str, Any]],
        original_query: str,
        topic: Topic,
        decomp: ScientificDecomposition,
    ) -> List[Dict[str, Any]]:
        """Rank collections using the collection ranking component."""
        if not collections:
            return []

        # Limit collections if too many
        if len(collections) <= self.config.max_collections_to_search:
            return collections

        # Use collection ranking component
        ranking_input = CollectionRankingInputSchema(
            original_query=original_query,
            scientific_angle=safe_model_dump(decomp),  # Use decomposition as "angle"
            collections=collections,
            max_collections=self.config.max_collections_to_search,
        )

        try:
            ranking_result = await self.collection_ranking_component.arun(ranking_input)
            ranked_collections = [
                collections[rc.collection_index]
                for rc in ranking_result.ranked_collections
                if 0 <= rc.collection_index < len(collections)
            ]
            return ranked_collections
        except Exception as e:
            search_logger = ContextualLogger("collection_ranking")
            search_logger.error(f"Collection ranking failed: {e}")
            raise RuntimeError("Collection ranking failed") from e

    async def _search_granules_for_collections(
        self,
        collections: List[Dict[str, Any]],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """Search for granules in the provided collections."""
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
                search_logger = ContextualLogger("granule_search")
                search_logger.warning(
                    f"Granule search failed for collection {concept_id}: {e}",
                )

        return all_granules

    async def _search_collections_with_cmr_queries(
        self,
        cmr_queries: List,  # List of CMRCollectionSearchParams
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """Execute collection searches using LLM-generated CMR queries."""
        # Prepare search tasks
        search_tasks = []
        for i, cmr_query in enumerate(cmr_queries):
            # Convert CMRCollectionSearchParams to tool input schema
            search_params = self._convert_cmr_query_to_tool_params(cmr_query, params)
            tool_input = self.collection_search_tool.input_schema(**search_params)

            task = self._execute_collection_search(tool_input, f"cmr_query_{i}")
            search_tasks.append(task)

        # Execute searches in parallel
        if self.config.enable_parallel_search and len(search_tasks) > 1:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        else:
            results = []
            for task in search_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Filter successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if self.debug:
                    self.agent_logger.warning(f"CMR query {i} failed: {result}")
            else:
                successful_results.append(result.results)

        return successful_results

    def _convert_cmr_query_to_tool_params(
        self,
        cmr_query,
        params: DataSearchAgentInputSchema,  # CMRCollectionSearchParams
    ) -> Dict[str, Any]:
        """Convert CMRCollectionSearchParams to tool input parameters."""
        search_params = {}

        # Map CMR query fields to tool parameters
        if cmr_query.keyword:
            search_params["keyword"] = cmr_query.keyword
        if cmr_query.short_name:
            search_params["short_name"] = cmr_query.short_name
        if cmr_query.platform:
            search_params["platform"] = cmr_query.platform
        if cmr_query.instrument:
            search_params["instrument"] = cmr_query.instrument
        if cmr_query.temporal:
            search_params["temporal"] = cmr_query.temporal
        if cmr_query.bounding_box:
            search_params["bounding_box"] = cmr_query.bounding_box

        # Override with explicit input parameters
        if params.temporal_range:
            search_params["temporal"] = params.temporal_range
        if params.spatial_bounds:
            search_params["bounding_box"] = params.spatial_bounds

        # Add pagination
        search_params["page_size"] = self.config.collection_search_page_size

        return search_params

    def _create_legacy_query_params(
        self,
        original_query: str,
        angles: List[ScientificAngle],
    ) -> Dict[str, Any]:
        """Create legacy query params for backward compatibility with synthesis components."""
        return {
            "query": original_query,
            "keywords": [angle.title for angle in angles],
            "data_type_indicators": [],
            "platforms": [],
            "instruments": [],
            "temporal_start": None,
            "temporal_end": None,
            "spatial_bounds": None,
            "search_variations": [],
        }

    async def _execute_collection_search(self, search_params, search_id: str):
        """Execute a single collection search."""
        log_component_action(
            "CollectionSearch",
            "EXECUTE",
            {"search_id": search_id, "params": str(search_params)},
        )

        return await self.collection_search_tool.arun(search_params)

    def _deduplicate_collections_within_angle(
        self,
        collections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate collections within a single angle based on concept_id.

        Args:
            collections: List of collection dictionaries

        Returns:
            Deduplicated list of collections, preserving order of first occurrence
        """
        seen_concept_ids = set()
        deduplicated = []

        for collection in collections:
            concept_id = collection.get("concept_id")
            if concept_id and concept_id not in seen_concept_ids:
                seen_concept_ids.add(concept_id)
                deduplicated.append(collection)
            elif not concept_id:
                # Include collections without concept_id (shouldn't happen but be safe)
                deduplicated.append(collection)

        return deduplicated

    async def _process_single_angle(
        self,
        angle: ScientificAngle,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> AngleSearchResult:
        """
        Process a single scientific angle through the complete pipeline.

        Args:
            angle: Scientific angle to process
            original_query: Original user query
            params: Search parameters

        Returns:
            Complete search result for this angle
        """
        search_logger = ContextualLogger("angle_processing")
        search_logger.info(f"Processing angle: {angle.title}")

        # Step 3a: Generate CMR queries for this angle
        cmr_queries_output = await self.cmr_query_generation_component.process(
            angle,
            original_query,
        )
        cmr_queries = cmr_queries_output.search_queries

        # Step 3b: Execute collection searches for this angle's queries
        collection_results = await self._search_collections_with_cmr_queries(
            cmr_queries,
            params,
        )

        # Step 3c: Extract and deduplicate collections for this angle
        angle_collections = []
        for result_dict in collection_results:
            if isinstance(result_dict, dict) and "collections" in result_dict:
                collections = result_dict["collections"]
                if isinstance(collections, list):
                    angle_collections.extend(collections)

        # Deduplicate within this angle
        deduplicated_collections = self._deduplicate_collections_within_angle(
            angle_collections,
        )
        total_collections_found = len(angle_collections)

        search_logger.info(
            f"Found {total_collections_found} collections, {len(deduplicated_collections)} after deduplication",
        )

        # Step 3d: Rank collections for this angle (limit to top collections)
        ranked_collections = deduplicated_collections
        if len(deduplicated_collections) > self.config.max_collections_to_search:
            # Use collection ranking component to select best collections

            ranking_input = CollectionRankingInputSchema(
                original_query=original_query,
                scientific_angle=safe_model_dump(angle),
                collections=deduplicated_collections,
                max_collections=self.config.max_collections_to_search,
            )

            ranking_result = await self.collection_ranking_component.arun(ranking_input)
            ranked_collections = [
                deduplicated_collections[rc.collection_index]
                for rc in ranking_result.ranked_collections
                if 0 <= rc.collection_index < len(deduplicated_collections)
            ]

        search_logger.info(
            f"Selected {len(ranked_collections)} collections for granule search",
        )

        # Step 3e: Search granules for this angle's collections
        granule_results = []
        if ranked_collections:
            # Build legacy query params for granule search
            legacy_query_params = self._create_legacy_query_params(
                original_query,
                [angle],
            )
            granule_results = await self._search_granules(
                ranked_collections,
                legacy_query_params,
                params,
            )

        # Step 3f: Process granules for this angle
        angle_granules = []
        for result_dict in granule_results:
            if isinstance(result_dict, dict) and "granules" in result_dict:
                granules = result_dict["granules"]
                if isinstance(granules, list):
                    angle_granules.extend(granules)

        search_logger.info(f"Found {len(angle_granules)} granules for angle")

        # Return complete angle result
        return AngleSearchResult(
            scientific_angle=safe_model_dump(angle),
            cmr_queries=safe_model_dump_list(cmr_queries),
            collections=ranked_collections,
            granules=angle_granules,
            total_collections_found=total_collections_found,
            total_granules_found=len(angle_granules),
        )

    async def _synthesize_collections(
        self,
        collection_results: List[Dict[str, Any]],
        query_params: Dict[str, Any],
    ):
        """Filter and rank collection results."""
        # Extract actual collections from MCP response format
        all_collections = []
        for result_dict in collection_results:
            if isinstance(result_dict, dict) and "collections" in result_dict:
                collections = result_dict["collections"]
                if isinstance(collections, list):
                    all_collections.extend(collections)

        # Simple collection filtering - take up to max_collections_to_search
        max_collections = self.config.max_collections_to_search
        if len(all_collections) <= max_collections:
            selected_collections = all_collections
        else:
            selected_collections = all_collections[:max_collections]

        # Return proper Pydantic model
        return CollectionSynthesisResult(
            selected_collections=selected_collections,
            total_collections_found=len(all_collections),
        )

    async def _search_granules(
        self,
        selected_collections: List[Dict[str, Any]],
        query_params: Dict[str, Any],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """Search for granules in selected collections."""
        granule_tasks = []

        for collection in selected_collections:
            concept_id = collection.get("concept_id")
            if not concept_id:
                continue

            search_params = self._build_granule_search_params(
                concept_id,
                query_params,
                params,
            )

            task = self._execute_granule_search(search_params, concept_id)
            granule_tasks.append(task)

        # Execute granule searches
        if self.config.enable_parallel_search and len(granule_tasks) > 1:
            results = await asyncio.gather(*granule_tasks, return_exceptions=True)
        else:
            results = []
            for task in granule_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Filter successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log_component_action(
                    "GranuleSearch",
                    "FAILED",
                    {"search_index": i, "error": str(result)},
                )
            else:
                successful_results.append(result.results)

        return successful_results

    def _build_granule_search_params(
        self,
        collection_concept_id: str,
        query_params: Dict[str, Any],
        params: DataSearchAgentInputSchema,
    ) -> Dict[str, Any]:
        """Build granule search parameters."""
        granule_params = {
            "collection_concept_id": collection_concept_id,
            "page_size": self.config.granule_search_page_size,
        }

        # Add temporal constraint
        if query_params.get("temporal_start") and query_params.get("temporal_end"):
            granule_params["temporal"] = (
                f"{query_params['temporal_start']},{query_params['temporal_end']}"
            )

        # Add spatial constraints
        spatial_bounds = query_params.get("spatial_bounds")
        if spatial_bounds and isinstance(spatial_bounds, dict):
            granule_params["bounding_box"] = (
                f"{spatial_bounds['west']},{spatial_bounds['south']},"
                f"{spatial_bounds['east']},{spatial_bounds['north']}"
            )
        elif params.spatial_bounds:
            granule_params["bounding_box"] = params.spatial_bounds

        return granule_params

    async def _execute_granule_search(
        self,
        search_params: Dict[str, Any],
        collection_id: str,
    ):
        """Execute a single granule search."""
        log_component_action(
            "GranuleSearch",
            "EXECUTE",
            {"collection_id": collection_id},
        )

        granule_search_params = self.granule_search_tool.input_schema(**search_params)
        return await self.granule_search_tool.arun(granule_search_params)

    async def _synthesize_granules(
        self,
        granule_results: List[Dict[str, Any]],
        collection_info: List[Dict[str, Any]],
        query_params: Dict[str, Any],
        search_start_time: datetime,
    ):
        """Synthesize final granule results."""
        # Simple granule synthesis - flatten all results
        all_granules = []
        for collection_granules in granule_results:
            if isinstance(collection_granules, list):
                all_granules.extend(collection_granules)
            elif collection_granules.get("granules"):
                all_granules.extend(collection_granules["granules"])

        # Create proper Pydantic model result
        search_metadata = {
            "original_query": query_params.get("query", ""),
            "status": "completed",
            "search_timestamp": search_start_time.isoformat(),
            "collections_processed": len(collection_info),
            "total_granules": len(all_granules),
        }

        return GranuleSynthesisResult(
            granules=all_granules,
            search_metadata=search_metadata,
            total_granules_found=len(all_granules),
            collections_processed=len(collection_info),
        )

    def _create_empty_response(
        self,
        query: str,
        reason: str,
    ) -> DataSearchAgentOutputSchema:
        """Create empty response with explanation."""
        return DataSearchAgentOutputSchema(
            angles=[],  # Required field in new schema
            granules=[],
            search_metadata={
                "original_query": query,
                "status": "no_results",
                "reason": reason,
                "search_timestamp": datetime.now().isoformat(),
            },
            total_results=0,
            collections_searched=[],
        )

    def _create_error_response(
        self,
        query: str,
        error_msg: str,
    ) -> DataSearchAgentOutputSchema:
        """Create error response."""
        return DataSearchAgentOutputSchema(
            angles=[],  # Required field in new schema
            granules=[],
            search_metadata={
                "original_query": query,
                "status": "error",
                "error": error_msg,
                "search_timestamp": datetime.now().isoformat(),
            },
            total_results=0,
            collections_searched=[],
        )
