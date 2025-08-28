"""
CMR Data Search Agent - Discover NASA Earth science data through natural language queries.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from pydantic import Field, HttpUrl

from akd.tools.data_search import CMRCollectionSearchTool, CMRGranuleSearchTool
from akd.utils.logging import ContextualLogger, log_component_action, log_search_event
from akd.utils.serialization import safe_model_dump, safe_model_dump_list

from ._base import (
    BaseDataSearchAgent,
    CollectionSynthesisResult,
    DataSearchAgentConfig,
    DataSearchAgentInputSchema,
    DataSearchAgentOutputSchema,
    GranuleSynthesisResult,
)
from .components import (
    CMRQueryGenerationComponent,
    ScientificAnglesComponent,
    ScientificExpansionComponent,
)
from .components.scientific_angles import ScientificAngle


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
        scientific_expansion_component: ScientificExpansionComponent | None = None,
        scientific_angles_component: ScientificAnglesComponent | None = None,
        cmr_query_generation_component: CMRQueryGenerationComponent | None = None,
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

        # Initialize new LLM-driven components
        self.scientific_expansion_component = (
            scientific_expansion_component or ScientificExpansionComponent(debug=debug)
        )
        self.scientific_angles_component = (
            scientific_angles_component or ScientificAnglesComponent(debug=debug)
        )
        self.cmr_query_generation_component = (
            cmr_query_generation_component or CMRQueryGenerationComponent(debug=debug)
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
        Execute the complete CMR data search workflow.

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
        search_logger.info(f"Starting CMR data search: '{original_query}'")

        # Wait for progress handler to be ready before starting
        await self._wait_for_progress_handler_ready()

        # Emit progress update for search start
        await self._emit_progress_safely("on_search_started", original_query)

        try:
            # Step 1: Scientific Expansion (Document Retrieval)
            log_component_action(
                "ScientificExpansion",
                "STARTED",
                {"query": original_query},
            )
            search_logger.info("Step 1: Retrieving relevant scientific documents")

            await self._emit_progress_safely("on_scientific_expansion_started")

            documents = await self.scientific_expansion_component.process(
                original_query,
            )

            await self._emit_progress_safely(
                "on_scientific_expansion_completed",
                documents,
            )

            # Step 2: Scientific Angles Generation
            log_component_action(
                "ScientificAngles",
                "STARTED",
                {"documents_count": len(documents)},
            )
            search_logger.info("Step 2: Generating scientific angles")

            await self._emit_progress_safely("on_scientific_angles_started")

            angles_output = await self.scientific_angles_component.process(
                original_query,
                documents,
            )

            # Convert ScientificAngle objects to dicts for JSON serialization
            angles_data = safe_model_dump_list(angles_output.angles)
            await self._emit_progress_safely(
                "on_scientific_angles_generated",
                angles_data,
            )

            # Step 3: CMR Query Generation for Each Angle
            log_component_action(
                "CMRQueryGeneration",
                "STARTED",
                {"angles_count": len(angles_output.angles)},
            )
            search_logger.info(
                f"Step 3: Generating CMR queries for {len(angles_output.angles)} angles",
            )

            await self._emit_progress_safely("on_cmr_queries_started")

            all_cmr_queries = []
            for angle in angles_output.angles:
                cmr_queries_output = await self.cmr_query_generation_component.process(
                    angle,
                    original_query,
                )
                all_cmr_queries.extend(cmr_queries_output.search_queries)

            # Convert CMR query objects to dicts for JSON serialization
            queries_data = safe_model_dump_list(all_cmr_queries)

            log_component_action(
                "CMRQueryGeneration",
                "COMPLETED",
                {"queries_generated": len(queries_data)},
            )
            search_logger.debug(f"Generated {len(queries_data)} CMR queries")
            await self._emit_progress_safely("on_cmr_queries_generated", queries_data)

            # Step 4: Collection Search
            log_component_action(
                "CollectionSearch",
                "STARTED",
                {"queries_count": len(all_cmr_queries)},
            )
            search_logger.info(
                f"Step 4: Executing {len(all_cmr_queries)} collection searches",
            )

            await self._emit_progress_safely(
                "on_collections_search_started",
                len(all_cmr_queries),
            )

            collection_results = await self._search_collections_with_cmr_queries(
                all_cmr_queries,
                params,
            )

            # Count collection results for progress reporting (but don't send raw collections)
            total_collections_found = 0
            for result_dict in collection_results:
                if isinstance(result_dict, dict) and "collections" in result_dict:
                    collections = result_dict.get("collections", [])
                    if isinstance(collections, list):
                        total_collections_found += len(collections)

            log_component_action(
                "CollectionSearch",
                "COMPLETED",
                {"total_collections_found": total_collections_found},
            )
            search_logger.info(
                f"Found {total_collections_found} collections from searches",
            )

            # Send only a summary message, not the raw collections
            await self._emit_progress_safely(
                "on_collections_search_completed",
                {
                    "total_collections_found": total_collections_found,
                    "message": f"Found {total_collections_found} collections from {len(collection_results)} searches",
                },
            )

            # Step 5: Collection Synthesis
            log_component_action(
                "CollectionSynthesis",
                "STARTED",
                {"total_collections": total_collections_found},
            )
            search_logger.info("Step 5: Ranking and filtering collections")

            await self._emit_progress_safely("on_collections_synthesis_started")

            # Convert to legacy format for synthesis component
            legacy_query_params = self._create_legacy_query_params(
                original_query,
                angles_output.angles,
            )
            synthesis_result = await self._synthesize_collections(
                collection_results,
                legacy_query_params,
            )

            if not synthesis_result.selected_collections:
                log_search_event(
                    search_id,
                    "NO_COLLECTIONS_FOUND",
                    {"reason": "No relevant collections found"},
                )
                search_logger.warning("No relevant collections found")
                return self._create_empty_response(
                    original_query,
                    "No relevant collections found",
                )

            log_component_action(
                "CollectionSynthesis",
                "COMPLETED",
                {"selected_collections": len(synthesis_result.selected_collections)},
            )
            search_logger.info(
                f"Selected {len(synthesis_result.selected_collections)} collections for data search",
            )

            # Convert collection objects to dicts for JSON serialization
            collections_data = safe_model_dump_list(
                synthesis_result.selected_collections,
            )

            await self._emit_progress_safely(
                "on_collections_synthesized",
                collections_data,
            )

            # Step 6: Granule Search
            log_component_action(
                "GranuleSearch",
                "STARTED",
                {"collections_count": len(synthesis_result.selected_collections)},
            )
            search_logger.info(
                f"Step 6: Searching for granules in {len(synthesis_result.selected_collections)} collections",
            )

            await self._emit_progress_safely(
                "on_granules_search_started",
                len(synthesis_result.selected_collections),
            )

            granule_results = await self._search_granules(
                synthesis_result.selected_collections,
                legacy_query_params,
                params,
            )

            # Step 7: Granule Synthesis
            log_component_action("GranuleSynthesis", "STARTED")
            search_logger.info("Step 7: Synthesizing final results")

            final_result = await self._synthesize_granules(
                granule_results,
                synthesis_result.selected_collections,
                legacy_query_params,
                search_start_time,
            )

            # Calculate total file size
            total_size_mb = 0.0
            for granule in final_result.granules:
                if isinstance(granule, dict) and granule.get("file_size_mb"):
                    total_size_mb += float(granule["file_size_mb"])

            await self._emit_progress_safely(
                "on_granules_found",
                final_result.granules,
                total_size_mb if total_size_mb > 0 else None,
            )

            # Create response
            total_granules = len(final_result.granules)

            search_duration = (datetime.now() - search_start_time).total_seconds()
            log_search_event(
                search_id,
                "SEARCH_COMPLETED",
                {
                    "granules_found": total_granules,
                    "duration_seconds": search_duration,
                    "collections_searched": len(synthesis_result.selected_collections),
                },
            )
            search_logger.info(
                f"CMR Data Search completed: {total_granules} granules found in {search_duration:.1f}s",
            )

            final_response = DataSearchAgentOutputSchema(
                granules=final_result.granules,
                search_metadata=final_result.search_metadata,
                total_results=total_granules,
                collections_searched=synthesis_result.selected_collections,
            )

            await self._emit_progress_safely(
                "on_search_completed",
                safe_model_dump(final_response),
            )

            return final_response

        except Exception as e:
            error_msg = f"CMR data search failed: {e}"
            log_search_event(search_id, "SEARCH_FAILED", {"error": str(e)})
            search_logger.error(error_msg)

            await self._emit_progress_safely("on_search_error", error_msg)

            return self._create_error_response(original_query, error_msg)

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
