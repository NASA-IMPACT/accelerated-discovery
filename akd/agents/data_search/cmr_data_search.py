"""
CMR Data Search Agent - Discover NASA Earth science data through natural language queries.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger
from pydantic import Field, HttpUrl

from akd.tools.data_search import CMRCollectionSearchTool, CMRGranuleSearchTool

from ._base import (
    BaseDataSearchAgent,
    DataSearchAgentConfig,
    DataSearchAgentInputSchema,
    DataSearchAgentOutputSchema,
)
from .components import (  # Deprecated - kept for backward compatibility
    CMRQueryGenerationComponent,
    QueryDecompositionComponent,
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
    max_collection_search_variations: int = Field(
        default=5,
        description="Maximum number of collection search variations to try",
    )
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
        # Deprecated - kept for backward compatibility
        query_decomposition_component: QueryDecompositionComponent | None = None,
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

        # Deprecated - kept for backward compatibility
        self.query_decomposition_component = (
            query_decomposition_component or QueryDecompositionComponent(debug=debug)
        )

        # Track search state
        self.search_history: List[Dict[str, Any]] = []

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

        if self.debug:
            logger.info(f"🔍 CMR Data Search - Starting query: '{original_query}'")

        try:
            # Step 1: Scientific Expansion (Document Retrieval)
            if self.debug:
                logger.info("📚 Step 1: Retrieving relevant scientific documents...")

            documents = await self.scientific_expansion_component.process(
                original_query,
            )

            # Step 2: Scientific Angles Generation
            if self.debug:
                logger.info("🧠 Step 2: Generating scientific angles...")

            angles_output = await self.scientific_angles_component.process(
                original_query,
                documents,
            )

            # Step 3: CMR Query Generation for Each Angle
            if self.debug:
                logger.info(
                    f"⚙️ Step 3: Generating CMR queries for {len(angles_output.angles)} angles...",
                )

            all_cmr_queries = []
            for angle in angles_output.angles:
                cmr_queries_output = await self.cmr_query_generation_component.process(
                    angle,
                    original_query,
                )
                all_cmr_queries.extend(cmr_queries_output.search_queries)

            # Step 4: Collection Search
            if self.debug:
                logger.info(
                    f"🔎 Step 4: Executing {len(all_cmr_queries)} collection searches...",
                )

            collection_results = await self._search_collections_with_cmr_queries(
                all_cmr_queries,
                params,
            )

            # Step 5: Collection Synthesis
            if self.debug:
                logger.info("⚖️ Step 5: Ranking and filtering collections...")

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
                logger.warning("No relevant collections found")
                return self._create_empty_response(
                    original_query,
                    "No relevant collections found",
                )

            # Step 6: Granule Search
            if self.debug:
                logger.info(
                    f"🗂️ Step 6: Searching for granules in {len(synthesis_result.selected_collections)} collections...",
                )

            granule_results = await self._search_granules(
                synthesis_result.selected_collections,
                legacy_query_params,
                params,
            )

            # Step 7: Granule Synthesis
            if self.debug:
                logger.info("📋 Step 7: Synthesizing final results...")

            final_result = await self._synthesize_granules(
                granule_results,
                synthesis_result.selected_collections,
                legacy_query_params,
                search_start_time,
            )

            # Create response
            total_granules = len(final_result.granules)

            if self.debug:
                search_duration = (datetime.now() - search_start_time).total_seconds()
                logger.info(
                    f"✅ CMR Data Search completed: {total_granules} granules found "
                    f"in {search_duration:.1f}s",
                )

            return DataSearchAgentOutputSchema(
                granules=final_result.granules,
                search_metadata=final_result.search_metadata,
                total_results=total_granules,
                collections_searched=synthesis_result.selected_collections,
            )

        except Exception as e:
            error_msg = f"CMR data search failed: {e}"
            logger.error(error_msg)
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
                    logger.warning(f"CMR query {i} failed: {result}")
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

    # Deprecated method - kept for backward compatibility
    async def _decompose_query(
        self,
        params: DataSearchAgentInputSchema,
    ) -> Dict[str, Any]:
        """Decompose natural language query into structured parameters (DEPRECATED)."""
        logger.warning(
            "_decompose_query is deprecated. Use LLM-driven workflow instead.",
        )
        decomposed = await self.query_decomposition_component.process(params.query)

        # Combine with explicit parameters from input
        query_params = {
            "query": params.query,
            "keywords": decomposed.keywords,
            "data_type_indicators": decomposed.data_type_indicators,
            "platforms": decomposed.platforms,
            "instruments": decomposed.instruments,
            "search_variations": decomposed.search_variations,
        }

        # Override with explicit input parameters
        if params.temporal_range:
            # Simple validation - pass through as string
            if "," in params.temporal_range:
                start, end = params.temporal_range.split(",", 1)
                query_params["temporal_start"] = start.strip()
                query_params["temporal_end"] = end.strip()
        else:
            query_params["temporal_start"] = decomposed.temporal_start
            query_params["temporal_end"] = decomposed.temporal_end

        if params.spatial_bounds:
            # Simple validation - pass through as string
            query_params["spatial_bounds"] = params.spatial_bounds
        else:
            query_params["spatial_bounds"] = decomposed.spatial_bounds

        return query_params

    # Deprecated method - kept for backward compatibility
    async def _search_collections(
        self,
        query_params: Dict[str, Any],
        params: DataSearchAgentInputSchema,
    ) -> List[Dict[str, Any]]:
        """Execute collection searches using multiple parameter variations (DEPRECATED)."""
        logger.warning(
            "_search_collections is deprecated. Use _search_collections_with_cmr_queries instead.",
        )
        search_variations = query_params.get("search_variations", [])

        if not search_variations:
            # Fallback: create basic search from available parameters
            basic_search = {}
            if query_params.get("keywords"):
                basic_search["keyword"] = " ".join(query_params["keywords"][:3])
            search_variations = [basic_search]

        # Limit variations
        search_variations = search_variations[
            : self.config.max_collection_search_variations
        ]

        # Prepare search tasks
        search_tasks = []
        for i, variation in enumerate(search_variations):
            search_params = self.collection_search_tool.input_schema(
                **self._build_collection_search_params(variation, query_params, params),
            )

            task = self._execute_collection_search(search_params, f"variation_{i}")
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
                    logger.warning(f"Collection search variation {i} failed: {result}")
            else:
                successful_results.append(result.results)

        return successful_results

    def _build_collection_search_params(
        self,
        variation: Dict[str, Any],
        query_params: Dict[str, Any],
        params: DataSearchAgentInputSchema,
    ) -> Dict[str, Any]:
        """Build collection search parameters from variation and query params."""
        search_params = {}

        # Add variation-specific parameters
        for key, value in variation.items():
            if value and key != "platforms" and key != "instruments":
                search_params[key] = value

        # Handle lists specially
        if variation.get("platforms"):
            search_params["platform"] = variation["platforms"][0]
        elif query_params.get("platforms"):
            search_params["platform"] = query_params["platforms"][0]

        if variation.get("instruments"):
            search_params["instrument"] = variation["instruments"][0]
        elif query_params.get("instruments"):
            search_params["instrument"] = query_params["instruments"][0]

        # Add temporal constraints
        if query_params.get("temporal_start") and query_params.get("temporal_end"):
            search_params["temporal"] = (
                f"{query_params['temporal_start']},{query_params['temporal_end']}"
            )

        # Add spatial constraints
        spatial_bounds = query_params.get("spatial_bounds")
        if spatial_bounds and isinstance(spatial_bounds, dict):
            search_params["bounding_box"] = (
                f"{spatial_bounds['west']},{spatial_bounds['south']},"
                f"{spatial_bounds['east']},{spatial_bounds['north']}"
            )
        elif params.spatial_bounds:
            search_params["bounding_box"] = params.spatial_bounds

        return search_params

    async def _execute_collection_search(self, search_params, search_id: str):
        """Execute a single collection search."""
        if self.debug:
            logger.debug(f"Executing collection search {search_id}: {search_params}")

        return await self.collection_search_tool.arun(search_params)

    async def _synthesize_collections(
        self,
        collection_results: List[Dict[str, Any]],
        query_params: Dict[str, Any],
    ):
        """Filter and rank collection results."""
        # Simple collection filtering - take up to max_collections_to_search
        max_collections = self.config.max_collections_to_search
        if len(collection_results) <= max_collections:
            selected_collections = collection_results
        else:
            selected_collections = collection_results[:max_collections]

        # Return a simple object with selected_collections attribute
        class SynthesisResult:
            def __init__(self, selected_collections):
                self.selected_collections = selected_collections

        return SynthesisResult(selected_collections)

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
                if self.debug:
                    logger.warning(f"Granule search {i} failed: {result}")
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
        if self.debug:
            logger.debug(f"Searching granules for collection {collection_id}")

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

        # Create a simple result object
        class GranuleSynthesisResult:
            def __init__(self, granules, search_metadata):
                self.granules = granules
                self.search_metadata = search_metadata

        search_metadata = {
            "original_query": query_params.get("query", ""),
            "status": "completed",
            "search_timestamp": search_start_time.isoformat(),
            "collections_processed": len(collection_info),
            "total_granules": len(all_granules),
        }

        return GranuleSynthesisResult(all_granules, search_metadata)

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
