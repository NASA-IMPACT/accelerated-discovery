"""
Multi-Repository Data Search Agent.

Orchestrates data discovery across multiple NASA repositories (CMR, PDS4, etc.)
and external data sources.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from pydantic import Field

from akd.agents._base import BaseAgentConfig
from akd.utils.logging import ContextualLogger, log_component_action, log_search_event
from akd.utils.serialization import safe_model_dump

from ._base import (
    BaseDataSearchAgent,
    BaseDataSearchConfig,
    DataSearchAgentInputSchema,
    DataSearchAgentOutputSchema,
    DecompositionResult,
    TopicResult,
)
from .components import (
    RepositoryRouterComponent,
    ScientificDecompositionComponent,
    TopicSplittingComponent,
)
from .components.repository_router import NASARepositoryEnum
from .handlers import (
    HANDLER_STATUS,
    HANDLERS,
    CMRHandlerConfig,
    HandlerStatus,
    PDS4HandlerConfig,
)


class DataSearchAgentConfig(BaseDataSearchConfig):
    """Configuration for data search agent with multi-repository support."""

    # Handler-specific configurations
    cmr: CMRHandlerConfig = Field(
        default_factory=CMRHandlerConfig,
        description="CMR handler configuration",
    )
    pds4: PDS4HandlerConfig = Field(
        default_factory=PDS4HandlerConfig,
        description="PDS4 handler configuration",
    )


class DataSearchAgent(BaseDataSearchAgent):
    """
    Multi-repository data search agent.

    Orchestrates data discovery workflow across multiple repositories:
    1. Splits query into functional topics
    2. Decomposes topics into scientific observables
    3. Routes each decomposition to best data source
    4. Dispatches to repository-specific handlers (CMR, PDS4, etc.)
    5. Returns organized results with metadata

    Example usage:
        config = DataSearchAgentConfig(
            cmr=CMRHandlerConfig(mcp_endpoint="http://localhost:8080/mcp/cmr/mcp/")
        )
        agent = DataSearchAgent(config=config)
        result = await agent.arun(DataSearchAgentInputSchema(
            query="Find MODIS sea surface temperature data from 2023"
        ))
    """

    input_schema = DataSearchAgentInputSchema
    output_schema = DataSearchAgentOutputSchema
    config_schema = DataSearchAgentConfig

    def __init__(
        self,
        config: DataSearchAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the data search agent."""
        super().__init__(
            config=config or DataSearchAgentConfig(),
            debug=debug,
        )

        # Initialize universal components (used for all queries)
        topic_config = BaseAgentConfig(model_name=self.config.topic_splitting_model)
        decomp_config = BaseAgentConfig(
            model_name=self.config.scientific_decomposition_model,
        )
        router_config = BaseAgentConfig(
            model_name=self.config.repository_routing_model,
        )

        self.topic_splitting_component = TopicSplittingComponent(
            config=topic_config,
            debug=debug,
        )
        self.scientific_decomposition_component = ScientificDecompositionComponent(
            config=decomp_config,
            debug=debug,
        )
        self.repository_router_component = RepositoryRouterComponent(
            config=router_config,
            debug=debug,
        )

        # Handlers - initialized lazily when needed
        self._handlers = {}

        # Optional progress handler for frontend integration
        self.progress_handler = None
        self._progress_handler_ready = False

        # Logger
        self.agent_logger = ContextualLogger("DataSearchAgent")

    def set_progress_handler(self, progress_handler):
        """Set the progress handler for real-time frontend updates."""
        self.progress_handler = progress_handler
        self._progress_handler_ready = True
        self.agent_logger.debug("Progress handler set and marked as ready")

    async def _wait_for_progress_handler_ready(self, timeout: float = 5.0):
        """Wait for progress handler to be ready or timeout."""
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
        if not self.progress_handler or not hasattr(
            self.progress_handler,
            method_name,
        ):
            return

        try:
            method = getattr(self.progress_handler, method_name)
            await method(*args, **kwargs)
            self.agent_logger.debug(f"Progress update sent: {method_name}")
        except Exception as e:
            self.agent_logger.warning(f"Progress update failed for {method_name}: {e}")

    def _get_handler(self, repository: NASARepositoryEnum):
        """Get or create handler for repository."""
        if repository not in self._handlers:
            handler_class = HANDLERS.get(repository)
            if not handler_class:
                raise ValueError(f"No handler available for repository: {repository}")

            # Get repository-specific config
            if repository == NASARepositoryEnum.CMR:
                handler_config = self.config.cmr
            elif repository == NASARepositoryEnum.PDS4:
                handler_config = self.config.pds4
            else:
                raise ValueError(f"No config defined for repository: {repository}")

            self._handlers[repository] = handler_class(
                config=handler_config,
                debug=self.config.debug,
            )

        return self._handlers[repository]

    def get_cmr_handler(self):
        """
        Get the CMR handler for direct access to CMR-specific components.

        This is useful for testing and debugging workflows that need to access
        individual CMR components (known_parameters_component, searchable_parameters_component,
        collection_search_tool, granule_search_tool).

        Returns:
            CMRHandler instance with access to CMR-specific components
        """
        return self._get_handler(NASARepositoryEnum.CMR)

    def _create_external_result(
        self,
        decomposition,
        route,
    ) -> DecompositionResult:
        """Create result for external data sources."""
        return DecompositionResult(
            decomposition=safe_model_dump(decomposition),
            repository=route.repository,
            query_approaches=[],
            searchable_queries=[],
            data_results=[],
            total_results_found=0,
            note=f"Data available from {route.repository}. {route.rationale}",
        )

    def _create_stub_result(
        self,
        decomposition,
        route,
    ) -> DecompositionResult:
        """Create result for NASA repositories with stub handlers."""
        return DecompositionResult(
            decomposition=safe_model_dump(decomposition),
            repository=route.repository,
            query_approaches=[],
            searchable_queries=[],
            data_results=[],
            total_results_found=0,
            note=f"{route.repository} handler implementation is in progress. {route.rationale}",
        )

    def _create_error_result(
        self,
        decomposition,
        route,
        error_message: str,
    ) -> DecompositionResult:
        """Create result for handler errors."""
        return DecompositionResult(
            decomposition=safe_model_dump(decomposition),
            repository=route.repository,
            query_approaches=[],
            searchable_queries=[],
            data_results=[],
            total_results_found=0,
            note=f"Error processing with {route.repository}: {error_message}",
        )

    async def _arun(
        self,
        params: DataSearchAgentInputSchema,
        **kwargs: Any,
    ) -> DataSearchAgentOutputSchema:
        """
        Execute the multi-repository data search workflow.

        Flow:
        1. Topic Splitting → 1-6 topics
        2. For each topic:
           a. Scientific Decomposition → 1-6 decompositions
           b. For each decomposition:
              - Repository Routing → single best repository
              - If external: create note result
              - If NASA repo: dispatch to handler
        3. Return organized results

        Args:
            params: Input parameters with natural language query

        Returns:
            DataSearchAgentOutputSchema with discovered data
        """
        search_start_time = datetime.now()
        original_query = params.query

        # Create search-specific logger
        search_id = (
            getattr(self.progress_handler, "search_id", "unknown")
            if self.progress_handler
            else "unknown"
        )
        search_logger = ContextualLogger("DataSearchAgent", search_id)

        log_search_event(
            search_id,
            "SEARCH_STARTED",
            {"query": original_query, "start_time": search_start_time.isoformat()},
        )
        search_logger.info(
            f"Starting multi-repository data search: '{original_query}'",
        )

        await self._wait_for_progress_handler_ready()
        await self._emit_progress_safely("on_search_started", original_query)

        try:
            # Step 1: Topic Splitting
            log_component_action("TopicSplitting", "STARTED", {"query": original_query})
            search_logger.info("Step 1: Identifying functional topics")

            topics_output = await self.topic_splitting_component.process(original_query)
            search_logger.info(
                f"Identified {len(topics_output.topics)} functional topics",
            )

            # Step 2: Process topics in parallel
            topic_tasks = []
            for topic in topics_output.topics:
                task = self._process_single_topic(topic, original_query, params)
                topic_tasks.append(task)

            search_logger.info(f"Processing {len(topic_tasks)} topics in parallel...")
            topic_results = await asyncio.gather(*topic_tasks, return_exceptions=True)

            # Handle any exceptions from parallel execution
            final_topic_results = []
            for i, result in enumerate(topic_results):
                if isinstance(result, Exception):
                    search_logger.error(
                        f"Topic {i + 1} processing failed: {result}",
                    )
                    # Create error result
                    error_result = TopicResult(
                        topic=safe_model_dump(topics_output.topics[i]),
                        data_source="Unknown",
                        decomposition_results=[],
                        note=f"Processing error: {str(result)}",
                    )
                    final_topic_results.append(error_result)
                else:
                    final_topic_results.append(result)

            # Calculate totals
            total_results = sum(
                sum(dr.total_results_found for dr in tr.decomposition_results)
                for tr in final_topic_results
            )

            search_duration = (datetime.now() - search_start_time).total_seconds()

            log_component_action(
                "TopicProcessing",
                "COMPLETED",
                {
                    "topics_processed": len(final_topic_results),
                    "total_results": total_results,
                    "duration_seconds": search_duration,
                },
            )

            search_logger.info(
                f"Search completed: {total_results} results found across "
                f"{len(final_topic_results)} topics in {search_duration:.1f}s",
            )

            # Build search metadata
            search_metadata = {
                "search_id": search_id,
                "original_query": original_query,
                "timestamp": search_start_time.isoformat(),
                "duration_seconds": search_duration,
                "topics_processed": len(final_topic_results),
                "workflow_version": "multi-repo-v1",
            }

            # Create response
            final_response = DataSearchAgentOutputSchema(
                topics=final_topic_results,
                search_metadata=search_metadata,
                total_results=total_results,
            )

            await self._emit_progress_safely(
                "on_search_completed",
                safe_model_dump(final_response),
            )

            return final_response

        except Exception as e:
            error_msg = f"Multi-repository data search failed: {e}"
            log_search_event(search_id, "SEARCH_FAILED", {"error": str(e)})
            search_logger.error(error_msg)

            await self._emit_progress_safely("on_search_error", error_msg)

            return self._create_error_response(original_query, error_msg)

    async def _process_single_topic(
        self,
        topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> TopicResult:
        """
        Process a single topic through decomposition and routing.

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

        # Process each decomposition in parallel
        decomp_tasks = []
        for decomp in decomp_output.decompositions:
            task = self._process_single_decomposition(
                topic,
                decomp,
                original_query,
                params,
            )
            decomp_tasks.append(task)

        search_logger.info(
            f"Processing {len(decomp_tasks)} decompositions in parallel",
        )
        decomp_results = await asyncio.gather(*decomp_tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(decomp_results):
            if isinstance(result, Exception):
                search_logger.error(f"Decomposition {i + 1} failed: {result}")
                error_result = DecompositionResult(
                    decomposition=safe_model_dump(decomp_output.decompositions[i]),
                    repository=None,
                    query_approaches=[],
                    searchable_queries=[],
                    data_results=[],
                    total_results_found=0,
                    note=f"Processing error: {str(result)}",
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        # Determine primary data source from first successful result
        data_source = "Unknown"
        for result in final_results:
            if result.repository:
                data_source = result.repository
                break

        return TopicResult(
            topic=safe_model_dump(topic),
            data_source=data_source,
            decomposition_results=final_results,
        )

    async def _process_single_decomposition(
        self,
        topic,
        decomposition,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> DecompositionResult:
        """
        Process a single decomposition through routing and handler dispatch.

        Args:
            topic: Parent topic
            decomposition: Scientific decomposition to process
            original_query: Original research question
            params: Search parameters

        Returns:
            Complete decomposition result
        """
        search_logger = ContextualLogger("decomposition_processing")
        search_logger.info(f"Processing decomposition: {decomposition.title}")

        # Route decomposition to best repository
        log_component_action(
            "RepositoryRouting",
            "STARTED",
            {"decomposition": decomposition.title},
        )
        routing_output = await self.repository_router_component.process(
            original_query,
            topic,
            decomposition,
        )
        route = routing_output.route

        search_logger.info(
            f"Routed to: {route.repository} (external: {route.is_external})",
        )

        # Handle external sources
        if route.is_external:
            search_logger.info(f"External source: {route.repository}")
            return self._create_external_result(decomposition, route)

        # Dispatch to NASA repository handler
        try:
            # Parse repository enum
            repository = NASARepositoryEnum(route.repository)

            # Check handler status before dispatching
            handler_status = HANDLER_STATUS.get(repository)
            if handler_status == HandlerStatus.STUB:
                search_logger.info(
                    f"Handler for {repository.value} is a stub - returning not implemented result",
                )
                return self._create_stub_result(decomposition, route)

            # Handler is implemented - dispatch to it
            handler = self._get_handler(repository)

            search_logger.info(
                f"Dispatching to {repository.value} handler",
            )

            result = await handler.process_decomposition(
                decomposition,
                topic,
                original_query,
                params,
            )

            return result

        except NotImplementedError as e:
            # Fallback in case handler raises NotImplementedError despite status check
            search_logger.warning(f"Handler not implemented: {e}")
            return self._create_stub_result(decomposition, route)

        except Exception as e:
            search_logger.error(f"Handler dispatch failed: {e}")
            return self._create_error_result(decomposition, route, str(e))

    def _create_error_response(
        self,
        query: str,
        error_msg: str,
    ) -> DataSearchAgentOutputSchema:
        """Create error response."""
        return DataSearchAgentOutputSchema(
            topics=[],
            search_metadata={
                "original_query": query,
                "status": "error",
                "error": error_msg,
                "search_timestamp": datetime.now().isoformat(),
            },
            total_results=0,
        )
