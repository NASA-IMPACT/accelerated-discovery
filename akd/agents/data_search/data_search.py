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

        # Store component configs for creating per-call instances
        # This prevents race conditions in parallel execution
        self.topic_config = BaseAgentConfig(
            model_name=self.config.topic_splitting_model,
        )
        self.decomp_config = BaseAgentConfig(
            model_name=self.config.scientific_decomposition_model,
        )
        self.router_config = BaseAgentConfig(
            model_name=self.config.repository_routing_model,
        )

        # Create singleton for topic splitting (only called once per query, never parallel)
        self.topic_splitting_component = TopicSplittingComponent(
            config=self.topic_config,
            debug=debug,
        )

        # Handlers - initialized lazily when needed
        self._handlers = {}

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
                single_path_mode=self.config.single_path_mode,
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

    @property
    def scientific_decomposition_component(self):
        """
        Create and return a fresh ScientificDecompositionComponent instance.

        This property creates a new component instance each time it's accessed to avoid
        race conditions in parallel execution. Use for testing/demos only.

        Returns:
            Fresh ScientificDecompositionComponent instance
        """
        return ScientificDecompositionComponent(
            config=self.decomp_config,
            debug=self.config.debug,
        )

    @property
    def repository_router_component(self):
        """
        Create and return a fresh RepositoryRouterComponent instance.

        This property creates a new component instance each time it's accessed to avoid
        race conditions in parallel execution. Use for testing/demos only.

        Returns:
            Fresh RepositoryRouterComponent instance
        """
        return RepositoryRouterComponent(
            config=self.router_config,
            debug=self.config.debug,
        )

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
            total_results_from_cmr=0,
            total_results_after_filtering=0,
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
            total_results_from_cmr=0,
            total_results_after_filtering=0,
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
            total_results_from_cmr=0,
            total_results_after_filtering=0,
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

        # Import metadata utilities
        from .utils.metadata import generate_run_id, generate_search_id

        # Generate unique run ID for this search
        run_id = generate_run_id()

        # Generate search ID using run_id
        search_id = generate_search_id(original_query, run_id)

        try:
            # Step 1: Topic Splitting
            # Recreate with run_id for prompt saving
            topic_splitting_component = TopicSplittingComponent(
                config=self.topic_config,
                debug=self.config.debug,
                run_id=run_id,
            )
            topics_output = await topic_splitting_component.process(original_query)

            # Select topics based on execution mode
            if self.config.single_path_mode and topics_output.topics:
                topics_to_process = [topics_output.topics[0]]
                if self.config.debug:
                    print(
                        f"Single-path mode: Processing topic[0], "
                        f"skipping {len(topics_output.topics) - 1} others",
                    )
            else:
                topics_to_process = topics_output.topics

            # Step 2: Process topics in parallel
            topic_tasks = []
            for topic in topics_to_process:
                task = self._process_single_topic(topic, original_query, params, run_id)
                topic_tasks.append(task)

            topic_results = await asyncio.gather(*topic_tasks, return_exceptions=True)

            # Handle any exceptions from parallel execution
            final_topic_results = []
            for i, result in enumerate(topic_results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = TopicResult(
                        topic=safe_model_dump(topics_output.topics[i]),
                        decomposition_results=[],
                        total_cmr_results=0,
                        total_filtered_results=0,
                        note=f"Processing error: {str(result)}",
                    )
                    final_topic_results.append(error_result)
                else:
                    final_topic_results.append(result)

            # Calculate totals - aggregate across all topics
            total_cmr_all = sum(tr.total_cmr_results for tr in final_topic_results)
            total_filtered_all = sum(
                tr.total_filtered_results for tr in final_topic_results
            )

            search_duration = (datetime.now() - search_start_time).total_seconds()

            # Build search metadata
            search_metadata = {
                "search_id": search_id,
                "original_query": original_query,
                "timestamp": search_start_time.isoformat(),
                "duration_seconds": search_duration,
                "topics_processed": len(final_topic_results),
                "single_path_mode": self.config.single_path_mode,
                "workflow_version": "multi-repo-v1",
            }

            # Create response
            final_response = DataSearchAgentOutputSchema(
                topics=final_topic_results,
                search_metadata=search_metadata,
                total_cmr_results=total_cmr_all,
                total_filtered_results=total_filtered_all,
            )

            # Auto-save if configured
            if self.config.auto_save:
                from .utils.metadata import build_output_filename, save_with_metadata

                output_file = build_output_filename(search_id)
                save_with_metadata(final_response, self.config, output_file)
                if self.config.debug:
                    print(f"Results auto-saved to {output_file}")

            return final_response

        except Exception as e:
            error_msg = f"Multi-repository data search failed: {e}"
            return self._create_error_response(original_query, error_msg)

    async def _process_single_topic(
        self,
        topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
        run_id: str,
    ) -> TopicResult:
        """
        Process a single topic through decomposition and routing.

        Args:
            topic: Topic to process
            original_query: Original research question
            params: Search parameters
            run_id: Unique ID for this search run

        Returns:
            Complete topic result with all decompositions
        """
        # Create fresh decomposition component for this topic to avoid race conditions
        # in parallel execution (multiple topics may be processing simultaneously)
        decomposition_component = ScientificDecompositionComponent(
            config=self.decomp_config,
            debug=self.config.debug,
            run_id=run_id,
        )

        # Scientific Decomposition
        decomp_output = await decomposition_component.process(
            original_query,
            topic,
        )

        # Select decompositions based on execution mode
        if self.config.single_path_mode and decomp_output.decompositions:
            decompositions_to_process = [decomp_output.decompositions[0]]
            if self.config.debug:
                print(
                    f"Single-path mode: Processing decomp[0], "
                    f"skipping {len(decomp_output.decompositions) - 1} others",
                )
        else:
            decompositions_to_process = decomp_output.decompositions

        # Process each decomposition in parallel
        decomp_tasks = []
        for decomp in decompositions_to_process:
            task = self._process_single_decomposition(
                topic,
                decomp,
                original_query,
                params,
                run_id,
            )
            decomp_tasks.append(task)

        decomp_results = await asyncio.gather(*decomp_tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(decomp_results):
            if isinstance(result, Exception):
                error_result = DecompositionResult(
                    decomposition=safe_model_dump(decomp_output.decompositions[i]),
                    repository=None,
                    query_approaches=[],
                    searchable_queries=[],
                    data_results=[],
                    total_results_from_cmr=0,
                    total_results_after_filtering=0,
                    note=f"Processing error: {str(result)}",
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        # Aggregate statistics from decomposition results
        total_cmr = sum(dr.total_results_from_cmr for dr in final_results)
        total_filtered = sum(dr.total_results_after_filtering for dr in final_results)

        return TopicResult(
            topic=safe_model_dump(topic),
            decomposition_results=final_results,
            total_cmr_results=total_cmr,
            total_filtered_results=total_filtered,
        )

    async def _process_single_decomposition(
        self,
        topic,
        decomposition,
        original_query: str,
        params: DataSearchAgentInputSchema,
        run_id: str,
    ) -> DecompositionResult:
        """
        Process a single decomposition through routing and handler dispatch.

        Args:
            topic: Parent topic
            decomposition: Scientific decomposition to process
            original_query: Original research question
            params: Search parameters
            run_id: Unique ID for this search run

        Returns:
            Complete decomposition result
        """
        # Create fresh router component for this decomposition to avoid race conditions
        # in parallel execution (multiple decompositions may be processing simultaneously)
        router_component = RepositoryRouterComponent(
            config=self.router_config,
            debug=self.config.debug,
            run_id=run_id,
        )

        # Route decomposition to best repository
        routing_output = await router_component.process(
            original_query,
            topic,
            decomposition,
        )
        route = routing_output.route

        # Handle external sources
        if route.is_external:
            return self._create_external_result(decomposition, route)

        # Dispatch to NASA repository handler
        try:
            # Parse repository enum
            repository = NASARepositoryEnum(route.repository)

            # Check handler status before dispatching
            handler_status = HANDLER_STATUS.get(repository)
            if handler_status == HandlerStatus.STUB:
                return self._create_stub_result(decomposition, route)

            # Handler is implemented - dispatch to it
            handler = self._get_handler(repository)

            result = await handler.process_decomposition(
                decomposition,
                topic,
                original_query,
                params,
                run_id,
            )

            return result

        except NotImplementedError:
            # Fallback in case handler raises NotImplementedError despite status check
            return self._create_stub_result(decomposition, route)

        except Exception as e:
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
            total_cmr_results=0,
            total_filtered_results=0,
        )
