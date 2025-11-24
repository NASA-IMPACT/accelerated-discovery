"""
Shared logic for parameter extraction components.

Contains generic implementations for:
- Known Parameters: Extract hard filters from research context
- Searchable Parameters: Generate keyword search variations

Repository-specific subclasses only need to define schemas and template names.
"""

from typing import Generic, List, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from ..constants import (
    CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH,
    CMR_MIN_SEARCH_VARIATIONS_PER_APPROACH,
)
from ..utils.prompt_loader import load_and_format_prompt
from ._base import BaseDataSearchComponent, TInput, TOutput
from ._base_parameters import (
    BaseKnownParametersComponent,
    BaseSearchableParametersComponent,
)

# Type variables for generic components
TQueryApproach = TypeVar("TQueryApproach", bound=BaseModel)
TSearchableQuery = TypeVar("TSearchableQuery", bound=BaseModel)


class SearchVariations(BaseModel):
    """Search variations for a single query approach."""

    search_strings: List[str] = Field(
        description="List of search strings for separate searches. Empty string means no additional search string needed (use known parameters only).",
        min_items=CMR_MIN_SEARCH_VARIATIONS_PER_APPROACH,
        max_items=CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH,
        default_factory=list,
    )
    reasoning: str = Field(
        description="Explanation of search strategy for this approach",
        default="Generated search variations based on approach parameters.",
    )


class SharedKnownParametersComponent(
    BaseDataSearchComponent[TInput, TOutput],
    BaseKnownParametersComponent[TQueryApproach],
    Generic[TInput, TOutput, TQueryApproach],
):
    """
    Shared implementation for known parameters extraction.

    Generic logic for extracting hard filters (instruments, temporal/spatial bounds, etc.)
    from scientific research context using LLM.

    Repository-specific subclasses should:
    - Set input_schema and output_schema
    - Set template_name (e.g., "known_parameters")
    - Define repository-specific TQueryApproach schema
    """

    async def process(
        self,
        original_query: str,
        topic,  # Topic from topic_splitting
        decomposition,  # ScientificDecomposition
    ) -> TOutput:
        """
        Extract known parameters from research context.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification

        Returns:
            Output with query approaches (repository-specific schema)
        """
        if self.debug:
            logger.debug(
                f"Extracting known parameters for decomposition: '{decomposition.title}'",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(original_query, topic, decomposition)

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "known_parameters",
            decomposition.title,
        )

        # Add user message to memory
        self._add_user_message(user_prompt)

        # Execute with retry logic
        response = await self._execute_with_retry(
            operation_name="extract known parameters",
            custom_error_prefix="Failed to extract known parameters",
        )

        if self.debug:
            logger.debug(
                f"Generated {len(response.query_approaches)} query approaches",
            )

        return response

    def _format_user_prompt(
        self,
        original_query: str,
        topic,
        decomposition,
    ) -> str:
        """Format the user prompt with research context."""
        # Get min/max values from output schema
        # metadata[0] = MinLen, metadata[1] = MaxLen
        min_approaches = self.output_schema.model_fields["query_approaches"].metadata[0].min_length
        max_approaches = self.output_schema.model_fields["query_approaches"].metadata[1].max_length

        return load_and_format_prompt(
            f"{self.template_name}_user",
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            min_approaches=min_approaches,
            max_approaches=max_approaches,
            prompts_dir=self.prompts_dir,
        )

    async def _arun(
        self,
        params: TInput,
        **kwargs,
    ) -> TOutput:
        """Execute the known parameters extraction."""
        return await self.process(
            params.original_query,
            params.topic,
            params.decomposition,
        )


class SharedSearchableParametersComponent(
    BaseDataSearchComponent[TInput, TOutput],
    BaseSearchableParametersComponent[TQueryApproach, TSearchableQuery],
    Generic[TInput, TOutput, TQueryApproach, TSearchableQuery],
):
    """
    Shared implementation for searchable parameters generation.

    Generic logic for generating keyword search variations combined with known parameters
    using LLM.

    Repository-specific subclasses should:
    - Set input_schema and output_schema
    - Set template_name (e.g., "searchable_parameters")
    - Define repository-specific TQueryApproach and TSearchableQuery schemas
    - Implement _create_searchable_query() to construct query objects
    """

    async def _process_single_approach(
        self,
        original_query: str,
        topic,
        decomposition,
        approach: TQueryApproach,
        approach_index: int,
    ) -> List[TSearchableQuery]:
        """
        Process a single approach to generate searchable queries.

        Args:
            original_query: Original research query
            topic: Topic being processed
            decomposition: Scientific decomposition
            approach: Query approach to process
            approach_index: Index of this approach

        Returns:
            List of searchable queries for this approach
        """
        if self.debug:
            logger.debug(f"Processing query approach {approach_index + 1}")

        # Format the user prompt for this specific approach
        user_prompt = self._format_user_prompt(
            original_query,
            topic,
            decomposition,
            approach,
        )

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "searchable_parameters",
            f"{decomposition.title}_approach{approach_index}",
        )

        # Clear and set memory for this approach
        # Note: Each approach gets a fresh component instance in parallel execution,
        # so memory state is isolated and thread-safe
        self.memory.clear()
        self.memory.append({"role": "user", "content": user_prompt})

        # Generate multiple search variations for this approach
        approach_queries = await self._generate_search_variations_with_retry(
            approach,
            approach_index,
        )

        return approach_queries

    async def process(
        self,
        original_query: str,
        topic,  # Topic from topic_splitting
        decomposition,  # ScientificDecomposition
        query_approaches: List[TQueryApproach],
    ) -> TOutput:
        """
        Generate searchable parameters for query approaches in parallel.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification
            query_approaches: Known parameter approaches to enhance

        Returns:
            Output with searchable queries (repository-specific schema)
        """
        if self.debug:
            logger.debug(
                f"Generating searchable parameters for {len(query_approaches)} query approaches",
            )

        # Process all approaches in parallel with isolated component instances
        # to avoid race conditions with shared memory state
        import asyncio

        async def process_approach_isolated(approach, index):
            """Process single approach with fresh component instance for thread safety."""
            # Create fresh component for this approach to avoid race conditions
            fresh_component = self.__class__(
                config=self.config,
                debug=self.debug,
                prompts_dir=self.prompts_dir,
                run_id=self.run_id,
            )
            return await fresh_component._process_single_approach(
                original_query,
                topic,
                decomposition,
                approach,
                index,
            )

        approach_tasks = [process_approach_isolated(approach, i) for i, approach in enumerate(query_approaches)]

        # Execute in parallel
        approach_results = await asyncio.gather(*approach_tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        searchable_queries = []
        for i, result in enumerate(approach_results):
            if isinstance(result, Exception):
                if self.debug:
                    logger.warning(
                        f"Failed to process approach {i + 1}: {result}",
                    )
                continue
            searchable_queries.extend(result)

        # Generate overall strategy explanation
        strategy_explanation = self._generate_strategy_explanation(
            decomposition,
            searchable_queries,
        )

        return self._create_output(searchable_queries, strategy_explanation)

    async def _generate_search_variations_with_retry(
        self,
        approach: TQueryApproach,
        approach_index: int,
    ) -> List[TSearchableQuery]:
        """Generate multiple search variations for a single approach with retry logic."""

        # Helper method to do the actual LLM call
        async def _do_llm_call():
            try:
                # Temporarily override output schema for this call
                # Use module-level SearchVariations schema
                original_output_schema = self.output_schema
                self.output_schema = SearchVariations

                messages = [self._default_system_message()] + self.memory
                variations_response = await self.get_response_async(messages=messages)

                # Restore original output schema
                self.output_schema = original_output_schema

                # Create searchable queries for each search string variation
                searchable_queries = []
                for search_string in variations_response.search_strings:
                    # Let subclass create the repository-specific query object
                    searchable_query = self._create_searchable_query(
                        approach=approach,
                        approach_index=approach_index,
                        keyword_string=search_string,
                    )
                    searchable_queries.append(searchable_query)

                return searchable_queries
            except Exception:
                # Propagate exception to be handled by retry wrapper
                raise

        # Use base class retry logic with custom operation name
        return await self._execute_with_retry_custom(
            _do_llm_call,
            operation_name="generate search variations",
            custom_error_prefix="Failed to generate search variations",
        )

    def _format_user_prompt(
        self,
        original_query: str,
        topic,
        decomposition,
        approach: TQueryApproach,
    ) -> str:
        """
        Format the user prompt with research context and query approach.

        Subclasses can override to customize prompt formatting.
        """
        # Get approach fields for prompt formatting
        approach_fields = self._extract_approach_fields(approach)

        # Extract min/max variations from SearchVariations schema
        # metadata[0] = MinLen, metadata[1] = MaxLen
        min_variations = SearchVariations.model_fields["search_strings"].metadata[0].min_length
        max_variations = SearchVariations.model_fields["search_strings"].metadata[1].max_length

        return load_and_format_prompt(
            f"{self.template_name}_user",
            original_query=original_query,
            topic_title=topic.title,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            min_variations=min_variations,
            max_variations=max_variations,
            prompts_dir=self.prompts_dir,
            **approach_fields,
        )

    def _extract_approach_fields(self, approach: TQueryApproach) -> dict:
        """
        Extract approach fields for prompt formatting.

        Default implementation handles common CMR-like fields.
        Subclasses can override for repository-specific fields.
        """
        fields = {}
        common_fields = [
            "instrument",
            "platform",
            "processing_level",
            "temporal",
            "bounding_box",
            "temporal_resolution",
            "spatial_resolution",
        ]

        for field_name in common_fields:
            if hasattr(approach, field_name):
                value = getattr(approach, field_name)
                fields[f"approach_{field_name}"] = value or ""

        return fields

    def _create_searchable_query(
        self,
        approach: TQueryApproach,
        approach_index: int,
        keyword_string: str,
    ) -> TSearchableQuery:
        """
        Create a repository-specific searchable query object.

        Must be implemented by repository-specific subclass.

        Args:
            approach: Source query approach with known parameters
            approach_index: Index of source approach
            keyword_string: Generated keyword string

        Returns:
            Repository-specific searchable query object
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_searchable_query()",
        )

    def _generate_strategy_explanation(
        self,
        decomposition,
        searchable_queries: List[TSearchableQuery],
    ) -> str:
        """
        Generate explanation of search variation strategy.

        Default implementation provides basic statistics.
        Subclasses can override for repository-specific explanations.
        """
        total_queries = len(searchable_queries)

        return f"Generated {total_queries} search variations targeting '{decomposition.title}'."

    def _create_output(
        self,
        searchable_queries: List[TSearchableQuery],
        strategy_explanation: str,
    ) -> TOutput:
        """
        Create output object with searchable queries and strategy explanation.

        Default implementation assumes output_schema has these fields.
        Subclasses can override if needed.
        """
        return self.output_schema(
            searchable_queries=searchable_queries,
            keyword_strategy=strategy_explanation,
        )

    async def _arun(
        self,
        params: TInput,
        **kwargs,
    ) -> TOutput:
        """Execute the searchable parameters generation."""
        return await self.process(
            params.original_query,
            params.topic,
            params.decomposition,
            params.query_approaches,
        )
