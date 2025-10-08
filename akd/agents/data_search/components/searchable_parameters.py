"""
Searchable Parameters Component.

Uses LLM to generate search terms and keywords for dataset discovery
when combined with known parameters.
"""

import asyncio
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from ..utils.prompt_loader import load_and_format_prompt, load_prompt_template
from .known_parameters import QueryApproach, ScientificDecomposition
from .topic_splitting import Topic


class SearchableQuery(BaseModel):
    """Complete CMR query with known parameters + searchable parameters."""

    # Approach tracking
    approach_index: int = Field(
        ...,
        description="Index of source QueryApproach (0-based)",
    )

    # Known parameters (carried forward)
    instrument: Optional[str] = Field(None, description="Instrument name")
    platform: Optional[str] = Field(None, description="Platform/satellite name")
    processing_level: Optional[str] = Field(None, description="Data processing level")
    temporal: Optional[str] = Field(None, description="Temporal range in ISO format")
    bounding_box: Optional[str] = Field(None, description="Spatial bounding box")
    temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution",
    )
    spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution",
    )

    # Searchable parameters (generated)
    primary_keywords: List[str] = Field(
        default_factory=list,
        description="Primary search keywords for CMR metadata search",
        max_items=4,
    )
    alternative_keywords: List[str] = Field(
        default_factory=list,
        description="Alternative/synonym keywords",
        max_items=3,
    )
    combined_keyword_string: str = Field(
        default="",
        description="Combined keyword string for CMR search (empty string if no keywords needed)",
    )

    def get_mcp_parameters(self) -> dict[str, str]:
        """
        Get parameters that can be passed directly to CMR MCP API.

        Returns:
            Dictionary with only CMR-compatible search parameters
        """
        mcp_params = {}

        if self.instrument:
            mcp_params["instrument"] = self.instrument
        if self.platform:
            mcp_params["platform"] = self.platform
        if self.processing_level:
            mcp_params["processing_level"] = self.processing_level
        if self.temporal:
            mcp_params["temporal"] = self.temporal
        if self.bounding_box:
            mcp_params["bounding_box"] = self.bounding_box

        # Add searchable parameters only if they exist
        if self.combined_keyword_string and self.combined_keyword_string.strip():
            mcp_params["keyword"] = self.combined_keyword_string

        return mcp_params

    def get_filter_parameters(self) -> dict[str, str]:
        """
        Get parameters that must be used for post-search filtering.

        Returns:
            Dictionary with resolution and other filter-only parameters
        """
        filter_params = {}

        if self.temporal_resolution:
            filter_params["temporal_resolution"] = self.temporal_resolution
        if self.spatial_resolution:
            filter_params["spatial_resolution"] = self.spatial_resolution

        return filter_params


class SearchableParametersOutput(BaseModel):
    """Output from searchable parameters component."""

    searchable_queries: List[SearchableQuery] = Field(
        ...,
        description="Complete queries with known + searchable parameters (expanded from approaches)",
        min_items=1,
        max_items=25,  # Up to 5 approaches × 5 variations each
    )
    keyword_strategy: str = Field(
        ...,
        description="Explanation of keyword selection strategy",
    )


class SearchableParametersInputSchema(InputSchema):
    """Input schema for searchable parameters generation."""

    original_query: str = Field(..., description="Original research question")
    topic: Topic = Field(..., description="Topic being processed")
    decomposition: ScientificDecomposition = Field(
        ...,
        description="Scientific decomposition",
    )
    query_approaches: List[QueryApproach] = Field(
        ...,
        description="Known parameter approaches",
    )


class SearchableParametersComponent(
    InstructorBaseAgent[SearchableParametersInputSchema, SearchableParametersOutput],
):
    """
    Component for generating searchable parameters using LLM.

    Takes known parameter approaches and adds keyword search terms to create
    complete CMR queries ready for execution.
    """

    input_schema = SearchableParametersInputSchema
    output_schema = SearchableParametersOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the searchable parameters component."""
        # Set up specialized configuration for searchable parameters
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for searchable parameters
        config.system_prompt = load_prompt_template("searchable_parameters_system")
        config.temperature = 0.1  # Low temperature for consistent keyword generation

        super().__init__(config=config, debug=debug)

    async def process(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
        query_approaches: List[QueryApproach],
    ) -> SearchableParametersOutput:
        """
        Generate searchable parameters for query approaches.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification
            query_approaches: Known parameter approaches to enhance

        Returns:
            SearchableParametersOutput with complete queries
        """
        if self.debug:
            logger.debug(
                f"Generating searchable parameters for {len(query_approaches)} query approaches",
            )

        searchable_queries = []

        for i, approach in enumerate(query_approaches):
            if self.debug:
                parts = []
                if approach.instrument:
                    parts.append(f"instrument: {approach.instrument}")
                if approach.platform:
                    parts.append(f"platform: {approach.platform}")
                if approach.temporal:
                    parts.append(f"temporal: {approach.temporal[:10]}...")
                summary = ", ".join(parts) if parts else "no specific parameters"
                logger.debug(f"Processing query approach {i + 1}: {summary}")

            # Format the user prompt for this specific approach
            user_prompt = self._format_user_prompt(
                original_query,
                topic,
                decomposition,
                approach,
            )

            # Clear previous context and add user message to memory for each approach
            self.memory.clear()
            self.memory.append({"role": "user", "content": user_prompt})

            # Generate multiple search variations for this approach
            approach_queries = await self._generate_search_variations_with_retry(
                approach,
                i,
            )
            searchable_queries.extend(approach_queries)

        # Generate overall strategy explanation
        strategy_explanation = self._generate_strategy_explanation(
            decomposition,
            searchable_queries,
        )

        return SearchableParametersOutput(
            searchable_queries=searchable_queries,
            keyword_strategy=strategy_explanation,
        )

    async def _generate_search_variations_with_retry(
        self,
        approach: QueryApproach,
        approach_index: int,
    ) -> List[SearchableQuery]:
        """Generate multiple search variations for a single approach with retry logic."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Get search variation suggestions from LLM
                from pydantic import BaseModel

                class SearchVariations(BaseModel):
                    search_queries: List[str] = Field(
                        description="List of keyword combinations for separate searches (0-5 queries). Empty string means no additional keywords needed.",
                        max_items=5,
                    )
                    reasoning: str = Field(description="Explanation of search strategy")

                # Temporarily override output schema for this call
                original_output_schema = self.output_schema
                self.output_schema = SearchVariations

                variations_response = await self.get_response_async()

                # Restore original output schema
                self.output_schema = original_output_schema

                # Create searchable queries for each variation
                searchable_queries = []
                for keyword_string in variations_response.search_queries:
                    # Parse keywords for metadata (though we primarily use the combined string)
                    keywords_list = (
                        keyword_string.split() if keyword_string.strip() else []
                    )
                    primary_keywords = keywords_list[:4] if keywords_list else []
                    alternative_keywords = (
                        keywords_list[4:7] if len(keywords_list) > 4 else []
                    )

                    searchable_queries.append(
                        SearchableQuery(
                            # Track source approach
                            approach_index=approach_index,
                            # Copy known parameters
                            instrument=approach.instrument,
                            platform=approach.platform,
                            processing_level=approach.processing_level,
                            temporal=approach.temporal,
                            bounding_box=approach.bounding_box,
                            temporal_resolution=approach.temporal_resolution,
                            spatial_resolution=approach.spatial_resolution,
                            # Add searchable parameters
                            primary_keywords=primary_keywords,
                            alternative_keywords=alternative_keywords,
                            combined_keyword_string=keyword_string.strip(),
                        ),
                    )

                return searchable_queries

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to generate search variations after {max_retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # Check if it's a rate limit error
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = base_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})",
                        )
                    await asyncio.sleep(delay)
                else:
                    # Non-rate-limit error, don't retry
                    error_msg = f"Failed to generate search variations: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _format_user_prompt(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
        approach: QueryApproach,
    ) -> str:
        """Format the user prompt with research context and query approach."""
        return load_and_format_prompt(
            "searchable_parameters_user",
            original_query=original_query,
            topic_title=topic.title,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            # Pass individual approach fields directly
            approach_instrument=approach.instrument or "",
            approach_platform=approach.platform or "",
            approach_processing_level=approach.processing_level or "",
            approach_temporal=approach.temporal or "",
            approach_bounding_box=approach.bounding_box or "",
            approach_temporal_resolution=approach.temporal_resolution or "",
            approach_spatial_resolution=approach.spatial_resolution or "",
        )

    def _generate_strategy_explanation(
        self,
        decomposition: ScientificDecomposition,
        searchable_queries: List[SearchableQuery],
    ) -> str:
        """Generate explanation of search variation strategy."""
        total_queries = len(searchable_queries)
        queries_with_keywords = sum(
            1 for q in searchable_queries if q.combined_keyword_string.strip()
        )
        queries_without_keywords = total_queries - queries_with_keywords

        unique_keywords = set()
        for query in searchable_queries:
            if query.combined_keyword_string.strip():
                unique_keywords.update(query.combined_keyword_string.split())

        strategy_parts = [
            f"Generated {total_queries} search variations targeting '{decomposition.title}'.",
        ]

        if queries_without_keywords > 0:
            strategy_parts.append(
                f"{queries_without_keywords} searches use only known parameters.",
            )

        if queries_with_keywords > 0:
            strategy_parts.append(
                f"{queries_with_keywords} searches add focused keywords from {len(unique_keywords)} unique terms.",
            )

        return " ".join(strategy_parts)

    async def _arun(
        self,
        params: SearchableParametersInputSchema,
        **kwargs,
    ) -> SearchableParametersOutput:
        """Execute the searchable parameters generation."""
        return await self.process(
            params.original_query,
            params.topic,
            params.decomposition,
            params.query_approaches,
        )
