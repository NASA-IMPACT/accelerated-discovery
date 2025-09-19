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

from .known_parameters import QueryApproach, ScientificDecomposition
from .prompt_loader import load_and_format_prompt, load_prompt_template
from .topic_splitting import Topic


class SearchableQuery(BaseModel):
    """Complete CMR query with known parameters + searchable parameters."""

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
        ...,
        description="Combined keyword string for CMR search",
    )


class SearchableParametersOutput(BaseModel):
    """Output from searchable parameters component."""

    searchable_queries: List[SearchableQuery] = Field(
        ...,
        description="Complete queries with known + searchable parameters",
        min_items=1,
        max_items=5,
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
                logger.debug(
                    f"Processing query approach {i + 1}: {self._summarize_approach(approach)}",
                )

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

            # Generate keywords for this approach
            keywords_output = await self._generate_keywords_with_retry(approach)
            searchable_queries.append(keywords_output)

        # Generate overall strategy explanation
        strategy_explanation = self._generate_strategy_explanation(
            decomposition,
            searchable_queries,
        )

        return SearchableParametersOutput(
            searchable_queries=searchable_queries,
            keyword_strategy=strategy_explanation,
        )

    async def _generate_keywords_with_retry(
        self,
        approach: QueryApproach,
    ) -> SearchableQuery:
        """Generate keywords for a single approach with retry logic."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Get keyword suggestions from LLM
                # We'll create a simpler output model for this internal call
                from pydantic import BaseModel

                class KeywordSuggestion(BaseModel):
                    primary_keywords: List[str] = Field(max_items=4)
                    alternative_keywords: List[str] = Field(max_items=3)
                    reasoning: str

                # Temporarily override output schema for this call
                original_output_schema = self.output_schema
                self.output_schema = KeywordSuggestion

                keyword_response = await self.get_response_async()

                # Restore original output schema
                self.output_schema = original_output_schema

                # Combine keywords into search string
                all_keywords = (
                    keyword_response.primary_keywords
                    + keyword_response.alternative_keywords
                )
                combined_string = " ".join(
                    all_keywords[:3],
                )  # Limit to top 3 for search

                # Create complete searchable query
                return SearchableQuery(
                    # Copy known parameters
                    instrument=approach.instrument,
                    platform=approach.platform,
                    processing_level=approach.processing_level,
                    temporal=approach.temporal,
                    bounding_box=approach.bounding_box,
                    temporal_resolution=approach.temporal_resolution,
                    spatial_resolution=approach.spatial_resolution,
                    # Add searchable parameters
                    primary_keywords=keyword_response.primary_keywords,
                    alternative_keywords=keyword_response.alternative_keywords,
                    combined_keyword_string=combined_string,
                )

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to generate keywords after {max_retries + 1} attempts: {e}"
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
                    error_msg = f"Failed to generate keywords: {e}"
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
        approach_summary = self._summarize_approach(approach)

        return load_and_format_prompt(
            "searchable_parameters_user",
            original_query=original_query,
            topic_title=topic.title,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            query_approach_summary=approach_summary,
        )

    def _summarize_approach(self, approach: QueryApproach) -> str:
        """Create a summary of known parameters for the prompt."""
        parts = []
        if approach.instrument:
            parts.append(f"Instrument: {approach.instrument}")
        if approach.platform:
            parts.append(f"Platform: {approach.platform}")
        if approach.temporal:
            parts.append(f"Temporal: {approach.temporal}")
        if approach.bounding_box:
            parts.append(f"Spatial: {approach.bounding_box}")
        if approach.processing_level:
            parts.append(f"Processing Level: {approach.processing_level}")

        return "; ".join(parts) if parts else "No specific known parameters identified"

    def _generate_strategy_explanation(
        self,
        decomposition: ScientificDecomposition,
        searchable_queries: List[SearchableQuery],
    ) -> str:
        """Generate explanation of keyword selection strategy."""
        total_approaches = len(searchable_queries)
        unique_keywords = set()

        for query in searchable_queries:
            unique_keywords.update(query.primary_keywords)
            unique_keywords.update(query.alternative_keywords)

        return (
            f"Generated searchable parameters for {total_approaches} query approaches "
            f"targeting '{decomposition.title}'. Using {len(unique_keywords)} unique keywords "
            f"to discover datasets through metadata search."
        )

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
