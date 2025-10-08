"""
Approach Collection Filtering Component.

Filters and ranks collections within a single query approach using LLM evaluation.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from ..utils.prompt_loader import load_prompt_template


class ApproachCollectionFilteringInputSchema(InputSchema):
    """Input schema for filtering collections within a single approach."""

    original_query: str = Field(
        ...,
        description="The original scientific research question",
    )
    topic_title: str = Field(
        ...,
        description="Title of the functional topic being processed",
    )
    topic_context: str = Field(
        ...,
        description="Functional context explaining why this is a distinct area",
    )
    decomposition_title: str = Field(
        ...,
        description="Title of the scientific decomposition",
    )
    decomposition_justification: str = Field(
        ...,
        description="Scientific justification for the decomposition",
    )

    # Single approach context
    approach_instrument: Optional[str] = Field(
        None,
        description="Instrument for this query approach",
    )
    approach_platform: Optional[str] = Field(
        None,
        description="Platform for this query approach",
    )
    approach_processing_level: Optional[str] = Field(
        None,
        description="Processing level for this query approach",
    )
    approach_temporal_range: Optional[str] = Field(
        None,
        description="Temporal range for this query approach",
    )
    approach_spatial_bounds: Optional[str] = Field(
        None,
        description="Spatial bounds for this query approach",
    )
    approach_temporal_resolution: Optional[str] = Field(
        None,
        description="Required temporal resolution for this query approach",
    )
    approach_spatial_resolution: Optional[str] = Field(
        None,
        description="Required spatial resolution for this query approach",
    )
    approach_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords used in this query approach",
    )

    collections: List[Dict[str, Any]] = Field(
        ...,
        max_items=25,
        description="Collections to filter (from this approach)",
    )
    max_collections: int = Field(
        default=5,
        description="Maximum collections to select from this approach",
    )


class FilteredRankedCollection(BaseModel):
    """Collection that passed filtering, with rank within approach."""

    collection_index: int = Field(
        ...,
        description="Index in the input collections list (0-based)",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score for this collection",
    )
    ranking_reasoning: str = Field(
        ...,
        description="Why this collection was selected and how it ranks",
    )


class ApproachCollectionFilteringOutput(BaseModel):
    """Output from per-approach filtering."""

    selected_collections: List[FilteredRankedCollection] = Field(
        ...,
        max_items=5,
        description="Top collections for this approach, ranked",
    )
    total_reviewed: int = Field(
        ...,
        description="Total number of collections reviewed",
    )
    total_filtered_out: int = Field(
        ...,
        description="Number of collections filtered out",
    )
    filtering_summary: str = Field(
        ...,
        description="Summary of filtering decisions and selection rationale",
    )


class ApproachCollectionFilteringComponent(
    InstructorBaseAgent[
        ApproachCollectionFilteringInputSchema,
        ApproachCollectionFilteringOutput,
    ],
):
    """
    Component for filtering and ranking collections within a single query approach.

    Two-part process:
    1. Filter out obviously bad matches (spatial/temporal/resolution mismatches)
    2. Select and rank the best 0-5 collections from remaining candidates
    """

    input_schema = ApproachCollectionFilteringInputSchema
    output_schema = ApproachCollectionFilteringOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the approach collection filtering component."""
        if config is None:
            config = BaseAgentConfig()

        # Load and set system prompt in config
        config.system_prompt = load_prompt_template("approach_filtering_system")
        self.user_prompt_template = load_prompt_template("approach_filtering_user")

        config.temperature = 0.0  # Consistent filtering

        super().__init__(config=config, debug=debug)

        logger.info("Approach collection filtering component initialized")

    async def _arun(
        self,
        params: ApproachCollectionFilteringInputSchema,
    ) -> ApproachCollectionFilteringOutput:
        """Execute approach-level filtering and ranking."""
        logger.info(
            f"Filtering {len(params.collections)} collections for approach "
            f"(instrument: {params.approach_instrument}, "
            f"platform: {params.approach_platform})",
        )

        # Prepare collection summaries (limit metadata for token efficiency)
        collections_summary = []
        for i, collection in enumerate(params.collections):
            # Create concise summary with essential fields
            summary_parts = [f"{i}. {collection.get('dataset_id', 'Unknown ID')}"]

            if collection.get("title"):
                summary_parts.append(f"   Title: {collection['title']}")

            if collection.get("abstract"):
                abstract = (
                    collection["abstract"][:300] + "..."
                    if len(collection["abstract"]) > 300
                    else collection["abstract"]
                )
                summary_parts.append(f"   Abstract: {abstract}")

            # Key metadata for filtering
            if collection.get("time_start") or collection.get("time_end"):
                temporal = (
                    f"{collection.get('time_start', 'N/A')} to "
                    f"{collection.get('time_end', 'N/A')}"
                )
                summary_parts.append(f"   Temporal: {temporal}")

            if collection.get("boxes"):
                summary_parts.append(f"   Spatial: {collection.get('boxes')}")

            if collection.get("processing_level_id"):
                summary_parts.append(
                    f"   Level: {collection.get('processing_level_id')}",
                )

            collections_summary.append("\n".join(summary_parts))

        # Format user prompt
        user_prompt = self.user_prompt_template.format(
            original_query=params.original_query,
            topic_title=params.topic_title,
            topic_context=params.topic_context,
            decomposition_title=params.decomposition_title,
            decomposition_justification=params.decomposition_justification,
            approach_instrument=params.approach_instrument or "Not specified",
            approach_platform=params.approach_platform or "Not specified",
            approach_processing_level=params.approach_processing_level
            or "Not specified",
            approach_temporal_range=params.approach_temporal_range or "Not specified",
            approach_spatial_bounds=params.approach_spatial_bounds or "Not specified",
            approach_temporal_resolution=params.approach_temporal_resolution
            or "Not specified",
            approach_spatial_resolution=params.approach_spatial_resolution
            or "Not specified",
            approach_keywords=", ".join(params.approach_keywords)
            if params.approach_keywords
            else "None",
            num_collections=len(params.collections),
            collections_list="\n\n".join(collections_summary),
            max_collections=params.max_collections,
        )

        self.messages = [{"role": "user", "content": user_prompt}]
        result = await self.get_response_async()

        logger.info(
            f"Approach filtering complete: {len(result.selected_collections)} "
            f"selected from {result.total_reviewed} reviewed",
        )

        return result
