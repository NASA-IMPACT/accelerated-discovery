"""
Collection Ranking Component.

Uses LLM to rank and select the most relevant collections for a scientific angle.
"""

from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig

from ._base import BaseDataSearchComponent


class CollectionRankingInputSchema(InputSchema):
    """Input schema for collection ranking component."""

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
    # Query approach fields (individual fields instead of summary)
    approach_instruments: List[str] = Field(
        default_factory=list,
        description="List of instruments used in search queries",
    )
    approach_platforms: List[str] = Field(
        default_factory=list,
        description="List of platforms used in search queries",
    )
    approach_keywords: List[str] = Field(
        default_factory=list,
        description="List of keywords used in search queries",
    )
    approach_temporal_ranges: List[str] = Field(
        default_factory=list,
        description="List of temporal ranges used in search queries",
    )
    approach_spatial_bounds: List[str] = Field(
        default_factory=list,
        description="List of spatial bounds used in search queries",
    )
    approach_processing_levels: List[str] = Field(
        default_factory=list,
        description="List of processing levels used in search queries",
    )
    approach_temporal_resolutions: List[str] = Field(
        default_factory=list,
        description="List of temporal resolutions from query approaches",
    )
    approach_spatial_resolutions: List[str] = Field(
        default_factory=list,
        description="List of spatial resolutions from query approaches",
    )
    collections: List[Dict[str, Any]] = Field(
        ...,
        description="List of collections to rank and filter",
    )
    max_collections: int = Field(
        default=5,
        description="Maximum number of collections to select",
    )


class RankedCollection(BaseModel):
    """A collection with ranking information."""

    collection_index: int = Field(
        ...,
        description="Index of the collection in the original input list (0-based)",
    )
    relevance_score: float = Field(
        ...,
        description="Relevance score from 0.0 to 1.0",
    )
    ranking_reasoning: str = Field(
        ...,
        description="Explanation for why this collection was selected and ranked",
    )


class CollectionRankingOutput(BaseModel):
    """Output from collection ranking component."""

    ranked_collections: List[RankedCollection] = Field(
        ...,
        description="Top collections ranked by relevance to the scientific angle",
        max_items=10,
    )
    total_collections_reviewed: int = Field(
        ...,
        description="Total number of collections that were evaluated",
    )
    ranking_summary: str = Field(
        ...,
        description="Summary of the ranking process and key selection criteria",
    )


class CollectionRankingComponent(
    BaseDataSearchComponent[CollectionRankingInputSchema, CollectionRankingOutput],
):
    """
    Component for ranking collections by relevance to scientific angles.

    Takes collections found from CMR searches and uses LLM to evaluate and rank
    them based on their relevance to the specific scientific angle and original query.
    Selects the top 3-5 most relevant collections.
    """

    input_schema = CollectionRankingInputSchema
    output_schema = CollectionRankingOutput

    # Base class configuration
    template_name = "collection_ranking"
    default_temperature = 0.0  # Low temperature for consistent ranking
    retry_enabled = False  # No retry for ranking components (legacy)

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the collection ranking component."""
        super().__init__(config=config, debug=debug, template_name=self.template_name)
        logger.info("Collection ranking component initialized")

    async def _arun(
        self,
        params: CollectionRankingInputSchema,
    ) -> CollectionRankingOutput:
        """Execute collection ranking workflow."""
        logger.info(
            f"Ranking {len(params.collections)} collections for scientific angle",
        )

        # Prepare collections summary for LLM
        collections_summary = []
        for i, collection in enumerate(params.collections):
            summary = f"{i + 1}. {collection.get('dataset_id', 'Unknown ID')}"
            if collection.get("title"):
                summary += f" - {collection['title']}"
            if collection.get("abstract"):
                # Truncate long abstracts
                abstract = (
                    collection["abstract"][:200] + "..."
                    if len(collection["abstract"]) > 200
                    else collection["abstract"]
                )
                summary += f"\n   Abstract: {abstract}"
            collections_summary.append(summary)

        # Format user prompt
        user_prompt = self._format_user_prompt_from_template(
            original_query=params.original_query,
            topic_title=params.topic_title,
            topic_context=params.topic_context,
            decomposition_title=params.decomposition_title,
            decomposition_justification=params.decomposition_justification,
            # Individual approach fields
            approach_instruments=", ".join(params.approach_instruments)
            if params.approach_instruments
            else "None specified",
            approach_platforms=", ".join(params.approach_platforms)
            if params.approach_platforms
            else "None specified",
            approach_keywords=", ".join(params.approach_keywords)
            if params.approach_keywords
            else "None specified",
            approach_temporal_ranges=", ".join(params.approach_temporal_ranges)
            if params.approach_temporal_ranges
            else "None specified",
            approach_spatial_bounds=", ".join(params.approach_spatial_bounds)
            if params.approach_spatial_bounds
            else "None specified",
            approach_processing_levels=", ".join(params.approach_processing_levels)
            if params.approach_processing_levels
            else "None specified",
            approach_temporal_resolutions=", ".join(
                params.approach_temporal_resolutions,
            )
            if params.approach_temporal_resolutions
            else "None specified",
            approach_spatial_resolutions=", ".join(params.approach_spatial_resolutions)
            if params.approach_spatial_resolutions
            else "None specified",
            num_collections=len(params.collections),
            collections_list="\n\n".join(collections_summary),
            max_collections=params.max_collections,
        )

        # Set the user prompt and get LLM ranking
        self._set_messages(user_prompt)
        ranking_result = await self.get_response_async()

        # Map collection indices back to actual collection data
        mapped_collections = []
        for ranked_col in ranking_result.ranked_collections:
            if 0 <= ranked_col.collection_index < len(params.collections):
                mapped_collections.append(
                    RankedCollection(
                        collection_index=ranked_col.collection_index,
                        relevance_score=ranked_col.relevance_score,
                        ranking_reasoning=ranked_col.ranking_reasoning,
                    ),
                )
            else:
                logger.warning(
                    f"Invalid collection index: {ranked_col.collection_index}",
                )

        # Create final result with mapped collections
        final_result = CollectionRankingOutput(
            ranked_collections=mapped_collections,
            total_collections_reviewed=len(params.collections),
            ranking_summary=ranking_result.ranking_summary,
        )

        logger.info(
            f"Collection ranking completed: {len(final_result.ranked_collections)} collections selected",
        )

        return final_result
