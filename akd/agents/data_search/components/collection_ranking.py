"""
Collection Ranking Component.

Uses LLM to rank and select the most relevant collections for a scientific angle.
"""

from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from .prompt_loader import load_prompt_template


class CollectionRankingInputSchema(InputSchema):
    """Input schema for collection ranking component."""

    original_query: str = Field(
        ...,
        description="The original scientific research question",
    )
    scientific_angle: Dict[str, Any] = Field(
        ...,
        description="The scientific angle these collections relate to",
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
    InstructorBaseAgent[CollectionRankingInputSchema, CollectionRankingOutput],
):
    """
    Component for ranking collections by relevance to scientific angles.

    Takes collections found from CMR searches and uses LLM to evaluate and rank
    them based on their relevance to the specific scientific angle and original query.
    Selects the top 3-5 most relevant collections.
    """

    input_schema = CollectionRankingInputSchema
    output_schema = CollectionRankingOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the collection ranking component."""
        # Set up specialized configuration for collection ranking
        if config is None:
            config = BaseAgentConfig()

        # Load and set system prompt in config
        config.system_prompt = load_prompt_template("collection_ranking_system")
        self.user_prompt_template = load_prompt_template("collection_ranking_user")

        config.temperature = 0.0  # Low temperature for consistent ranking

        super().__init__(config=config, debug=debug)

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
        angle_data = params.scientific_angle
        user_prompt = self.user_prompt_template.format(
            original_query=params.original_query,
            angle_title=angle_data.get("title", "Unknown angle"),
            angle_justification=angle_data.get(
                "scientific_justification",
                "No justification provided",
            ),
            num_collections=len(params.collections),
            collections_list="\n\n".join(collections_summary),
            max_collections=params.max_collections,
        )

        # Set the user prompt and get LLM ranking
        self.messages = [{"role": "user", "content": user_prompt}]
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
