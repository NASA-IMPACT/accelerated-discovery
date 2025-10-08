"""
Final Collection Ranking Component.

Performs final cross-approach ranking of pre-filtered collections using LLM.
"""

from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent

from ..utils.prompt_loader import load_prompt_template


class FinalCollectionRankingInputSchema(InputSchema):
    """Input schema for final cross-approach ranking."""

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

    collections: List[Dict[str, Any]] = Field(
        ...,
        max_items=25,
        description="Pre-filtered collections from all approaches",
    )
    max_collections: int = Field(
        default=25,
        description="Maximum collections to return (ranked 1-N)",
    )


class FinalRankedCollection(BaseModel):
    """Collection with final rank across all approaches."""

    collection_index: int = Field(
        ...,
        description="Index in the input collections list (0-based)",
    )
    final_rank: int = Field(
        ...,
        ge=1,
        le=25,
        description="Final rank (1=best)",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final relevance score",
    )
    ranking_reasoning: str = Field(
        ...,
        description="Why this collection received this rank",
    )


class FinalCollectionRankingOutput(BaseModel):
    """Output from final ranking."""

    ranked_collections: List[FinalRankedCollection] = Field(
        ...,
        max_items=25,
        description="Collections ranked 1-25 (or fewer if less available)",
    )
    total_collections_ranked: int = Field(
        ...,
        description="Total number of collections in the ranking",
    )
    ranking_summary: str = Field(
        ...,
        description="Summary of ranking decisions and key differentiators",
    )


class FinalCollectionRankingComponent(
    InstructorBaseAgent[
        FinalCollectionRankingInputSchema,
        FinalCollectionRankingOutput,
    ],
):
    """
    Component for final ranking across all query approaches.

    No filtering - only comparative ranking of pre-filtered collections.
    """

    input_schema = FinalCollectionRankingInputSchema
    output_schema = FinalCollectionRankingOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the final collection ranking component."""
        if config is None:
            config = BaseAgentConfig()

        # Load and set system prompt in config
        config.system_prompt = load_prompt_template("final_ranking_system")
        self.user_prompt_template = load_prompt_template("final_ranking_user")

        config.temperature = 0.0  # Consistent ranking

        super().__init__(config=config, debug=debug)

        logger.info("Final collection ranking component initialized")

    async def _arun(
        self,
        params: FinalCollectionRankingInputSchema,
    ) -> FinalCollectionRankingOutput:
        """Execute final cross-approach ranking."""
        logger.info(f"Final ranking of {len(params.collections)} collections")

        # Prepare collection summaries
        collections_summary = []
        for i, collection in enumerate(params.collections):
            summary = f"{i}. {collection.get('dataset_id', 'Unknown')}"

            if collection.get("title"):
                summary += f"\n   Title: {collection['title']}"

            if collection.get("abstract"):
                abstract = (
                    collection["abstract"][:300] + "..."
                    if len(collection["abstract"]) > 300
                    else collection["abstract"]
                )
                summary += f"\n   Abstract: {abstract}"

            # Include key distinguishing features
            if collection.get("instrument"):
                summary += f"\n   Instrument: {collection.get('instrument')}"

            if collection.get("platform"):
                summary += f"\n   Platform: {collection.get('platform')}"

            if collection.get("processing_level_id"):
                summary += f"\n   Level: {collection.get('processing_level_id')}"

            collections_summary.append(summary)

        user_prompt = self.user_prompt_template.format(
            original_query=params.original_query,
            topic_title=params.topic_title,
            topic_context=params.topic_context,
            decomposition_title=params.decomposition_title,
            decomposition_justification=params.decomposition_justification,
            num_collections=len(params.collections),
            collections_list="\n\n".join(collections_summary),
            max_collections=params.max_collections,
        )

        self.messages = [{"role": "user", "content": user_prompt}]
        result = await self.get_response_async()

        logger.info(
            f"Final ranking complete: {len(result.ranked_collections)} "
            "collections ranked",
        )

        return result
