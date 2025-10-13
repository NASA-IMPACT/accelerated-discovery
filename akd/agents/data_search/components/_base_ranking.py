"""
Base classes for ranking and filtering components.

Defines abstract interfaces for repository-specific result filtering/ranking:
- Approach filtering: Filter and rank results within a single query approach
- Final ranking: Rank results across all query approaches
"""

from abc import ABC
from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel, Field

from akd._base import InputSchema

from ._base import BaseDataSearchComponent

# Type variable for repository-specific data items
TDataItem = TypeVar("TDataItem")  # Collections for CMR, Bundles for PDS4, etc.


class BaseApproachFilteringInputSchema(InputSchema):
    """
    Base input schema for per-approach filtering.

    Repository-specific implementations may extend this with additional fields.
    """

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

    # Repository-specific approach parameters should be added by subclass
    # e.g., approach: CMRQueryApproach (contains instrument, platform, etc.)
    # e.g., approach: PDS4QueryApproach (contains target, mission, etc.)

    data_items: List[Dict[str, Any]] = Field(
        ...,
        description="Data items to filter (collections/bundles/etc.)",
    )
    max_items: int = Field(
        default=5,
        description="Maximum items to select from this approach",
    )


class FilteredRankedItem(BaseModel):
    """Base schema for filtered/ranked data item."""

    item_index: int = Field(
        ...,
        description="Index in the input data items list (0-based)",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score for this item",
    )
    ranking_reasoning: str = Field(
        ...,
        description="Why this item was selected and how it ranks",
    )


class BaseApproachFilteringOutput(BaseModel):
    """Base output schema for per-approach filtering."""

    selected_items: List[FilteredRankedItem] = Field(
        ...,
        description="Top items for this approach, ranked",
    )
    total_reviewed: int = Field(
        ...,
        description="Total number of items reviewed",
    )
    total_filtered_out: int = Field(
        ...,
        description="Number of items filtered out",
    )
    filtering_summary: str = Field(
        ...,
        description="Summary of filtering decisions and selection rationale",
    )


class BaseApproachFilteringComponent(
    BaseDataSearchComponent[
        BaseApproachFilteringInputSchema,
        BaseApproachFilteringOutput,
    ],
    ABC,
):
    """
    Abstract base class for per-approach filtering and ranking.

    Repository-specific implementations inherit from this to filter and rank
    data items (collections, bundles, etc.) within a single query approach.

    Two-part process:
    1. Filter out obviously bad matches (spatial/temporal/resolution mismatches)
    2. Select and rank the best items from remaining candidates
    """

    input_schema = BaseApproachFilteringInputSchema
    output_schema = BaseApproachFilteringOutput

    # Subclasses should override template_name
    template_name = "approach_filtering"
    default_temperature = 0.0  # Consistent filtering
    retry_enabled = False  # No retry for ranking components


class BaseFinalRankingInputSchema(InputSchema):
    """Base input schema for final cross-approach ranking."""

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

    data_items: List[Dict[str, Any]] = Field(
        ...,
        description="Pre-filtered data items from all approaches",
    )
    max_items: int = Field(
        default=25,
        description="Maximum items to return (ranked 1-N)",
    )


class FinalRankedItem(BaseModel):
    """Base schema for final ranked data item."""

    item_index: int = Field(
        ...,
        description="Index in the input data items list (0-based)",
    )
    final_rank: int = Field(
        ...,
        ge=1,
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
        description="Why this item received this rank",
    )


class BaseFinalRankingOutput(BaseModel):
    """Base output schema for final ranking."""

    ranked_items: List[FinalRankedItem] = Field(
        ...,
        description="Items ranked by relevance",
    )
    total_items_ranked: int = Field(
        ...,
        description="Total number of items in the ranking",
    )
    ranking_summary: str = Field(
        ...,
        description="Summary of ranking decisions and key differentiators",
    )


class BaseFinalRankingComponent(
    BaseDataSearchComponent[
        BaseFinalRankingInputSchema,
        BaseFinalRankingOutput,
    ],
    ABC,
):
    """
    Abstract base class for final cross-approach ranking.

    Repository-specific implementations inherit from this to perform final
    ranking of pre-filtered data items across all query approaches.

    No filtering - only comparative ranking of pre-filtered items.
    """

    input_schema = BaseFinalRankingInputSchema
    output_schema = BaseFinalRankingOutput

    # Subclasses should override template_name
    template_name = "final_ranking"
    default_temperature = 0.0  # Consistent ranking
    retry_enabled = False  # No retry for ranking components
