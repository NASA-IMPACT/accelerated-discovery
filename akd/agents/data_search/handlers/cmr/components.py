"""CMR-specific component implementations."""

from pathlib import Path
from typing import Optional

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search.components._shared_parameters import (
    SharedKnownParametersComponent,
    SharedSearchableParametersComponent,
)
from akd.agents.data_search.components._shared_ranking import (
    SharedApproachFilteringComponent,
    SharedFinalRankingComponent,
)

from .schemas import (
    CMRApproachCollectionFilteringInputSchema,
    CMRApproachCollectionFilteringOutput,
    CMRFinalCollectionRankingInputSchema,
    CMRFinalCollectionRankingOutput,
    CMRKnownParametersInputSchema,
    CMRKnownParametersOutput,
    CMRQueryApproach,
    CMRSearchableParametersInputSchema,
    CMRSearchableParametersOutput,
    CMRSearchableQuery,
)


class CMRKnownParametersComponent(
    SharedKnownParametersComponent[
        CMRKnownParametersInputSchema,
        CMRKnownParametersOutput,
        CMRQueryApproach,
    ],
):
    """
    CMR-specific implementation of known parameters extraction.

    Thin wrapper that sets CMR schemas and template name.
    All logic is in SharedKnownParametersComponent.
    """

    input_schema = CMRKnownParametersInputSchema
    output_schema = CMRKnownParametersOutput

    # Base class configuration
    template_name = "known_parameters"
    default_temperature = 0.0  # Very low temperature for precise parameter extraction

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize the CMR known parameters component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
        )


class CMRSearchableParametersComponent(
    SharedSearchableParametersComponent[
        CMRSearchableParametersInputSchema,
        CMRSearchableParametersOutput,
        CMRQueryApproach,
        CMRSearchableQuery,
    ],
):
    """
    CMR-specific implementation of searchable parameters generation.

    Thin wrapper that sets CMR schemas and template name.
    All logic is in SharedSearchableParametersComponent.
    """

    input_schema = CMRSearchableParametersInputSchema
    output_schema = CMRSearchableParametersOutput

    # Base class configuration
    template_name = "searchable_parameters"
    default_temperature = 0.1  # Low temperature for consistent keyword generation

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize the CMR searchable parameters component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
        )

    def _create_searchable_query(
        self,
        approach: CMRQueryApproach,
        approach_index: int,
        keyword_string: str,
    ) -> CMRSearchableQuery:
        """
        Create a CMR-specific searchable query object.

        Args:
            approach: Source query approach with known parameters
            approach_index: Index of source approach
            keyword_string: Generated keyword string

        Returns:
            CMR searchable query object
        """
        # Parse keywords for metadata (though we primarily use the combined string)
        keywords_list = keyword_string.split() if keyword_string.strip() else []
        primary_keywords = keywords_list[:4] if keywords_list else []
        alternative_keywords = keywords_list[4:7] if len(keywords_list) > 4 else []

        return CMRSearchableQuery(
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
        )

    def _generate_strategy_explanation(
        self,
        decomposition,
        searchable_queries,
    ) -> str:
        """
        Generate explanation of search variation strategy.

        CMR-specific implementation with detailed statistics.
        """
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


class CMRApproachCollectionFilteringComponent(
    SharedApproachFilteringComponent[
        CMRApproachCollectionFilteringInputSchema,
        CMRApproachCollectionFilteringOutput,
    ],
):
    """
    CMR-specific implementation of per-approach collection filtering.

    Thin wrapper that sets CMR schemas and template name.
    All logic is in SharedApproachFilteringComponent.
    """

    input_schema = CMRApproachCollectionFilteringInputSchema
    output_schema = CMRApproachCollectionFilteringOutput

    # Base class configuration
    template_name = "approach_filtering"
    default_temperature = 0.0  # Consistent filtering
    retry_enabled = False  # No retry for ranking components

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize the CMR approach collection filtering component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
        )


class CMRFinalCollectionRankingComponent(
    SharedFinalRankingComponent[
        CMRFinalCollectionRankingInputSchema,
        CMRFinalCollectionRankingOutput,
    ],
):
    """
    CMR-specific implementation of final cross-approach ranking.

    Thin wrapper that sets CMR schemas and template name.
    All logic is in SharedFinalRankingComponent.
    """

    input_schema = CMRFinalCollectionRankingInputSchema
    output_schema = CMRFinalCollectionRankingOutput

    # Base class configuration
    template_name = "final_ranking"
    default_temperature = 0.0  # Consistent ranking
    retry_enabled = False  # No retry for ranking components

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize the CMR final collection ranking component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
        )
