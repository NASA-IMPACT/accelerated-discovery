"""CMR-specific component implementations."""

from pathlib import Path
from typing import Any, Dict, Optional

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
        run_id: Optional[str] = None,
    ):
        """Initialize the CMR known parameters component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
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
        run_id: Optional[str] = None,
    ):
        """Initialize the CMR searchable parameters component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
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
            keyword_string: Generated search string

        Returns:
            CMR searchable query object
        """
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
            search_string=keyword_string.strip() if keyword_string.strip() else None,
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
        queries_with_search_strings = sum(
            1 for q in searchable_queries if q.search_string
        )
        queries_without_search_strings = total_queries - queries_with_search_strings

        unique_terms = set()
        for query in searchable_queries:
            if query.search_string:
                unique_terms.update(query.search_string.split())

        strategy_parts = [
            f"Generated {total_queries} search variations targeting '{decomposition.title}'.",
        ]

        if queries_without_search_strings > 0:
            strategy_parts.append(
                f"{queries_without_search_strings} searches use only known parameters.",
            )

        if queries_with_search_strings > 0:
            strategy_parts.append(
                f"{queries_with_search_strings} searches add focused search strings from {len(unique_terms)} unique terms.",
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
        run_id: Optional[str] = None,
    ):
        """Initialize the CMR approach collection filtering component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare CMR collection summary using correct CMR field names.

        CMR uses: entry_title, summary, concept_id (not title, abstract, dataset_id)
        """
        summary_parts = [
            f"Index {index}. {item.get('concept_id', item.get('dataset_id', 'Unknown ID'))}",
        ]

        if item.get("entry_title"):
            summary_parts.append(f"   Title: {item['entry_title']}")

        if item.get("summary"):
            abstract = (
                item["summary"][:500] + "..."
                if len(item["summary"]) > 500
                else item["summary"]
            )
            summary_parts.append(f"   Abstract: {abstract}")

        # Key metadata for filtering
        if item.get("time_start") or item.get("time_end"):
            temporal = (
                f"{item.get('time_start', 'N/A')} to {item.get('time_end', 'N/A')}"
            )
            summary_parts.append(f"   Temporal: {temporal}")

        if item.get("boxes"):
            summary_parts.append(f"   Spatial: {item.get('boxes')}")

        if item.get("processing_level_id"):
            summary_parts.append(f"   Level: {item.get('processing_level_id')}")

        # Resolution metadata
        if item.get("horizontal_data_resolution"):
            summary_parts.append(
                f"   Spatial Resolution: {item.get('horizontal_data_resolution')}",
            )

        if item.get("temporal_resolution"):
            summary_parts.append(
                f"   Temporal Resolution: {item.get('temporal_resolution')}",
            )

        return "\n".join(summary_parts)


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
        run_id: Optional[str] = None,
    ):
        """Initialize the CMR final collection ranking component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare CMR collection summary using correct CMR field names.

        CMR uses: entry_title, summary, concept_id (not title, abstract, dataset_id)
        """
        summary = f"Index {index}. {item.get('concept_id', item.get('dataset_id', 'Unknown'))}"

        if item.get("entry_title"):
            summary += f"\n   Title: {item['entry_title']}"

        if item.get("summary"):
            abstract = (
                item["summary"][:500] + "..."
                if len(item["summary"]) > 500
                else item["summary"]
            )
            summary += f"\n   Abstract: {abstract}"

        # Include key distinguishing features
        if item.get("instrument"):
            summary += f"\n   Instrument: {item.get('instrument')}"

        if item.get("platform"):
            summary += f"\n   Platform: {item.get('platform')}"

        if item.get("processing_level_id"):
            summary += f"\n   Level: {item.get('processing_level_id')}"

        # Temporal and spatial coverage
        if item.get("time_start") or item.get("time_end"):
            temporal = (
                f"{item.get('time_start', 'N/A')} to {item.get('time_end', 'N/A')}"
            )
            summary += f"\n   Temporal: {temporal}"

        if item.get("boxes"):
            summary += f"\n   Spatial: {item.get('boxes')}"

        # Resolution metadata
        if item.get("horizontal_data_resolution"):
            summary += (
                f"\n   Spatial Resolution: {item.get('horizontal_data_resolution')}"
            )

        if item.get("temporal_resolution"):
            summary += f"\n   Temporal Resolution: {item.get('temporal_resolution')}"

        return summary
