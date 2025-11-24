"""
Shared logic for ranking and filtering components.

Contains generic implementations for:
- Approach Filtering: Filter and rank data items within a single approach
- Final Ranking: Comparative ranking across all approaches

Repository-specific subclasses only need to define schemas and template names.
"""

from typing import Any, Dict, Generic

from loguru import logger

from ._base import TInput, TOutput
from ._base_ranking import BaseApproachFilteringComponent, BaseFinalRankingComponent


class SharedApproachFilteringComponent(
    BaseApproachFilteringComponent,
    Generic[TInput, TOutput],
):
    """
    Shared implementation for per-approach filtering and ranking.

    Generic logic for filtering out bad matches and ranking remaining items
    within a single query approach using LLM evaluation.

    Repository-specific subclasses should:
    - Set input_schema and output_schema
    - Set template_name (e.g., "approach_filtering")
    - Optionally override _prepare_item_summary() for custom formatting
    - Optionally override _extract_approach_context() for custom fields
    """

    async def _arun(
        self,
        params: TInput,
    ) -> TOutput:
        """Execute approach-level filtering and ranking."""
        # Get data items from params (using property/alias pattern)
        data_items = params.data_items
        max_items = params.max_items

        if self.debug:
            logger.info(
                f"Filtering {len(data_items)} items for approach (max_items: {max_items})",
            )

        # Prepare item summaries (limit metadata for token efficiency)
        items_summary = []
        for i, item in enumerate(data_items):
            summary = self._prepare_item_summary(i, item)
            items_summary.append(summary)

        # Extract approach context fields
        approach_context = self._extract_approach_context(params)

        # Get min/max values from output schema
        # metadata[0] = MinLen, metadata[1] = MaxLen
        min_items = self.output_schema.model_fields["selected_item_indexes"].metadata[0].min_length
        max_items_schema = self.output_schema.model_fields["selected_item_indexes"].metadata[1].max_length

        # Format user prompt
        user_prompt = self._format_user_prompt_from_template(
            original_query=params.original_query,
            topic_title=params.topic_title,
            topic_context=params.topic_context,
            decomposition_title=params.decomposition_title,
            decomposition_justification=params.decomposition_justification,
            num_items=len(data_items),
            items_list="\n\n".join(items_summary),
            min_items=min_items,
            max_items=max_items_schema,
            **approach_context,
        )

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "approach_filtering",
            params.decomposition_title,
            len(data_items),
        )

        self._add_user_message(user_prompt)
        messages = [self._default_system_message()] + self.memory
        result = await self.get_response_async(messages=messages)

        if self.debug:
            selected_count = len(result.selected_item_indexes)
            logger.info(
                f"Approach filtering complete: {selected_count} selected from {len(data_items)} reviewed",
            )

        return result

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare concise summary of a single data item for LLM evaluation.

        Default implementation provides basic summary.
        Subclasses can override for repository-specific formatting.

        Args:
            index: Index of item in list (0-based)
            item: Data item dictionary

        Returns:
            Formatted summary string
        """
        # Generic implementation - works for CMR-like collections
        summary_parts = [
            f"Index {index}. {item.get('dataset_id', item.get('id', 'Unknown ID'))}",
        ]

        if item.get("title"):
            summary_parts.append(f"   Title: {item['title']}")

        if item.get("abstract"):
            abstract = item["abstract"][:750] + "..." if len(item["abstract"]) > 750 else item["abstract"]
            summary_parts.append(f"   Abstract: {abstract}")

        # Key metadata for filtering
        if item.get("time_start") or item.get("time_end"):
            temporal = f"{item.get('time_start', 'N/A')} to {item.get('time_end', 'N/A')}"
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

    def _extract_approach_context(self, params: TInput) -> Dict[str, str]:
        """
        Extract approach-specific context fields for prompt formatting.

        Uses object composition pattern: params.approach contains all known parameters.

        Args:
            params: Input parameters with approach object

        Returns:
            Dictionary of approach context fields for prompt template
        """
        context = {}

        # Direct object access via params.approach
        if hasattr(params, "approach"):
            approach = params.approach
            context.update(
                {
                    "approach_instrument": approach.instrument or "Not specified",
                    "approach_platform": approach.platform or "Not specified",
                    "approach_processing_level": approach.processing_level or "Not specified",
                    "approach_temporal_range": approach.temporal or "Not specified",
                    "approach_spatial_bounds": approach.bounding_box or "Not specified",
                    "approach_temporal_resolution": approach.temporal_resolution or "Not specified",
                    "approach_spatial_resolution": approach.spatial_resolution or "Not specified",
                },
            )

        # Handle search string separately (may come from params)
        if hasattr(params, "approach_search_string"):
            search_string = params.approach_search_string
            context["approach_search_string"] = search_string if search_string else "None"

        return context


class SharedFinalRankingComponent(
    BaseFinalRankingComponent,
    Generic[TInput, TOutput],
):
    """
    Shared implementation for final cross-approach ranking.

    Generic logic for comparative ranking of pre-filtered items across
    all query approaches using LLM evaluation.

    Repository-specific subclasses should:
    - Set input_schema and output_schema
    - Set template_name (e.g., "final_ranking")
    - Optionally override _prepare_item_summary() for custom formatting
    """

    async def _arun(
        self,
        params: TInput,
    ) -> TOutput:
        """Execute final cross-approach ranking."""
        # Get data items from params (using property/alias pattern)
        data_items = params.data_items
        max_items = params.max_items

        if self.debug:
            logger.info(f"Final ranking of {len(data_items)} items (max: {max_items})")

        # Prepare item summaries
        items_summary = []
        for i, item in enumerate(data_items):
            summary = self._prepare_item_summary(i, item)
            items_summary.append(summary)

        # Format user prompt
        user_prompt = self._format_user_prompt_from_template(
            original_query=params.original_query,
            topic_title=params.topic_title,
            topic_context=params.topic_context,
            decomposition_title=params.decomposition_title,
            decomposition_justification=params.decomposition_justification,
            num_items=len(data_items),
            items_list="\n\n".join(items_summary),
            max_items=max_items,
        )

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "final_ranking",
            params.decomposition_title,
            len(data_items),
        )

        self._add_user_message(user_prompt)
        messages = [self._default_system_message()] + self.memory
        result = await self.get_response_async(messages=messages)

        if self.debug:
            ranked_count = len(result.ranked_item_indexes)
            logger.info(f"Final ranking complete: {ranked_count} items ranked")

        return result

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare concise summary of a single data item for LLM evaluation.

        Default implementation provides basic summary.
        Subclasses can override for repository-specific formatting.

        Args:
            index: Index of item in list (0-based)
            item: Data item dictionary

        Returns:
            Formatted summary string
        """
        # Generic implementation - works for CMR-like collections
        summary = f"Index {index}. {item.get('dataset_id', item.get('id', 'Unknown'))}"

        if item.get("title"):
            summary += f"\n   Title: {item['title']}"

        if item.get("abstract"):
            abstract = item["abstract"][:300] + "..." if len(item["abstract"]) > 300 else item["abstract"]
            summary += f"\n   Abstract: {abstract}"

        # Include key distinguishing features
        if item.get("instrument"):
            summary += f"\n   Instrument: {item.get('instrument')}"

        if item.get("platform"):
            summary += f"\n   Platform: {item.get('platform')}"

        if item.get("processing_level_id"):
            summary += f"\n   Level: {item.get('processing_level_id')}"

        return summary
