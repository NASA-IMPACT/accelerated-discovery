"""PDS4-specific component implementations."""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search.components._base import BaseDataSearchComponent
from akd.agents.data_search.components._shared_ranking import (
    SharedApproachFilteringComponent,
    SharedFinalRankingComponent,
)
from akd.agents.data_search.utils.prompt_loader import load_and_format_prompt

from .schemas import (
    PDS4FinalCollectionRankingInputSchema,
    PDS4FinalCollectionRankingOutput,
    PDS4ParameterExtractionInputSchema,
    PDS4ParameterExtractionOutput,
    PDS4StrategyCollectionFilteringInputSchema,
    PDS4StrategyCollectionFilteringOutput,
)


class PDS4ParameterExtractionComponent(
    BaseDataSearchComponent[
        PDS4ParameterExtractionInputSchema,
        PDS4ParameterExtractionOutput,
    ],
):
    """
    PDS4-specific implementation of unified parameter extraction.

    Unlike CMR's two-stage approach (known → searchable parameters),
    PDS4 uses a unified approach that generates complete tool execution
    strategies in a single step.

    Each strategy includes:
    - Context search parameters (mission keywords, target type)
    - URN references extracted from context searches
    - Data filtering parameters
    """

    input_schema = PDS4ParameterExtractionInputSchema
    output_schema = PDS4ParameterExtractionOutput

    # Base class configuration
    template_name = "parameter_extraction"
    default_temperature = 0.1  # Low temperature for consistent parameter extraction

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the PDS4 parameter extraction component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )

    async def process(
        self,
        original_query: str,
        topic,  # Topic from topic_splitting
        decomposition,  # ScientificDecomposition
    ) -> PDS4ParameterExtractionOutput:
        """
        Extract unified PDS4 tool strategies from research context.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification

        Returns:
            PDS4ParameterExtractionOutput with complete tool strategies
        """
        if self.debug:
            logger.debug(
                f"Extracting PDS4 tool strategies for decomposition: '{decomposition.title}'",
            )

        # Format user prompt
        user_prompt = load_and_format_prompt(
            template_name=f"{self.template_name}_user",
            prompts_dir=self.prompts_dir,
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.context,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.justification,
        )

        # Save prompt for debugging
        self._save_prompt_to_file(
            user_prompt,
            "parameter_extraction",
            decomposition.title,
        )

        # Get LLM response
        self._add_user_message(user_prompt)
        result = await self.get_response_async()

        if self.debug:
            logger.debug(
                f"Extracted {len(result.tool_strategies)} tool strategies: {result.reasoning}",
            )

        return result


class PDS4StrategyCollectionFilteringComponent(
    SharedApproachFilteringComponent[
        PDS4StrategyCollectionFilteringInputSchema,
        PDS4StrategyCollectionFilteringOutput,
    ],
):
    """
    PDS4-specific implementation of per-strategy collection filtering.

    Thin wrapper that sets PDS4 schemas and template name.
    All logic is in SharedApproachFilteringComponent.
    """

    input_schema = PDS4StrategyCollectionFilteringInputSchema
    output_schema = PDS4StrategyCollectionFilteringOutput

    # Base class configuration
    template_name = "strategy_filtering"
    default_temperature = 0.0  # Consistent filtering
    retry_enabled = False  # No retry for ranking components

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the PDS4 strategy collection filtering component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare PDS4 collection summary using correct PDS4 field names.

        PDS4 uses: title, description, lidvid (not entry_title, summary, concept_id)
        """
        summary_parts = [
            f"Index {index}. {item.get('lidvid', item.get('id', 'Unknown ID'))}",
        ]

        if item.get("title"):
            summary_parts.append(f"   Title: {item['title']}")

        if item.get("description"):
            abstract = (
                item["description"][:500] + "..."
                if len(item["description"]) > 500
                else item["description"]
            )
            summary_parts.append(f"   Description: {abstract}")

        # Key metadata for filtering
        if item.get("investigation_ref"):
            summary_parts.append(f"   Investigation: {item.get('investigation_ref')}")

        if item.get("target_ref"):
            summary_parts.append(f"   Target: {item.get('target_ref')}")

        if item.get("instrument_ref"):
            summary_parts.append(f"   Instrument: {item.get('instrument_ref')}")

        if item.get("instrument_host_ref"):
            summary_parts.append(
                f"   Instrument Host: {item.get('instrument_host_ref')}",
            )

        # Temporal coverage if available
        if item.get("start_date_time") or item.get("stop_date_time"):
            temporal = f"{item.get('start_date_time', 'N/A')} to {item.get('stop_date_time', 'N/A')}"
            summary_parts.append(f"   Temporal: {temporal}")

        # Processing level or data type
        if item.get("processing_level"):
            summary_parts.append(f"   Processing Level: {item.get('processing_level')}")

        return "\n".join(summary_parts)

    def _extract_approach_context(self, params: PDS4StrategyCollectionFilteringInputSchema) -> Dict[str, Any]:
        """
        Extract strategy-specific context for prompt formatting.

        PDS4 strategies include context type, keywords, and URN references.
        """
        return {
            "strategy_description": params.strategy_description,
            "target_context": params.target_context or "Not specified",
            "target_type": params.target_type or "Not specified",
            "mission_keywords": ", ".join(params.mission_keywords) if params.mission_keywords else "Not specified",
            "investigation_urn": params.investigation_urn or "Not specified",
            "target_urn": params.target_urn or "Not specified",
        }


class PDS4FinalCollectionRankingComponent(
    SharedFinalRankingComponent[
        PDS4FinalCollectionRankingInputSchema,
        PDS4FinalCollectionRankingOutput,
    ],
):
    """
    PDS4-specific implementation of final cross-strategy ranking.

    Thin wrapper that sets PDS4 schemas and template name.
    All logic is in SharedFinalRankingComponent.
    """

    input_schema = PDS4FinalCollectionRankingInputSchema
    output_schema = PDS4FinalCollectionRankingOutput

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
        """Initialize the PDS4 final collection ranking component."""
        super().__init__(
            config=config,
            debug=debug,
            template_name=self.template_name,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )

    def _prepare_item_summary(self, index: int, item: Dict[str, Any]) -> str:
        """
        Prepare PDS4 collection summary using correct PDS4 field names.

        PDS4 uses: title, description, lidvid (not entry_title, summary, concept_id)
        """
        summary = f"Index {index}. {item.get('lidvid', item.get('id', 'Unknown'))}"

        if item.get("title"):
            summary += f"\n   Title: {item['title']}"

        if item.get("description"):
            abstract = (
                item["description"][:500] + "..."
                if len(item["description"]) > 500
                else item["description"]
            )
            summary += f"\n   Description: {abstract}"

        # Include key distinguishing features
        if item.get("investigation_ref"):
            summary += f"\n   Investigation: {item.get('investigation_ref')}"

        if item.get("target_ref"):
            summary += f"\n   Target: {item.get('target_ref')}"

        if item.get("instrument_ref"):
            summary += f"\n   Instrument: {item.get('instrument_ref')}"

        if item.get("instrument_host_ref"):
            summary += f"\n   Instrument Host: {item.get('instrument_host_ref')}"

        # Temporal coverage
        if item.get("start_date_time") or item.get("stop_date_time"):
            temporal = f"{item.get('start_date_time', 'N/A')} to {item.get('stop_date_time', 'N/A')}"
            summary += f"\n   Temporal: {temporal}"

        # Processing level or product class
        if item.get("processing_level"):
            summary += f"\n   Processing Level: {item.get('processing_level')}"

        if item.get("product_class"):
            summary += f"\n   Product Class: {item.get('product_class')}"

        return summary
