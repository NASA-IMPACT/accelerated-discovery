"""PDS4-specific component implementations."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search.components._base import BaseDataSearchComponent
from akd.agents.data_search.components._shared_ranking import (
    SharedApproachFilteringComponent,
    SharedFinalRankingComponent,
)
from akd.agents.data_search.utils.prompt_loader import load_and_format_prompt

from .schemas import (
    PDS4ApproachCollectionFilteringInputSchema,
    PDS4ApproachCollectionFilteringOutput,
    PDS4ContextSearchURNFilteringInput,
    PDS4ContextSearchURNFilteringOutput,
    PDS4FinalCollectionRankingInputSchema,
    PDS4FinalCollectionRankingOutput,
    PDS4ParameterExtractionInputSchema,
    PDS4ParameterExtractionOutput,
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
            PDS4ParameterExtractionOutput with complete query approaches
        """
        if self.debug:
            logger.debug(
                f"Extracting PDS4 query approaches for decomposition: '{decomposition.title}'",
            )

        # Extract min/max approaches from output schema metadata
        # metadata[0] = MinLen, metadata[1] = MaxLen
        min_approaches = (
            self.output_schema.model_fields["query_approaches"].metadata[0].min_length
        )
        max_approaches = (
            self.output_schema.model_fields["query_approaches"].metadata[1].max_length
        )

        # Format user prompt
        user_prompt = load_and_format_prompt(
            template_name=f"{self.template_name}_user",
            prompts_dir=self.prompts_dir,
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            min_approaches=min_approaches,
            max_approaches=max_approaches,
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
                f"Extracted {len(result.query_approaches)} query approaches: {result.reasoning}",
            )
            if not result.query_approaches:
                logger.warning("⚠️ EMPTY query_approaches returned by LLM - workflow will fail!")
                logger.debug(f"Reasoning: {result.reasoning}")
            else:
                for idx, approach in enumerate(result.query_approaches):
                    logger.debug(f"  Approach {idx}: {approach.approach_description}")

        return result


class PDS4ContextSearchURNFilteringComponent(
    BaseDataSearchComponent[
        PDS4ContextSearchURNFilteringInput,
        PDS4ContextSearchURNFilteringOutput,
    ],
):
    """
    LLM-based URN filtering component for PDS4 context search results.

    This component replaces keyword-based scoring with LLM judgment to select
    relevant URNs from investigation, target, and instrument context searches.

    The LLM evaluates URNs based on:
    - Keyword matching against context search keywords (primary criterion)
    - Relevance to user query, topic, and decomposition
    - Compatibility across investigation/target/instrument combinations

    Unlike the old approach which selected exactly 3 URNs per type, the LLM
    has complete freedom to select 0 to unlimited URNs based on relevance.
    """

    input_schema = PDS4ContextSearchURNFilteringInput
    output_schema = PDS4ContextSearchURNFilteringOutput

    # Base class configuration
    template_name = "context_search_urn_filtering"
    default_temperature = 0.1  # Low temperature for consistent filtering decisions

    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
        prompts_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the PDS4 context search URN filtering component."""
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
        topic: str,
        decomposition: str,
        strategy_description: str,
        investigation_keywords: List[str],
        target_keywords: List[str],
        instrument_keywords: List[str],
        investigation_results: List[Dict[str, Any]],
        target_results: List[Dict[str, Any]],
        instrument_results: List[Dict[str, Any]],
    ) -> PDS4ContextSearchURNFilteringOutput:
        """
        Filter URNs from context search results using LLM judgment.

        Args:
            original_query: Original research query for context
            topic: Topic being processed
            decomposition: Scientific decomposition text
            strategy_description: Tool sequence and parameters
            investigation_keywords: Keywords used in investigation search
            target_keywords: Keywords used in target search
            instrument_keywords: Keywords used in instrument search
            investigation_results: Investigation search results
            target_results: Target search results
            instrument_results: Instrument search results

        Returns:
            PDS4ContextSearchURNFilteringOutput with selected URNs and reasoning
        """
        if self.debug:
            logger.debug(
                f"Filtering URNs for strategy: '{strategy_description[:100]}...'",
            )
            logger.debug(
                f"Context results: {len(investigation_results)} investigations, "
                f"{len(target_results)} targets, {len(instrument_results)} instruments"
            )

        # Format keywords for prompt (lists → comma-separated strings)
        formatted_investigation_keywords = ", ".join(investigation_keywords) if investigation_keywords else "None"
        formatted_target_keywords = ", ".join(target_keywords) if target_keywords else "None"
        formatted_instrument_keywords = ", ".join(instrument_keywords) if instrument_keywords else "None"

        # Format context search results for LLM
        formatted_investigation_results = self._format_context_results(
            investigation_results, "investigation"
        )
        formatted_target_results = self._format_context_results(
            target_results, "target"
        )
        formatted_instrument_results = self._format_context_results(
            instrument_results, "instrument"
        )

        # Format user prompt
        user_prompt = load_and_format_prompt(
            template_name=f"{self.template_name}_user",
            prompts_dir=self.prompts_dir,
            original_query=original_query,
            topic=topic,
            decomposition=decomposition,
            strategy_description=strategy_description,
            investigation_keywords=formatted_investigation_keywords,
            target_keywords=formatted_target_keywords,
            instrument_keywords=formatted_instrument_keywords,
            investigation_results=formatted_investigation_results,
            target_results=formatted_target_results,
            instrument_results=formatted_instrument_results,
        )

        # Save prompt for debugging
        self._save_prompt_to_file(
            user_prompt,
            "context_search_urn_filtering",
            f"{decomposition[:50]}",
        )

        # Get LLM response
        self._add_user_message(user_prompt)
        result = await self.get_response_async()

        if self.debug:
            logger.debug(
                f"Selected URNs: {len(result.selected_investigation_urns)} investigations, "
                f"{len(result.selected_target_urns)} targets, "
                f"{len(result.selected_instrument_urns)} instruments"
            )
            logger.debug(f"Reasoning: {result.reasoning[:200]}...")

        return result

    def _format_context_results(
        self,
        results: List[Dict[str, Any]],
        context_type: str,
    ) -> str:
        """
        Format context search results for LLM consumption.

        Args:
            results: List of context search results
            context_type: Type of context (investigation/target/instrument)

        Returns:
            Formatted string representation of results
        """
        if not results:
            return f"No {context_type} results found."

        formatted_lines = []
        for idx, result in enumerate(results, 1):
            # Extract key fields
            urn = result.get("urn", result.get("id", "Unknown URN"))
            title = result.get("title", "No title")
            description = result.get("description", "No description")

            # Truncate description if too long
            if len(description) > 200:
                description = description[:200] + "..."

            formatted_lines.append(
                f"{idx}. URN: {urn}\n"
                f"   Title: {title}\n"
                f"   Description: {description}"
            )

        return "\n\n".join(formatted_lines)


class PDS4ApproachCollectionFilteringComponent(
    SharedApproachFilteringComponent[
        PDS4ApproachCollectionFilteringInputSchema,
        PDS4ApproachCollectionFilteringOutput,
    ],
):
    """
    PDS4-specific implementation of per-approach collection filtering.

    Thin wrapper that sets PDS4 schemas and template name.
    All logic is in SharedApproachFilteringComponent.
    """

    input_schema = PDS4ApproachCollectionFilteringInputSchema
    output_schema = PDS4ApproachCollectionFilteringOutput

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
        """Initialize the PDS4 approach collection filtering component."""
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

    def _extract_approach_context(self, params: PDS4ApproachCollectionFilteringInputSchema) -> Dict[str, Any]:
        """
        Extract approach-specific context for prompt formatting.

        PDS4 approaches include keywords and URN references from context searches.
        """
        # Format URN lists as bullet points for the LLM
        def format_urn_list(urns: List[str], label: str) -> str:
            if not urns:
                return "Not specified"
            if len(urns) == 1:
                return urns[0]
            # Multiple URNs - format as indented bullet list
            return "\n  " + "\n  ".join(f"- {urn}" for urn in urns)

        return {
            "strategy_description": params.strategy_description,
            "investigation": ", ".join(params.investigation_keywords) if params.investigation_keywords else "Not specified",
            "target": ", ".join(params.target_keywords) if params.target_keywords else "Not specified",
            "instruments": ", ".join(params.instrument_keywords) if params.instrument_keywords else "Not specified",
            "temporal": params.temporal_context or "Not specified",
            "investigation_urns": format_urn_list(params.investigation_urns, "Investigation"),
            "target_urns": format_urn_list(params.target_urns, "Target"),
            "instrument_urns": format_urn_list(params.instrument_urns, "Instrument"),
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
