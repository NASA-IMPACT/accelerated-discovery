"""
Embedded research synthesis component for literature search agents.
"""

from typing import List, Optional

from loguru import logger

from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.agents.schemas import DeepResearchInputSchema, DeepResearchOutputSchema
from akd.configs.prompts import DEEP_RESEARCH_AGENT_PROMPT
from akd.structures import SearchResultItem


class ResearchSynthesisComponentConfig(BaseAgentConfig):
    """Configuration for the embedded research synthesis component."""

    system_prompt: str = DEEP_RESEARCH_AGENT_PROMPT
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4000


class ResearchSynthesisComponent:
    """
    Embedded research synthesis component that creates comprehensive research reports.

    This component is embedded within literature search agents to provide
    research synthesis functionality without requiring separate agent instantiation.
    """

    def __init__(
        self,
        config: Optional[ResearchSynthesisComponentConfig] = None,
        debug: bool = False,
    ):
        self.config = config or ResearchSynthesisComponentConfig()
        self.debug = debug

        # Create internal instructor agent for research synthesis
        self._agent = InstructorBaseAgent[
            DeepResearchInputSchema, DeepResearchOutputSchema
        ](config=self.config, debug=debug)
        self._agent.input_schema = DeepResearchInputSchema
        self._agent.output_schema = DeepResearchOutputSchema

    async def synthesize(
        self,
        results: List[SearchResultItem],
        research_instructions: str,
        original_query: str,
        quality_scores: List[float],
        research_trace: List[str],
        iterations_performed: int,
    ) -> DeepResearchOutputSchema:
        """
        Synthesize research results into a comprehensive report.

        Args:
            results: List of search results to synthesize
            research_instructions: Original research instructions
            original_query: The original user query
            quality_scores: Quality scores from iterations
            research_trace: Trace of research process
            iterations_performed: Number of iterations performed

        Returns:
            Comprehensive research output with report and metadata
        """
        if self.debug:
            logger.debug(f"Synthesizing {len(results)} results into research report")

        # Prepare research input for synthesis
        research_input = DeepResearchInputSchema(
            research_instructions=research_instructions,
            original_query=original_query,
            max_iterations=iterations_performed,
            quality_threshold=sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.5,
        )

        # For now, create a structured output based on results
        # In a full implementation, this would use the research agent for synthesis
        return await self._create_structured_output(
            results,
            research_input,
            quality_scores,
            research_trace,
            iterations_performed,
        )

    async def _create_structured_output(
        self,
        results: List[SearchResultItem],
        research_input: DeepResearchInputSchema,
        quality_scores: List[float],
        research_trace: List[str],
        iterations_performed: int,
    ) -> DeepResearchOutputSchema:
        """Create structured research output from results."""

        # Group results by relevance score if available
        if (
            results
            and hasattr(results[0], "relevancy_score")
            and results[0].relevancy_score is not None
        ):
            # Sort by relevancy score (highest first)
            sorted_results = sorted(
                results,
                key=lambda r: getattr(r, "relevancy_score", 0.0),
                reverse=True,
            )
            high_quality_results = sorted_results[:20]
        else:
            # Fall back to original ordering
            high_quality_results = results[:20]

        # Extract key findings
        key_findings = []
        for result in high_quality_results[:10]:
            if result.content:
                # Extract first significant sentence as a finding
                sentences = result.content.split(". ")
                if sentences:
                    key_findings.append(sentences[0] + ".")

        # Create research report structure
        report_sections = [
            "# Research Report",
            f"\nThis research on '{research_input.original_query}' analyzed {len(results)} sources",
            f"across {iterations_performed} iterations.",
            "\n## Key Findings",
        ]

        for i, finding in enumerate(key_findings[:5], 1):
            report_sections.append(f"{i}. {finding}")

        report_sections.extend(
            [
                "\n## Sources Consulted",
            ]
        )

        # Create citations list
        citations = []
        sources = []
        for result in high_quality_results:
            sources.append(str(result.url))
            citation = {
                "title": result.title or "Untitled",
                "url": str(result.url),
                "excerpt": (result.content[:200] if result.content else "") + "...",
            }
            citations.append(citation)
            report_sections.append(f"- [{result.title}]({result.url})")

        research_report = "\n".join(report_sections)

        # Create final output
        return DeepResearchOutputSchema(
            research_report=research_report,
            key_findings=key_findings,
            sources_consulted=sources,
            evidence_quality_score=sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.5,
            gaps_identified=[
                "Potential newer research not yet indexed",
                "Non-English sources not included",
            ],
            citations=citations,
            iterations_performed=iterations_performed,
            research_trace=research_trace,
        )
