"""
Embedded research synthesis component for deep literature search agent.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.configs.prompts import DEEP_RESEARCH_AGENT_PROMPT
from akd.structures import SearchResultItem


class ResearchSynthesisInputSchema(InputSchema):
    """Input schema for the ResearchSynthesisAgent."""

    query: str = Field(..., description="Research query to synthesize")
    search_results: List[SearchResultItem] = Field(
        ..., description="Search results to synthesize into a report"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional research context including instructions, quality scores, and trace",
    )


class ResearchSynthesisOutputSchema(OutputSchema):
    """Output schema for the ResearchSynthesisAgent."""

    research_report: str = Field(
        ..., description="Comprehensive research report in markdown format"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key research findings extracted from the sources",
    )
    sources_consulted: List[str] = Field(
        default_factory=list,
        description="URLs of sources that were analyzed",
    )
    evidence_quality_score: float = Field(
        default=0.5,
        description="Overall quality score of the evidence (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured citation information for key sources",
    )


class ResearchSynthesisAgentConfig(BaseAgentConfig):
    """Configuration for the ResearchSynthesisAgent."""

    system_prompt: str = DEEP_RESEARCH_AGENT_PROMPT
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4000


class ResearchSynthesisAgent(
    InstructorBaseAgent[ResearchSynthesisInputSchema, ResearchSynthesisOutputSchema]
):
    """
    Agent that synthesizes research results into comprehensive reports.

    This agent takes search results and research context, then creates
    well-structured research reports with key findings, citations, and
    quality assessments following scientific research standards.
    """

    input_schema = ResearchSynthesisInputSchema
    output_schema = ResearchSynthesisOutputSchema
    config_schema = ResearchSynthesisAgentConfig

    def __init__(
        self,
        config: ResearchSynthesisAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the ResearchSynthesisAgent with configuration."""
        config = config or ResearchSynthesisAgentConfig()
        super().__init__(config=config, debug=debug)


class ResearchSynthesisComponent:
    """
    Embedded research synthesis component that wraps ResearchSynthesisAgent.

    This component provides the interface expected by literature search agents
    while using the clean agent pattern internally.
    """

    def __init__(
        self,
        config: Optional[ResearchSynthesisAgentConfig] = None,
        debug: bool = False,
    ):
        self.debug = debug
        # Create the internal agent
        self._agent = ResearchSynthesisAgent(config=config, debug=debug)

    async def synthesize(
        self,
        results: List[SearchResultItem],
        research_instructions: str,
        original_query: str,
        quality_scores: List[float],
        research_trace: List[str],
        iterations_performed: int,
    ):
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
            Object with research_report, key_findings, sources_consulted,
            evidence_quality_score, and citations attributes
        """
        if self.debug:
            logger.debug(f"Synthesizing {len(results)} results into research report")

        # Prepare context with all relevant information
        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        )
        context = (
            f"Research Instructions: {research_instructions}\n"
            f"Iterations Performed: {iterations_performed}\n"
            f"Average Quality Score: {avg_quality:.2f}\n"
            f"Research Trace:\n"
            + "\n".join(f"- {trace}" for trace in research_trace[-5:])
        )

        # Progressive shrinking retry strategy for context length errors
        current_results: List[SearchResultItem] = results
        shrink_factor: float = 0.9  # reduce by 10% on each retry
        min_title_chars: int = 1
        min_content_chars: int = 0
        max_attempts: int = 12

        for attempt in range(max_attempts):
            # Create input for the agent with current results
            agent_input = ResearchSynthesisInputSchema(
                query=original_query,
                search_results=current_results,
                context=context,
            )

            try:
                agent_output = await self._agent.arun(agent_input)

                if self.debug:
                    logger.debug("Agent synthesis completed successfully")
                    logger.debug(f"Key findings: {len(agent_output.key_findings)}")
                    logger.debug(
                        f"Evidence quality: {agent_output.evidence_quality_score}"
                    )

                return agent_output

            except Exception as e:
                if self._is_context_length_error(e):
                    if self.debug:
                        logger.warning(
                            "Context length exceeded during synthesis (attempt {}): shrinking results by 10% and retrying".format(
                                attempt + 1,
                            )
                        )
                    # Shrink the current results by the shrink_factor
                    next_results: List[SearchResultItem] = self._shrink_results_by(
                        current_results,
                        shrink_factor,
                        min_title_chars,
                        min_content_chars,
                    )

                    # If shrinking no longer changes the payload meaningfully, stop retrying
                    if self._results_size(next_results) >= self._results_size(
                        current_results,
                    ):
                        break
                    current_results = next_results
                    continue

                # Non context-length errors: log and break to fallback
                logger.error(f"Error in agent synthesis: {e}")
                break

        # Fallback if all retries failed
        return self._create_fallback_output(
            results,
            original_query,
            quality_scores,
            research_trace,
            iterations_performed,
        )

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        """Heuristically detect context length errors from provider exceptions."""
        msg = str(error).lower()
        indicators = [
            "maximum context length",
            "context length exceeded",
            "context_length_exceeded",
            "too many tokens",
            "invalid_request_error",
        ]
        return any(indicator in msg for indicator in indicators)

    @staticmethod
    def _shrink_results_by(
        results: List[SearchResultItem],
        shrink_factor: float,
        min_title_chars: int,
        min_content_chars: int,
    ) -> List[SearchResultItem]:
        """Return a new list of results with title and content shortened by shrink_factor.

        - Title is guaranteed to keep at least min_title_chars if non-empty.
        - Content may be reduced to empty string if needed.
        """

        def truncate(text: str, factor: float, min_chars: int) -> str:
            if not text:
                return text
            new_len = int(len(text) * factor)
            if len(text) > 0 and new_len < min_chars:
                new_len = min(len(text), min_chars)
            return text[:new_len]

        shrunk: List[SearchResultItem] = []
        for item in results:
            new_title = truncate(item.title, shrink_factor, min_title_chars)
            new_content = truncate(item.content, shrink_factor, min_content_chars)
            # Keep other fields the same
            shrunk.append(
                SearchResultItem(
                    url=item.url,
                    title=new_title,
                    query=item.query,
                    pdf_url=item.pdf_url,
                    content=new_content,
                    category=item.category,
                    doi=item.doi,
                    published_date=item.published_date,
                    engine=item.engine,
                    tags=item.tags,
                    score=item.score,
                    extra=item.extra,
                )
            )
        return shrunk

    @staticmethod
    def _results_size(results: List[SearchResultItem]) -> int:
        """Approximate size measure to detect whether shrinking is effective."""
        total = 0
        for r in results:
            total += len(r.title) + len(r.content)
        return total

    def _create_fallback_output(
        self,
        results: List[SearchResultItem],
        original_query: str,
        quality_scores: List[float],
        research_trace: List[str],
        iterations_performed: int,
        num_results_to_analyze: int = 100,
    ):
        """Create a basic fallback output if agent synthesis fails."""

        num_results_to_analyze = min(num_results_to_analyze, len(results))
        if self.debug:
            logger.debug("Using fallback synthesis method")

        # Extract basic information
        source_urls = [str(result.url) for result in results[:num_results_to_analyze]]
        key_findings = []

        for result in results[:num_results_to_analyze]:
            if result.content:
                # Extract first sentence as a simple finding
                sentences = result.content.split(". ")
                if sentences:
                    key_findings.append(sentences[0] + ".")

        # Create basic report
        report = f"""# Research Report

        This research on '{original_query}' analyzed {num_results_to_analyze} sources across {iterations_performed} iterations."""

        report += "\n\n## Key Findings\n\n"
        report += "\n".join(
            f"{i + 1}. {finding}" for i, finding in enumerate(key_findings)
        )
        report += "\n\n## Sources Consulted\n\n"
        report += "\n".join(f"- {url}" for url in source_urls) + "\n"

        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.3
        )

        # Return object with expected attributes
        return ResearchSynthesisOutputSchema(
            research_report=report,
            key_findings=key_findings,
            sources_consulted=source_urls,
            evidence_quality_score=avg_quality,
            citations=[{"url": url, "title": "N/A"} for url in source_urls],
        )
