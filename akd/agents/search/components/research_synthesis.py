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
from .content_condensation import (
    ContentCondensationComponent,
    ContentCondensationConfig,
    ContentCondensationInputSchema,
)
from .content_manager import ContentManager


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
        
        # Create content condensation component
        condensation_config = ContentCondensationConfig(
            model_name="gpt-4o-mini",  # Use a cheaper model for condensation
            temperature=0.1,
            debug=debug
        )
        self._condensation_component = ContentCondensationComponent(
            config=condensation_config, 
            debug=debug
        )
        
        # Create production-grade content manager
        from akd.configs.content_management import LLMContentManagerConfig
        content_config = LLMContentManagerConfig(
            model_name="gpt-4o-mini",  # Use smaller model for content management
            debug=debug
        )
        content_config.token_budget.max_synthesis_tokens = 100000
        self._content_manager = ContentManager(content_config)
    
    async def _condense_content_for_synthesis(
        self,
        results: List[SearchResultItem],
        research_instructions: str,
        max_tokens_per_source: int = 800,
    ) -> List[SearchResultItem]:
        """
        Condense content in search results to focus only on information 
        relevant to the research questions.
        """
        condensed_results = []
        
        for result in results:
            # Skip results without content or with very short content
            if not result.content or len(result.content.strip()) < 100:
                condensed_results.append(result)
                continue
            
            try:
                # Prepare condensation input
                condensation_input = ContentCondensationInputSchema(
                    research_questions=research_instructions,
                    full_text_content=result.content,
                    max_tokens=max_tokens_per_source,
                    source_url=str(result.url),
                    source_title=result.title or ""
                )
                
                # Condense the content
                condensation_output = await self._condensation_component.arun(condensation_input)
                
                # Create new result with condensed content
                condensed_result = result.model_copy()
                if condensation_output.has_relevant_content:
                    condensed_result.content = condensation_output.condensed_content
                    
                    if self.debug:
                        logger.debug(
                            f"Condensed content for {result.url}: "
                            f"{condensation_output.original_tokens} -> {condensation_output.condensed_tokens} tokens "
                            f"(ratio: {condensation_output.compression_ratio:.3f})"
                        )
                else:
                    # If no relevant content found, keep only title and metadata
                    condensed_result.content = ""
                    if self.debug:
                        logger.debug(f"No relevant content found for {result.url}")
                
                condensed_results.append(condensed_result)
                
            except Exception as e:
                if self.debug:
                    logger.warning(f"Error condensing content for {result.url}: {e}")
                # On error, keep original result
                condensed_results.append(result)
        
        return condensed_results

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

        # Use intelligent content management instead of blind shrinking
        if self.debug:
            logger.debug("Using intelligent content management for synthesis")
        
        # Calculate quality-weighted scores for content allocation
        quality_weights = [
            quality_scores[min(i, len(quality_scores) - 1)] if quality_scores 
            else 0.5 for i in range(len(results))
        ]
        
        # Use production-grade content management
        try:
            managed_results, metadata = await self._content_manager.manage_content(
                results=results,
                research_query=original_query,
                target_tokens=self._content_manager.config.token_budget.max_synthesis_tokens,
                context=context
            )
            
            if self.debug:
                logger.debug(
                    f"Content management: {len(results)} -> {len(managed_results)} results, "
                    f"method: {metadata.get('method', 'unknown')}, "
                    f"reduction: {metadata.get('token_reduction', 0):.1%}"
                )
        except Exception as e:
            if self.debug:
                logger.warning(f"Intelligent content management failed, using fallback: {e}")
            managed_results = results[:10]  # Simple fallback: take first 10 results

        # Single synthesis attempt with properly sized content
        agent_input = ResearchSynthesisInputSchema(
            query=original_query,
            search_results=managed_results,
            context=context,
        )

        try:
            agent_output = await self._agent.arun(agent_input)

            if self.debug:
                logger.debug("Agent synthesis completed successfully")
                logger.debug(f"Key findings: {len(agent_output.key_findings)}")
                logger.debug(f"Evidence quality: {agent_output.evidence_quality_score}")

            return agent_output

        except Exception as e:
            if self._is_context_length_error(e):
                if self.debug:
                    logger.error(
                        f"Context length still exceeded after intelligent management. "
                        f"Final token count: {self._content_manager.estimate_total_tokens(managed_results, context)}"
                    )
                
                # Emergency fallback: try with even fewer results
                emergency_results = managed_results[:5]
                emergency_input = ResearchSynthesisInputSchema(
                    query=original_query,
                    search_results=emergency_results,
                    context=context,
                )
                
                try:
                    return await self._agent.arun(emergency_input)
                except Exception as emergency_e:
                    if self.debug:
                        logger.error(f"Emergency fallback also failed: {emergency_e}")
                    raise emergency_e
            else:
                # Non context-length errors: log and fallback
                if self.debug:
                    logger.error(f"Error in agent synthesis: {e}")
                # Fall through to fallback

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
