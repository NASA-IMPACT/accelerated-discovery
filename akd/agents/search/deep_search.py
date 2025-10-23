"""
Deep Literature Search Agent with Embedded Components

Advanced literature search agent implementing multi-agent deep research pattern with
embedded triage, clarification, instruction building, and research synthesis components.
refer to akd/docs/deep_research_agent.md for more details.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field

from akd.agents.query import (
    FollowUpQueryAgent,
    FollowUpQueryAgentInputSchema,
    QueryAgent,
    QueryAgentInputSchema,
)
from akd.agents.relevancy import (
    ContentDepthLabel,
    EvidenceQualityLabel,
    MethodologicalRelevanceLabel,
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
    RecencyRelevanceLabel,
    ScopeRelevanceLabel,
    TopicAlignmentLabel,
)
from akd.structures import SearchResultItem
from akd.tools.search import SearchTool
from akd.tools.search.pipeline import SearchPipeline
from akd.tools.search.searxng import SearxNGSearchTool

from ._base import (
    LitBaseAgent,
    LitSearchAgentConfig,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
)
from .components import (
    ClarificationComponent,
    InstructionBuilderComponent,
    ResearchSynthesisComponent,
    TriageComponent,
)


class DeepLitSearchAgentConfig(LitSearchAgentConfig):
    """
    Configuration for the DeepLitSearchAgent that implements multi-agent deep research.
    """

    # Research parameters
    max_research_iterations: int = Field(
        default=5,
        description="Maximum number of research iterations",
    )

    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Quality threshold for stopping research (0-1)",
    )

    # Agent behavior
    auto_clarify: bool = Field(
        default=True,
        description="Automatically ask clarifying questions if needed",
    )

    max_clarifying_rounds: int = Field(
        default=1,
        description="Maximum rounds of clarification",
    )

    # Streaming and progress
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming of research progress",
    )


class DeepLitSearchAgent(LitBaseAgent):
    """
    Advanced literature search agent implementing multi-agent deep research pattern
    with embedded components.

    This agent orchestrates embedded components to:
    1. Triage and clarify research queries
    2. Build detailed research instructions
    3. Perform iterative deep research with quality checks
    4. Produce comprehensive, well-structured research reports

    The implementation follows the OpenAI Deep Research pattern but is adapted
    to work within the akd framework using embedded components.
    """

    input_schema = LitSearchAgentInputSchema
    output_schema = LitSearchAgentOutputSchema
    config_schema = DeepLitSearchAgentConfig

    def __init__(
        self,
        config: DeepLitSearchAgentConfig | None = None,
        search_tool: SearchTool | SearchPipeline | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        triage_component: TriageComponent | None = None,
        clarification_component: ClarificationComponent | None = None,
        instruction_component: InstructionBuilderComponent | None = None,
        research_synthesis_component: ResearchSynthesisComponent | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the DeepLitSearchAgent with embedded components.
        Args:
            config: Configuration for the agent.
            search_tool: Primary search tool or pipeline to use.
                Note: SearchPipeline is also an implementation of SearchTool.
            query_agent: Agent for generating initial search queries.
            followup_query_agent: Agent for refining search queries.
            relevancy_agent: Agent for evaluating research quality.
            triage_component: Embedded component for query triage.
            clarification_component: Embedded component for query clarification.
            instruction_component: Embedded component for building research instructions.
            research_synthesis_component: Embedded component for synthesizing research findings.
            debug: Enable debug logging.
        """
        super().__init__(config=config or DeepLitSearchAgentConfig(), debug=debug)

        self.query_agent = query_agent or QueryAgent()
        self.followup_query_agent = followup_query_agent or FollowUpQueryAgent()
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()

        # default to searxng-based pipeline if no search tool provided
        self.search_tool = search_tool or SearchPipeline(
            search_tool=SearxNGSearchTool(debug=debug),
            debug=debug,
        )

        # Initialize embedded components
        self.triage_component = triage_component or TriageComponent(debug=debug)
        self.clarification_component = clarification_component or ClarificationComponent(debug=debug)
        self.instruction_component = instruction_component or InstructionBuilderComponent(debug=debug)
        self.research_synthesis_component = research_synthesis_component or ResearchSynthesisComponent(debug=debug)

        # Track research state
        self.research_history = []
        self.clarification_history = []

    async def _handle_triage(self, query: str) -> dict:
        """Handle query triage using embedded component."""
        if self.debug:
            logger.debug(f"Starting triage for query: {query}")

        triage_output = await self.triage_component.process(query)

        if self.debug:
            logger.debug(f"Triage decision: {triage_output.routing_decision}")
            logger.debug(f"Reasoning: {triage_output.reasoning}")

        try:
            return {
                "routing_decision": triage_output.routing_decision,
                "needs_clarification": triage_output.needs_clarification,
                "reasoning": triage_output.reasoning,
            }
        except Exception as e:
            if self.debug:
                logger.warning(
                    f"Triage component failed: {e}. Using fallback behavior.",
                )

            # Fallback: assume no clarification needed, proceed with research
            return {
                "routing_decision": "research",
                "needs_clarification": False,
                "reasoning": "Triage component failed - proceeding with fallback behavior",
            }

    async def _handle_clarification(
        self,
        query: str,
        mock_answers: Optional[Dict[str, str]] = None,
    ) -> tuple[str, List[str]]:
        """Handle the clarification process using embedded component."""
        if self.debug:
            logger.debug("Starting clarification process")

        enriched_query, clarifications = await self.clarification_component.process(
            query,
            search_results=None,  # No search results available at clarification stage
            mock_answers=mock_answers,
        )

        self.clarification_history.extend(clarifications)

        if self.debug:
            logger.debug(f"Generated {len(clarifications)} clarifications")

        return enriched_query, clarifications

    async def _build_research_instructions(
        self,
        query: str,
        clarifications: Optional[List[str]] = None,
    ) -> str:
        """Build detailed research instructions using embedded component."""
        if self.debug:
            logger.debug("Building research instructions")

        instructions = await self.instruction_component.process(query, clarifications)

        if self.debug:
            logger.debug(f"Generated instructions ({len(instructions)} chars)")

        return instructions

    async def _perform_deep_research(
        self,
        instructions: str,
        original_query: str,
    ) -> dict:
        """
        Perform the actual deep research using iterative search and synthesis.

        This method coordinates search tools, relevancy checking, and the
        embedded research synthesis component to produce comprehensive results.
        """
        # Initialize research tracking
        all_results = []
        iterations = 0
        quality_scores = []
        research_trace = []

        # Initial search queries from instructions
        initial_queries = await self._generate_initial_queries(instructions)

        while iterations < self.config.max_research_iterations:
            iterations += 1
            research_trace.append(
                f"Iteration {iterations}: Searching with queries: {initial_queries}",
            )

            if self.debug:
                logger.debug(
                    f"Research iteration {iterations}/{self.config.max_research_iterations}",
                )

            # Perform searches
            search_results = await self._execute_searches(
                initial_queries,
                original_query,
                is_reformulated=(iterations > 1),
            )

            if not search_results:
                research_trace.append(f"Iteration {iterations}: No new results found")
                break

            # Deduplicate and add to results
            new_results = self._deduplicate_results(search_results, all_results)
            all_results.extend(new_results)

            # Cap total results to prevent memory issues
            if len(all_results) > 50:
                all_results = all_results[:50]

                if self.debug:
                    logger.debug(
                        f"Capped results: keeping first {len(all_results)} results",
                    )
            # Evaluate quality
            if new_results:
                quality_score = await self._evaluate_research_quality(
                    new_results,
                    original_query,
                )
                quality_scores.append(quality_score)

                research_trace.append(
                    f"Iteration {iterations}: Found {len(new_results)} new results, quality score: {quality_score:.2f}",
                )

                # Check if we've reached quality threshold
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality >= self.config.quality_threshold and len(all_results) >= 10:
                    research_trace.append(
                        f"Stopping: Quality threshold reached ({avg_quality:.2f})",
                    )
                    break

            # Generate refined queries for next iteration
            if iterations < self.config.max_research_iterations:
                initial_queries = await self._generate_refined_queries(
                    initial_queries,
                    all_results,
                    instructions,
                )

        # Synthesize final research report using embedded component
        research_output = await self.research_synthesis_component.synthesize(
            all_results,
            instructions,
            original_query,
            quality_scores,
            research_trace,
            iterations,
        )

        return {
            "research_report": research_output.research_report,
            "key_findings": research_output.key_findings,
            "evidence_quality_score": research_output.evidence_quality_score,
            "citations": research_output.citations,
            "iterations_performed": iterations,
            "results": all_results,
        }

    async def _generate_initial_queries(self, instructions: str) -> List[str]:
        """Generate initial search queries from research instructions."""
        query_input = QueryAgentInputSchema(
            query=instructions,
            num_queries=5,  # More queries for comprehensive coverage
        )

        if self.debug:
            logger.debug(
                f"QueryAgent input preview | instructions: {instructions[:200]}",
            )

        query_output = await self.query_agent.arun(query_input)

        if self.debug:
            logger.info("🧠 DeepLitSearchAgent - INITIAL QUERIES GENERATED:")
            for i, query in enumerate(query_output.queries, 1):
                logger.info(f"  {i}. '{query}'")
            logger.debug(
                f"QueryAgent output preview | first query: {(query_output.queries[0] if query_output.queries else '')[:200]}",
            )

        return query_output.queries

    async def _execute_searches(
        self,
        queries: List[str],
        original_query: str | None = None,
        is_reformulated: bool = False,
    ) -> List[SearchResultItem]:
        """Execute searches using available search tools."""
        all_results = []

        # Use primary search tool (SearchPipeline)
        tasks: List[asyncio.Task] = []
        tool_names: List[str] = []

        reformulated_query = None
        if is_reformulated and original_query:
            reformulated_query = queries[0] if queries and queries[0] != original_query else None

        domain_context = f"Research iteration with {len(queries)} query variations" if len(queries) > 1 else None

        # Primary search tool
        try:
            tool_input = self.search_tool.input_schema(
                queries=queries,
            )
            tasks.append(
                asyncio.create_task(
                    self.search_tool.arun(
                        tool_input,
                        original_query=original_query,
                        reformulated_query=reformulated_query,
                        domain_context=domain_context,
                    ),
                ),
            )
            tool_names.append(type(self.search_tool).__name__)
        except Exception as e:
            logger.warning(f"search tool error: {e}")

        if tasks:
            results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, res in enumerate(results_or_errors):
                name = tool_names[idx] if idx < len(tool_names) else f"Tool#{idx}"
                if isinstance(res, Exception):
                    logger.warning(f"{name} failed: {res}")
                    continue
                try:
                    all_results.extend(res.results)
                except Exception as e:  # defensive against unexpected shapes
                    logger.warning(f"{name} unexpected search result shape: {e}")

        return all_results

    def _deduplicate_results(
        self,
        new_results: List[SearchResultItem],
        existing_results: List[SearchResultItem],
    ) -> List[SearchResultItem]:
        """Remove duplicate results based on URL or title."""
        existing_urls = {r.url for r in existing_results}
        existing_titles = {r.title.lower() for r in existing_results if r.title}

        unique_results = []
        for result in new_results:
            if result.url not in existing_urls and (not result.title or result.title.lower() not in existing_titles):
                unique_results.append(result)

        return unique_results

    async def _evaluate_research_quality(
        self,
        results: List[SearchResultItem],
        query: str,
    ) -> float:
        """Evaluate the quality of research results."""
        if not results:
            return 0.0

        # Accumulate content for evaluation
        content = "\n\n".join(
            [
                f"Title: {r.title}\nContent: {r.content}"
                for r in results[:5]  # Evaluate top 5 results
            ],
        )

        rubric_input = MultiRubricRelevancyInputSchema(
            content=content,
            query=query,
        )

        if self.debug:
            logger.debug(
                f"RelevancyAgent input preview | query: {query[:200]} | content: {content[:200]}",
            )

        rubric_output = await self.relevancy_agent.arun(rubric_input)

        if self.debug:
            logger.debug(
                f"RelevancyAgent output preview | topic_alignment: {rubric_output.topic_alignment} | content_depth: {rubric_output.content_depth}",
            )

        # Calculate quality score from rubrics
        positive_count = sum(
            [
                rubric_output.topic_alignment == TopicAlignmentLabel.ALIGNED,
                rubric_output.content_depth == ContentDepthLabel.COMPREHENSIVE,
                rubric_output.evidence_quality == EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
                rubric_output.methodological_relevance == MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
                rubric_output.recency_relevance == RecencyRelevanceLabel.CURRENT,
                rubric_output.scope_relevance == ScopeRelevanceLabel.IN_SCOPE,
            ],
        )

        return positive_count / 6  # Total number of rubrics

    async def _generate_refined_queries(
        self,
        previous_queries: List[str],
        results: List[SearchResultItem],
        instructions: str,
    ) -> List[str]:
        """Generate refined queries based on current results."""
        # Create content summary from results
        content = "\n\n".join(
            [
                f"Title: {r.title}\nSummary: {r.content[:200]}..."
                for r in results[-10:]  # Use recent results
            ],
        )

        # Enhance content with research instructions context
        enhanced_content = f"Research Instructions: {instructions}\n\nCurrent Results:\n{content}"

        followup_input = FollowUpQueryAgentInputSchema(
            original_queries=previous_queries,
            content=enhanced_content,
            num_queries=3,
        )

        if self.debug:
            logger.debug(
                f"FollowUpQueryAgent input preview | content: {enhanced_content[:200]}",
            )

        followup_output = await self.followup_query_agent.arun(followup_input)

        if self.debug:
            logger.info("🔄 DeepLitSearchAgent - REFINED QUERIES GENERATED:")
            for i, query in enumerate(followup_output.followup_queries, 1):
                is_original = query in previous_queries
                marker = "🎯" if is_original else "🔄"
                logger.info(f"  {i}. {marker} '{query}'")
            logger.debug(
                f"FollowUpQueryAgent output preview | first query: {(followup_output.followup_queries[0] if followup_output.followup_queries else '')[:200]}",
            )

        return followup_output.followup_queries

    async def _generate_report(
        self,
        query: str,
        results: List[SearchResultItem],
        **kwargs,
    ) -> str:
        """
        Generate a detailed research report from search results.

        This is handled by the ResearchSynthesisComponent in DeepLitSearchAgent,
        so this method returns the pre-generated report from kwargs.

        Args:
            query: The original research query
            results: List of search results
            **kwargs: Must contain 'research_report' key with the generated report

        Returns:
            The detailed research report
        """
        # For DeepLitSearchAgent, the report is generated by ResearchSynthesisComponent
        # So we just return it from kwargs
        return kwargs.get("research_report", "")

    async def _arun(
        self,
        params: LitSearchAgentInputSchema,
        **kwargs: Any,
    ) -> LitSearchAgentOutputSchema:
        """
        Run the DeepLitSearchAgent with multi-agent orchestration using embedded components.

        This implements the full deep research pipeline:
        1. Triage the query
        2. Clarify if needed
        3. Build research instructions
        4. Perform deep research
        5. Return structured results
        """
        original_query = params.query

        # Step 1: Triage the query using embedded component
        triage_result = await self._handle_triage(original_query)

        # Step 2: Clarification loop (LLM-driven) if needed
        enriched_query = original_query
        clarifications: List[str] | None = []

        if triage_result["needs_clarification"] and self.config.auto_clarify:
            max_rounds = max(1, getattr(self.config, "max_clarifying_rounds", 1))
            for _ in range(max_rounds):
                enriched_query, new_clarifications = await self._handle_clarification(
                    enriched_query,
                    kwargs.get("mock_answers"),
                )
                if new_clarifications:
                    clarifications.extend(new_clarifications)

                # Re-triage to see if more clarification is needed
                try:
                    triage_result = await self._handle_triage(enriched_query)
                except Exception:
                    break

                if not triage_result.get("needs_clarification"):
                    break

        # Step 3: Build research instructions using embedded component
        instructions = await self._build_research_instructions(
            enriched_query,
            clarifications or None,
        )

        # Step 4: Perform deep research using embedded components
        research_output = await self._perform_deep_research(
            instructions,
            original_query,
        )

        # Step 5: Generate shortform answer and report
        detailed_report = await self._generate_report(
            query=original_query,
            results=research_output["results"],
            research_report=research_output["research_report"],
        )

        shortform_answer = await self._generate_answer(
            query=original_query,
            search_results=research_output["results"],
            additional_context=detailed_report,
        )

        # Step 6: Return research output with SearchResultItem objects directly
        return LitSearchAgentOutputSchema(
            answer=shortform_answer.answer,
            report=detailed_report,
            results=research_output["results"],
            iterations_performed=research_output["iterations_performed"],
            extra={
                "key_findings": research_output["key_findings"],
                "evidence_quality_score": research_output["evidence_quality_score"],
                "citations": research_output["citations"],
                "answer_reasoning_traces": shortform_answer.reasoning_traces,
            },
        )
