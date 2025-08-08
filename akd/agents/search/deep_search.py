"""
Deep Literature Search Agent with Embedded Components

Advanced literature search agent implementing multi-agent deep research pattern with
embedded triage, clarification, instruction building, and research synthesis components.
refer to akd/docs/deep_research_agent.md for more details.
"""

from __future__ import annotations

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
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
)
from akd.structures import SearchResultItem
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
    LinkRelevancyAssessorInputSchema,
)
from akd.tools.scrapers import (
    ScraperToolInputSchema,
    SimplePDFScraper,
    SimpleWebScraper,
)
from akd.tools.search.searxng_search import SearxNGSearchTool
from akd.tools.search.semantic_scholar_search import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolInputSchema,
)
from akd.tools.source_validator import (
    SourceValidator,
    SourceValidatorInputSchema,
    create_source_validator,
)

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


class DeepSearchResultItem(SearchResultItem):
    """
    Extended SearchResultItem to include additional fields for deep research.
    This class can be used to store additional metadata like relevancy scores,
    full content fetching status, etc.

    Note:
        - Needed for LinkRelevancyAssessor and DeepLitSearchAgent to handle additional metadata and processing.
    """

    should_fetch_full_content: bool = Field(
        False,
        description="Whether to fetch full content for this result",
    )
    query_alignment_details: Dict[str, Any] | None = Field(
        default_factory=lambda: {},
        description="Details on how this result aligns with the original query",
    )
    relevancy_assessment: Dict[str, Any] | None = Field(
        default_factory=lambda: {},
        description="Relevancy assessment details for this result",
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

    # Search tool selection
    use_semantic_scholar: bool = Field(
        default=True,
        description="Include Semantic Scholar in searches",
    )

    # Link relevancy assessment
    enable_per_link_assessment: bool = Field(
        default=True,
        description="Enable per-link relevancy assessment",
    )
    min_relevancy_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevancy score to include link in results",
    )
    full_content_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relevancy score threshold to trigger full content fetching",
    )
    enable_full_content_scraping: bool = Field(
        default=True,
        description="Enable scraping of full content for high-relevancy links",
    )

    # ISSN whitelist filtering
    enable_issn_whitelist_filter: bool = Field(
        default=False,
        description=(
            "When true, discard search results whose CrossRef ISSN does not match the ISSN whitelist."
        ),
    )
    issn_whitelist_file_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to docs/issn_whitelist.json. If None, tool default is used."
        ),
    )
    issn_validation_timeout_seconds: int = Field(
        default=25,
        ge=1,
        description="Timeout for CrossRef lookups used during ISSN validation (seconds)",
    )
    issn_validation_max_concurrency: int = Field(
        default=8,
        ge=1,
        description="Max concurrent CrossRef requests for ISSN validation",
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

    config_schema = DeepLitSearchAgentConfig

    def __init__(
        self,
        config: DeepLitSearchAgentConfig | None = None,
        search_tool=None,
        semantic_scholar_tool: SemanticScholarSearchTool | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        link_relevancy_assessor: LinkRelevancyAssessor | None = None,
        web_scraper: SimpleWebScraper | None = None,
        pdf_scraper: SimplePDFScraper | None = None,
        triage_component: TriageComponent | None = None,
        clarification_component: ClarificationComponent | None = None,
        instruction_component: InstructionBuilderComponent | None = None,
        research_synthesis_component: ResearchSynthesisComponent | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the DeepLitSearchAgent with embedded components."""
        super().__init__(config=config or DeepLitSearchAgentConfig(), debug=debug)

        self.query_agent = query_agent or QueryAgent()
        self.followup_query_agent = followup_query_agent or FollowUpQueryAgent()

        # Initialize search tools
        self.search_tool = search_tool or SearxNGSearchTool()
        self.semantic_scholar_tool = (
            (semantic_scholar_tool or SemanticScholarSearchTool())
            if self.config.use_semantic_scholar
            else None
        )

        # Initialize relevancy agent
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()

        # Initialize link relevancy assessor if enabled
        if self.config.enable_per_link_assessment:
            if link_relevancy_assessor is not None:
                self.link_relevancy_assessor = link_relevancy_assessor
            else:
                assessor_config = LinkRelevancyAssessorConfig(
                    min_relevancy_score=self.config.min_relevancy_score,
                    full_content_threshold=self.config.full_content_threshold,
                    debug=debug,
                )
                self.link_relevancy_assessor = LinkRelevancyAssessor(
                    config=assessor_config,
                    relevancy_agent=self.relevancy_agent,
                    debug=debug,
                )
        else:
            self.link_relevancy_assessor = (
                link_relevancy_assessor  # Could be None or an injected instance
            )

        # Initialize scrapers for full content fetching
        if self.config.enable_full_content_scraping:
            self.web_scraper = web_scraper or SimpleWebScraper(debug=debug)
            self.pdf_scraper = pdf_scraper or SimplePDFScraper(debug=debug)
        else:
            self.web_scraper = web_scraper  # Could be None or an injected instance
            self.pdf_scraper = pdf_scraper  # Could be None or an injected instance

        # Initialize ISSN whitelist validator if enabled
        self._issn_validator: SourceValidator | None = None
        if self.config.enable_issn_whitelist_filter:
            try:
                self._issn_validator = create_source_validator(
                    whitelist_file_path=self.config.issn_whitelist_file_path,
                    timeout_seconds=self.config.issn_validation_timeout_seconds,
                    max_concurrent_requests=self.config.issn_validation_max_concurrency,
                    debug=debug,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ISSN whitelist validator: {e}")
                self._issn_validator = None

        # Initialize embedded components
        self.triage_component = triage_component or TriageComponent(debug=debug)
        self.clarification_component = (
            clarification_component or ClarificationComponent(debug=debug)
        )
        self.instruction_component = (
            instruction_component or InstructionBuilderComponent(debug=debug)
        )
        self.research_synthesis_component = (
            research_synthesis_component or ResearchSynthesisComponent(debug=debug)
        )

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

        return {
            "routing_decision": triage_output.routing_decision,
            "needs_clarification": triage_output.needs_clarification,
            "reasoning": triage_output.reasoning,
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
            mock_answers,
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

            # Evaluate quality
            if new_results:
                quality_score = await self._evaluate_research_quality(
                    new_results,
                    original_query,
                )
                quality_scores.append(quality_score)

                research_trace.append(
                    f"Iteration {iterations}: Found {len(new_results)} new results, "
                    f"quality score: {quality_score:.2f}",
                )

                # Check if we've reached quality threshold
                avg_quality = sum(quality_scores) / len(quality_scores)
                if (
                    avg_quality >= self.config.quality_threshold
                    and len(all_results) >= 10
                ):
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
            "sources_consulted": research_output.sources_consulted,
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

        query_output = await self.query_agent.arun(query_input)

        if self.debug:
            logger.info("🧠 DeepLitSearchAgent - INITIAL QUERIES GENERATED:")
            for i, query in enumerate(query_output.queries, 1):
                logger.info(f"  {i}. '{query}'")

        return query_output.queries

    async def _execute_searches(
        self,
        queries: List[str],
        original_query: str | None = None,
        is_reformulated: bool = False,
    ) -> List[DeepSearchResultItem]:
        """Execute searches using available search tools."""
        all_results = []

        search_input = self.search_tool.input_schema(
            queries=queries,
            max_results=20,
            category="science",
        )

        primary_results = await self.search_tool.arun(search_input)
        all_results.extend(primary_results.results)

        # Search with Semantic Scholar if enabled
        if self.semantic_scholar_tool and self.config.use_semantic_scholar:
            ss_input = SemanticScholarSearchToolInputSchema(
                queries=queries,
                max_results=20,
                category="science",
            )
            ss_results = await self.semantic_scholar_tool.arun(ss_input)
            all_results.extend(ss_results.results)

        all_results = list(map(lambda r: DeepSearchResultItem(**r.dict()), all_results))

        # Optional ISSN whitelist filtering
        if self._issn_validator and all_results:
            try:
                input_payload = SourceValidatorInputSchema(search_results=all_results)
                validation_output = await self._issn_validator.arun(input_payload)
                # Keep only items that passed the whitelist
                filtered: List[DeepSearchResultItem] = []
                for item, v in zip(all_results, validation_output.validated_results):
                    if v.is_whitelisted:
                        filtered.append(item)
                if self.debug:
                    logger.debug(
                        "ISSN whitelist filter: kept %d of %d results",
                        len(filtered),
                        len(all_results),
                    )
                all_results = filtered
            except Exception as e:
                logger.warning(
                    f"ISSN whitelist filtering failed; proceeding unfiltered: {e}"
                )
        # Apply per-link relevancy assessment if enabled
        if self.link_relevancy_assessor and all_results:
            if self.debug:
                logger.debug(
                    f"Assessing relevancy for {len(all_results)} search results",
                )

            reformulated_query = None
            if is_reformulated and original_query:
                reformulated_query = (
                    queries[0] if queries and queries[0] != original_query else None
                )

            assessor_input = LinkRelevancyAssessorInputSchema(
                search_results=all_results,
                original_query=original_query or queries[0],
                reformulated_query=reformulated_query,
                domain_context=f"Research iteration with {len(queries)} query variations"
                if len(queries) > 1
                else None,
            )

            try:
                assessment_output = await self.link_relevancy_assessor.arun(
                    assessor_input,
                )
                if self.debug:
                    logger.debug(
                        f"Relevancy assessment summary: {assessment_output.assessment_summary}",
                    )
                return assessment_output.filtered_results
            except Exception as e:
                logger.warning(f"Error in relevancy assessment: {e}")

        # Fetch full content for high-relevancy results if enabled
        if self.web_scraper and all_results:
            all_results = await self._fetch_full_content_for_high_relevancy(all_results)

        return all_results

    async def _fetch_full_content_for_high_relevancy(
        self,
        results: List[DeepSearchResultItem],
    ) -> List[DeepSearchResultItem]:
        """Fetch full content for results marked as high-relevancy."""
        high_relevancy_results = [
            r for r in results if getattr(r, "should_fetch_full_content", False)
        ]

        if not high_relevancy_results:
            return results

        if self.debug:
            logger.debug(
                f"Fetching full content for {len(high_relevancy_results)} high-relevancy results",
            )

        for result in high_relevancy_results:
            try:
                # Try PDF first if available
                if hasattr(result, "pdf_url") and result.pdf_url and self.pdf_scraper:
                    try:
                        pdf_input = ScraperToolInputSchema(url=str(result.pdf_url))
                        pdf_content = await self.pdf_scraper.arun(pdf_input)
                        if pdf_content.content and len(pdf_content.content) > 500:
                            result.content = pdf_content.content
                            if self.debug:
                                logger.debug(
                                    f"Fetched PDF content for {result.url} ({len(result.content)} chars)",
                                )
                            continue
                    except Exception as e:
                        if self.debug:
                            logger.debug(
                                f"PDF scraping failed for {result.pdf_url}: {e}",
                            )

                # Fall back to web scraping
                if self.web_scraper:
                    web_input = ScraperToolInputSchema(url=str(result.url))
                    web_content = await self.web_scraper.arun(web_input)
                    if web_content.content and len(web_content.content) > len(
                        result.content or "",
                    ):
                        result.content = web_content.content
                        if self.debug:
                            logger.debug(
                                f"Fetched web content for {result.url} ({len(result.content)} chars)",
                            )

            except Exception as e:
                if self.debug:
                    logger.debug(f"Content fetching failed for {result.url}: {e}")
                continue

        return results

    def _deduplicate_results(
        self,
        new_results: List[DeepSearchResultItem],
        existing_results: List[DeepSearchResultItem],
    ) -> List[DeepSearchResultItem]:
        """Remove duplicate results based on URL and title."""
        existing_urls = {r.url for r in existing_results}
        existing_titles = {r.title.lower() for r in existing_results if r.title}

        unique_results = []
        for result in new_results:
            if result.url not in existing_urls:
                if not result.title or result.title.lower() not in existing_titles:
                    unique_results.append(result)

        return unique_results

    async def _evaluate_research_quality(
        self,
        results: List[DeepSearchResultItem],
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

        rubric_output = await self.relevancy_agent.arun(rubric_input)

        # Calculate quality score from rubrics
        from akd.agents.relevancy import (
            ContentDepthLabel,
            EvidenceQualityLabel,
            MethodologicalRelevanceLabel,
            RecencyRelevanceLabel,
            ScopeRelevanceLabel,
            TopicAlignmentLabel,
        )

        positive_count = sum(
            [
                rubric_output.topic_alignment == TopicAlignmentLabel.ALIGNED,
                rubric_output.content_depth == ContentDepthLabel.COMPREHENSIVE,
                rubric_output.evidence_quality
                == EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
                rubric_output.methodological_relevance
                == MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
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
        enhanced_content = (
            f"Research Instructions: {instructions}\n\nCurrent Results:\n{content}"
        )

        followup_input = FollowUpQueryAgentInputSchema(
            original_queries=previous_queries,
            content=enhanced_content,
            num_queries=3,
        )

        followup_output = await self.followup_query_agent.arun(followup_input)

        if self.debug:
            logger.info("🔄 DeepLitSearchAgent - REFINED QUERIES GENERATED:")
            for i, query in enumerate(followup_output.followup_queries, 1):
                is_original = query in previous_queries
                marker = "🎯" if is_original else "🔄"
                logger.info(f"  {i}. {marker} '{query}'")

        return followup_output.followup_queries

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

        # Step 2: Handle based on triage decision
        enriched_query = original_query
        clarifications = None

        if triage_result["needs_clarification"] and self.config.auto_clarify:
            enriched_query, clarifications = await self._handle_clarification(
                original_query,
                kwargs.get("mock_answers"),  # TODO: Get answers from user input
            )

        # Step 3: Build research instructions using embedded component
        instructions = await self._build_research_instructions(
            enriched_query,
            clarifications,
        )

        # Step 4: Perform deep research using embedded components
        research_output = await self._perform_deep_research(
            instructions,
            original_query,
        )

        # Step 5: Convert research output to agent output format
        # Convert SearchResultItem objects to dictionaries for output
        results_as_dicts = []
        for result in research_output["results"]:
            result_dict = {
                "url": str(result.url),
                "title": result.title or "Untitled",
                "content": result.content,
                "category": getattr(result, "category", "science"),
            }
            # Preserve relevancy information if available
            if (
                hasattr(result, "relevancy_score")
                and result.relevancy_score is not None
            ):
                result_dict["relevancy_score"] = result.relevancy_score
            if hasattr(result, "should_fetch_full_content"):
                result_dict["full_content_fetched"] = result.should_fetch_full_content
            results_as_dicts.append(result_dict)

        # Add research report as the first result
        if results_as_dicts:
            results_as_dicts.insert(
                0,
                {
                    "url": "deep-research://report",
                    "title": "Deep Research Report",
                    "content": research_output["research_report"],
                    "category": "research",
                    "key_findings": research_output["key_findings"],
                    "quality_score": research_output["evidence_quality_score"],
                    "iterations": research_output["iterations_performed"],
                },
            )

        return LitSearchAgentOutputSchema(
            results=results_as_dicts,
            category=params.category,
            iterations_performed=research_output["iterations_performed"],
        )
