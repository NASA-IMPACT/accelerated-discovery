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
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
)
from akd.structures import SearchResultItem
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
    LinkRelevancyAssessorInputSchema,
)
from akd.tools.resolvers import (
    ADSResolver,
    ArxivResolver,
    CrossRefDoiResolver,
    DOIResolver,
    PDFUrlResolver,
    ResearchArticleResolver,
    UnpaywallResolver,
)
from akd.tools.scrapers import (
    ScraperToolInputSchema,
    SimplePDFScraper,
    SimpleWebScraper,
)
from akd.tools.scrapers.composite import CompositeScraper
from akd.tools.scrapers.omni import DoclingScraper
from akd.tools.scrapers.web_scrapers import Crawl4AIWebScraper
from akd.tools.search.searxng_search import SearxNGSearchTool
from akd.tools.search.semantic_scholar_search import SemanticScholarSearchTool
from akd.tools.source_validator import SourceValidator, SourceValidatorInputSchema

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

    # Notebook compatibility options (accepted but not strictly required here)
    # Some callers may provide a concrete search tool via config; we accept and ignore it.
    search_tool: Any | None = Field(
        default=None,
        description="Optional search tool; accepted for compatibility, not used",
    )
    source_validation: bool = Field(
        default=True,
        description="Enable ISSN-based source validation",
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

        # Compose active search tools
        self.search_tools: List[Any] = []
        if self.search_tool is not None:
            self.search_tools.append(self.search_tool)
        if self.semantic_scholar_tool is not None:
            self.search_tools.append(self.semantic_scholar_tool)

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

        # Initialize scrapers/resolvers for full content fetching
        if self.config.enable_full_content_scraping:
            # Centralized, best-defaults resolver and scraper
            self.resolver = (
                web_scraper  # type: ignore[assignment]
                if False
                else ResearchArticleResolver(
                    PDFUrlResolver(debug=debug),
                    ArxivResolver(debug=debug),
                    ADSResolver(debug=debug),
                    DOIResolver(debug=debug),
                    CrossRefDoiResolver(debug=debug),
                    UnpaywallResolver(debug=debug),
                    debug=debug,
                )
            )
            # Build a composite scraper once; prefer Docling, then Crawl4AI, then simple web/pdf
            self.scraper = CompositeScraper(
                DoclingScraper(debug=debug),
                Crawl4AIWebScraper(debug=debug),
                (web_scraper or SimpleWebScraper(debug=debug)),
                (pdf_scraper or SimplePDFScraper(debug=debug)),
                debug=debug,
            )
        else:
            self.resolver = None
            self.scraper = None

        # Optional source validator (ISSN whitelist). Ensure attribute always exists.
        self._source_validator = None
        try:
            if getattr(self.config, "source_validation", False):
                self._source_validator = SourceValidator(debug=debug)
        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to initialize SourceValidator: {e}")

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

        try:
            triage_output = await self.triage_component.process(query)

            if self.debug:
                logger.debug(f"Triage decision: {triage_output.routing_decision}")
                logger.debug(f"Reasoning: {triage_output.reasoning}")

            return {
                "routing_decision": triage_output.routing_decision,
                "needs_clarification": triage_output.needs_clarification,
                "reasoning": triage_output.reasoning,
            }
        except Exception as e:
            if self.debug:
                logger.warning(
                    f"Triage component failed: {e}. Using fallback behavior."
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

            # Cap total results to prevent memory issues
            if len(all_results) > 50:
                all_results = all_results[:50]

                if self.debug:
                    logger.debug(
                        f"Capped results: keeping first {len(all_results)} results"
                    )

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
                f"QueryAgent input preview | instructions: {instructions[:200]}"
            )

        query_output = await self.query_agent.arun(query_input)

        if self.debug:
            logger.info("🧠 DeepLitSearchAgent - INITIAL QUERIES GENERATED:")
            for i, query in enumerate(query_output.queries, 1):
                logger.info(f"  {i}. '{query}'")
            logger.debug(
                f"QueryAgent output preview | first query: {(query_output.queries[0] if query_output.queries else '')[:200]}"
            )

        return query_output.queries

    async def _execute_searches(
        self,
        queries: List[str],
        original_query: str | None = None,
        is_reformulated: bool = False,
    ) -> List[DeepSearchResultItem]:
        """Execute searches using available search tools."""
        all_results = []

        # Launch all configured search tools concurrently
        tasks: List[asyncio.Task] = []
        tool_names: List[str] = []
        for tool in getattr(self, "search_tools", []):
            try:
                # Use each tool's own input schema to ensure compatibility
                tool_input = tool.input_schema(
                    queries=queries,
                    max_results=self.search_tool.max_results,
                    category="science",
                )
                tasks.append(asyncio.create_task(tool.arun(tool_input)))
                tool_names.append(type(tool).__name__)
            except Exception as e:
                logger.warning(
                    f"Skipping tool {type(tool).__name__} due to init error: {e}"
                )

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

        all_results = list(
            map(lambda r: DeepSearchResultItem(**r.model_dump()), all_results)
        )

        # Optional source validation (ISSN whitelist) handled intrinsically by the validator
        if self._source_validator and all_results:
            try:
                input_payload = SourceValidatorInputSchema(
                    search_results=all_results,
                )
                validation_output = await self._source_validator.arun(input_payload)
                # Keep only items that passed validation
                filtered: List[DeepSearchResultItem] = []
                for item, v in zip(all_results, validation_output.validated_results):
                    if v.is_whitelisted:
                        filtered.append(item)
                if self.debug:
                    logger.debug(
                        "Source validation filter: kept %d of %d results",
                        len(filtered),
                        len(all_results),
                    )
                all_results = filtered
            except Exception as e:
                logger.warning(
                    f"Source validation failed; dropping results due to strict validation mode: {e}"
                )
                all_results = []
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
        if getattr(self, "scraper", None) and all_results:
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
                # Determine best target URL: prefer explicit PDF, else resolved OA URL, else original
                target_url: str = str(result.url)
                # Try resolver once to improve URL and enrich metadata
                if getattr(self, "resolver", None) is not None:
                    try:
                        resolved = await self.resolver.arun(
                            self.resolver.input_schema(**result.model_dump())
                        )
                        target_url = str(
                            resolved.resolved_url or resolved.url or result.url
                        )
                        if getattr(resolved, "doi", None):
                            result.doi = resolved.doi
                        if getattr(resolved, "authors", None):
                            result.authors = resolved.authors
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"Resolution failed for {result.url}: {e}")

                # If the result already carries a direct PDF URL, prefer it
                if getattr(result, "pdf_url", None):
                    target_url = str(result.pdf_url)

                # Scrape once using the composite scraper
                web_content = await self.scraper.arun(
                    ScraperToolInputSchema(url=target_url)
                )
                if (
                    web_content
                    and web_content.content
                    and len(web_content.content) > len(result.content or "")
                ):
                    result.content = web_content.content
                    if self.debug:
                        logger.debug(
                            f"Fetched content for {target_url} ({len(result.content)} chars)",
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

        if self.debug:
            logger.debug(
                f"RelevancyAgent input preview | query: {query[:200]} | content: {content[:200]}"
            )

        rubric_output = await self.relevancy_agent.arun(rubric_input)

        if self.debug:
            logger.debug(
                f"RelevancyAgent output preview | topic_alignment: {rubric_output.topic_alignment} | content_depth: {rubric_output.content_depth}"
            )

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

        if self.debug:
            logger.debug(
                f"FollowUpQueryAgent input preview | content: {enhanced_content[:200]}"
            )

        followup_output = await self.followup_query_agent.arun(followup_input)

        if self.debug:
            logger.info("🔄 DeepLitSearchAgent - REFINED QUERIES GENERATED:")
            for i, query in enumerate(followup_output.followup_queries, 1):
                is_original = query in previous_queries
                marker = "🎯" if is_original else "🔄"
                logger.info(f"  {i}. {marker} '{query}'")
            logger.debug(
                f"FollowUpQueryAgent output preview | first query: {(followup_output.followup_queries[0] if followup_output.followup_queries else '')[:200]}"
            )

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
