from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from pydantic.fields import Field

from akd.agents.query import (
    FollowUpQueryAgent,
    FollowUpQueryAgentInputSchema,
    FollowUpQueryAgentOutputSchema,
    QueryAgent,
    QueryAgentInputSchema,
    QueryAgentOutputSchema,
)
from akd.agents.relevancy import (
    ContentDepthLabel,
    EvidenceQualityLabel,
    MethodologicalRelevanceLabel,
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
    MultiRubricRelevancyOutputSchema,
    RecencyRelevanceLabel,
    ScopeRelevanceLabel,
    TopicAlignmentLabel,
)
from akd.agents.schemas import DeepResearchInputSchema, DeepResearchOutputSchema
from akd.structures import SearchResultItem
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
    LinkRelevancyAssessorInputSchema,
)
from akd.tools.scrapers import (
    SimpleWebScraper,
    SimplePDFScraper,
    ScraperToolInputSchema,
)
from .base_search import (
    AgenticSearchTool,
    QueryFocusStrategy,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from .searxng_search import SearxNGSearchTool
from .semantic_scholar_search import SemanticScholarSearchTool, SemanticScholarSearchToolInputSchema

class ControlledAgenticLitSearchToolConfig(SearchToolConfig):
    """
    Configuration for the ControlledAgenticLitSearchTool.
    This tool uses multi-rubric analysis and agentic decision-making
    to perform intelligent iterative literature searches.
    """

    min_positive_rubrics: int = Field(
        default=3,
        description="Minimum number of positive rubrics needed to stop searching (out of 6).",
    )
    max_iteration: int = Field(
        default=5,
        description="Maximum number of iterations to perform.",
    )
    max_results_per_iteration: int = Field(
        default=10,
        description="Maximum number of results to return per iteration.",
    )
    use_followup_after_iteration: int = Field(
        default=1,
        description="Use follow-up query agent after this many iterations.",
    )
    rubric_improvement_threshold: int = Field(
        default=2,
        description="Stop if no rubric improvement for this many iterations.",
    )
    # Dynamic stopping thresholds
    early_stop_result_progress: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Minimum result progress (0.5-1.0) to allow early stopping with excellent quality.",
    )
    early_stop_quality_score: float = Field(
        default=0.67,
        ge=0.5,
        le=1.0,
        description="Minimum quality score (0.5-1.0) to allow early stopping with sufficient results.",
    )
    stagnation_result_progress: float = Field(
        default=0.6,
        ge=0.5,
        le=1.0,
        description="Minimum result progress (0.5-1.0) to allow stopping due to stagnation.",
    )
    stagnation_quality_score: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum quality score (0.5-1.0) to allow stopping due to stagnation.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )
    
    # Link relevancy assessment
    enable_per_link_assessment: bool = Field(
        default=False,  # Disabled by default to maintain backward compatibility
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


class ControlledAgenticLitSearchTool(AgenticSearchTool):
    """
    Tool for performing controlled agentic literature searches
    using multi-rubric analysis and agentic decision-making.
    This tool iteratively refines search queries based on rubric assessments
    and dynamically decides when to stop searching based on quality and quantity of results.

    Note:
        - It's not stateless. Meaning: we track the history of rubrics
          and decisions made during the search process.
    """

    config_schema = ControlledAgenticLitSearchToolConfig

    class _RubricAnalysis(BaseModel):
        """Analysis of multi-rubric assessment for agentic decision making."""

        topic_alignment_positive: bool = Field(default=False)
        content_depth_positive: bool = Field(default=False)
        recency_relevance_positive: bool = Field(default=False)
        methodological_relevance_positive: bool = Field(default=False)
        evidence_quality_positive: bool = Field(default=False)
        scope_relevance_positive: bool = Field(default=False)

        positive_rubric_count: int = Field(default=0)
        weak_rubrics: List[str] = Field(default_factory=list)
        strong_rubrics: List[str] = Field(default_factory=list)

        overall_assessment: str = Field(default="")
        reasoning_steps: List[str] = Field(default_factory=list)

    class _StoppingCriteria(BaseModel):
        stop_now: bool = Field(default=False)
        reasoning_trace: str = Field(default="")
        rubric_analysis: Optional["ControlledAgenticLitSearchTool._RubricAnalysis"] = (
            Field(
                default=None,
            )
        )
        recommended_query_focus: List[str] = Field(default_factory=list)

    def __init__(
        self,
        config: ControlledAgenticLitSearchToolConfig | None = None,
        search_tool: SearchTool | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config, debug=debug)
        self.search_tool = search_tool or SearxNGSearchTool()
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()
        self.query_agent = query_agent or QueryAgent()
        self.followup_query_agent = followup_query_agent or FollowUpQueryAgent()

        # Track rubric patterns for agentic learning
        self.rubric_history = []
        
        # Initialize link relevancy assessor if enabled
        if self.enable_per_link_assessment:
            assessor_config = LinkRelevancyAssessorConfig(
                min_relevancy_score=self.min_relevancy_score,
                full_content_threshold=self.full_content_threshold,
                debug=debug,
            )
            self.link_relevancy_assessor = LinkRelevancyAssessor(
                config=assessor_config,
                relevancy_agent=self.relevancy_agent,
                debug=debug,
            )
        else:
            self.link_relevancy_assessor = None

    def _deduplicate_results(
        self,
        new_results: List[SearchResultItem],
        existing_results: List[SearchResultItem],
    ) -> List[SearchResultItem]:
        """Remove duplicate results based on URL."""
        existing_urls = {r.url for r in existing_results}
        return [r for r in new_results if r.url not in existing_urls]

    def _accumulate_content(self, results: List[SearchResultItem]) -> str:
        content = ""
        for result in results:
            content += f"\nTitle: {result.title}\nContent: {result.content}\n"
        return content.strip()

    def _analyze_rubrics(
        self,
        rubric_output: MultiRubricRelevancyOutputSchema,
    ) -> "_RubricAnalysis":
        """Analyze multi-rubric output to determine positive/negative assessments."""
        analysis = self._RubricAnalysis()

        # Determine positive assessments using enum values directly
        analysis.topic_alignment_positive = (
            rubric_output.topic_alignment == TopicAlignmentLabel.ALIGNED
        )
        analysis.content_depth_positive = (
            rubric_output.content_depth == ContentDepthLabel.COMPREHENSIVE
        )
        analysis.recency_relevance_positive = (
            rubric_output.recency_relevance == RecencyRelevanceLabel.CURRENT
        )
        analysis.methodological_relevance_positive = (
            rubric_output.methodological_relevance
            == MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND
        )
        analysis.evidence_quality_positive = (
            rubric_output.evidence_quality == EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE
        )
        analysis.scope_relevance_positive = (
            rubric_output.scope_relevance == ScopeRelevanceLabel.IN_SCOPE
        )

        # Count positive rubrics
        positive_flags = [
            analysis.topic_alignment_positive,
            analysis.content_depth_positive,
            analysis.recency_relevance_positive,
            analysis.methodological_relevance_positive,
            analysis.evidence_quality_positive,
            analysis.scope_relevance_positive,
        ]
        analysis.positive_rubric_count = sum(positive_flags)

        # Identify weak and strong rubrics
        rubric_mapping = {
            "topic_alignment": analysis.topic_alignment_positive,
            "content_depth": analysis.content_depth_positive,
            "recency_relevance": analysis.recency_relevance_positive,
            "methodological_relevance": analysis.methodological_relevance_positive,
            "evidence_quality": analysis.evidence_quality_positive,
            "scope_relevance": analysis.scope_relevance_positive,
        }

        for rubric_name, is_positive in rubric_mapping.items():
            if is_positive:
                analysis.strong_rubrics.append(rubric_name)
            else:
                analysis.weak_rubrics.append(rubric_name)

        # Overall assessment
        if analysis.positive_rubric_count >= 4:
            analysis.overall_assessment = "strong"
        elif analysis.positive_rubric_count >= 2:
            analysis.overall_assessment = "moderate"
        else:
            analysis.overall_assessment = "weak"

        # Copy reasoning steps
        analysis.reasoning_steps = rubric_output.reasoning_steps

        return analysis

    def _make_agentic_stopping_decision(
        self,
        rubric_analysis: "_RubricAnalysis",
        iteration: int,
        current_result_count: int,
        desired_max_results: int,
        query: str,
    ) -> tuple[bool, str]:
        """Dynamic agentic stopping decision balancing quality, quantity, and adaptivity."""

        # Calculate progress metrics for dynamic decision making
        result_progress = (
            current_result_count / desired_max_results if desired_max_results > 0 else 0
        )
        quality_score = rubric_analysis.positive_rubric_count / 6

        # Dynamic minimum results threshold (adaptive based on quality)
        min_results_threshold = max(
            desired_max_results * 0.3,  # At least 30% of requested
            min(5, desired_max_results),  # But at least 5 or total requested if less
        )

        # Never stop if we have too few results
        if current_result_count < min_results_threshold:
            return False, (
                f"CONTINUE: Need more results ({current_result_count}/{min_results_threshold:.0f} minimum)"
            )

        # Dynamic quality threshold - lower if we have many results, higher if few
        quality_threshold = self.min_positive_rubrics
        if result_progress > 0.8:  # If we have most requested results
            quality_threshold = max(
                2,
                self.min_positive_rubrics - 1,
            )  # Lower quality bar
        elif result_progress < 0.5:  # If we have few results
            quality_threshold = min(
                5,
                self.min_positive_rubrics + 1,
            )  # Higher quality bar

        # Adaptive stopping based on quality + quantity balance
        if rubric_analysis.positive_rubric_count >= quality_threshold:
            # Good quality achieved
            if current_result_count >= desired_max_results:
                return (
                    True,
                    f"STOP: Target reached ({current_result_count}/{desired_max_results}) + quality good ({rubric_analysis.positive_rubric_count}/6)",
                )
            elif (
                result_progress >= self.early_stop_result_progress
                and quality_score >= self.early_stop_quality_score
            ):  # Configurable early stopping thresholds
                return (
                    True,
                    f"STOP: Sufficient results ({current_result_count}/{desired_max_results}, {result_progress:.1%}) + excellent quality ({rubric_analysis.positive_rubric_count}/6, {quality_score:.1%})",
                )

        # Force stop if we've exceeded target significantly (search overflow protection)
        if current_result_count >= desired_max_results * 1.5:
            return (
                True,
                f"STOP: Exceeded target ({current_result_count}/{desired_max_results}) - preventing overflow",
            )

        # Critical rubrics check - but balanced with results progress
        critical_rubrics = {"topic_alignment", "evidence_quality"}
        weak_critical = critical_rubrics.intersection(set(rubric_analysis.weak_rubrics))

        if (
            weak_critical and result_progress < 0.8
        ):  # Only block if we don't have most results
            return False, (
                f"CONTINUE: Critical rubrics weak ({weak_critical}) + need more results ({current_result_count}/{desired_max_results})"
            )

        # Adaptive stagnation check - more lenient if we need more results
        if len(self.rubric_history) >= self.rubric_improvement_threshold:
            recent_scores = [
                h["analysis"].positive_rubric_count
                for h in self.rubric_history[-self.rubric_improvement_threshold :]
            ]
            if all(
                score <= rubric_analysis.positive_rubric_count
                for score in recent_scores
            ):
                # Only stop for stagnation if we have BOTH reasonable quantity AND excellent quality
                # Never stop for stagnation if we have less than 50% of requested results
                if (
                    result_progress >= self.stagnation_result_progress
                    and quality_score >= self.stagnation_quality_score
                ):
                    return True, (
                        f"STOP: No improvement in {self.rubric_improvement_threshold} iterations + sufficient results ({current_result_count}/{desired_max_results}, {result_progress:.1%}) + excellent quality ({quality_score:.1%})"
                    )
                elif result_progress < 0.5:
                    # Force continue if we don't have enough results yet
                    return False, (
                        f"CONTINUE: Stagnation detected but insufficient results ({current_result_count}/{desired_max_results}) - need at least 50%"
                    )

        # Continue searching - provide specific guidance
        if result_progress < 0.5:
            return (
                False,
                f"CONTINUE: Need more results ({current_result_count}/{desired_max_results}) + improving quality ({rubric_analysis.positive_rubric_count}/6)",
            )
        else:
            return (
                False,
                f"CONTINUE: Refining quality ({rubric_analysis.positive_rubric_count}/6) with {current_result_count}/{desired_max_results} results",
            )

    def _generate_query_focus_recommendations(
        self,
        rubric_analysis: "_RubricAnalysis",
    ) -> List[str]:
        """Generate query focus recommendations based on weak rubrics."""
        # Mapping from weak rubrics to query focus strategies
        rubric_to_strategy = {
            "topic_alignment": QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY,
            "content_depth": QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS,
            "evidence_quality": QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES,
            "methodological_relevance": QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS,
            "recency_relevance": QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS,
            "scope_relevance": QueryFocusStrategy.ADJUST_QUERY_SCOPE,
        }

        recommendations = []
        for weak_rubric in rubric_analysis.weak_rubrics:
            if strategy := rubric_to_strategy.get(weak_rubric):
                recommendations.append(strategy.value)

        return recommendations

    async def _should_stop(
        self,
        iteration: int,
        query: str,
        all_results: list,
        current_results: list,
        max_results: int,
    ) -> "ControlledAgenticLitSearchTool._StoppingCriteria":
        criteria = self._StoppingCriteria(
            stop_now=False,
            reasoning_trace=f"Iteration {iteration}/{self.max_iteration}",
        )

        # Basic stopping conditions
        if iteration >= self.max_iteration:
            criteria.stop_now = True
            criteria.reasoning_trace = (
                f"Max iterations reached ({iteration}/{self.max_iteration})"
            )
            return criteria

        if iteration > 0 and not current_results:
            criteria.stop_now = True
            criteria.reasoning_trace = f"No new results found in iteration {iteration}"
            return criteria

        # Multi-rubric analysis for agentic decision making
        if (context := self._accumulate_content(current_results)) and iteration > 0:
            try:
                # Get multi-rubric assessment
                rubric_result = await self.relevancy_agent.arun(
                    MultiRubricRelevancyInputSchema(content=context, query=query),
                )

                if self.debug:
                    logger.debug(
                        f"Multi-rubric assessment for iteration {iteration}: {rubric_result}",
                    )

                # Analyze rubrics for decision making
                rubric_analysis = self._analyze_rubrics(rubric_result)
                criteria.rubric_analysis = rubric_analysis

                # Store rubric history for meta-learning (moved to _learn_from_iteration)

                # Agentic stopping decision based on multi-rubric analysis
                stop_decision, reasoning = self._make_agentic_stopping_decision(
                    rubric_analysis,
                    iteration,
                    len(all_results),
                    max_results,
                    query,
                )

                criteria.stop_now = stop_decision
                criteria.reasoning_trace = reasoning

                # Generate query focus recommendations for next iteration
                if not stop_decision:
                    criteria.recommended_query_focus = (
                        self._generate_query_focus_recommendations(
                            rubric_analysis,
                        )
                    )

            except Exception as e:
                logger.warning(f"Error in multi-rubric analysis: {str(e)}")
                # Fallback to continue searching if analysis fails
                criteria.stop_now = False
                criteria.reasoning_trace = (
                    "Multi-rubric analysis failed, continuing search"
                )

        return criteria

    async def _generate_queries(
        self,
        queries: List[str],
        iteration: int,
        num_queries: int = 3,
        results: Optional[List[SearchResultItem]] = None,
        accumulated_content: str = "",
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate queries using either initial query agent or follow-up query agent
        based on the iteration number, with adaptive focus based on rubric analysis.
        """
        if iteration <= self.use_followup_after_iteration:
            # Use initial query generation for early iterations
            return await self._generate_initial_queries(
                queries=queries,
                iteration=iteration,
                num_queries=num_queries,
                results=results,
                rubric_focus=rubric_focus,
            )
        else:
            # Use follow-up query generation for later iterations
            if accumulated_content:
                logger.debug(
                    f"Switching to follow-up query generation at iteration {iteration}",
                )
                return await self._generate_followup_queries(
                    original_queries=queries,
                    accumulated_content=accumulated_content,
                    num_queries=num_queries,
                    rubric_focus=rubric_focus,
                )
            else:
                # Fallback to initial query generation if no content available
                return await self._generate_initial_queries(
                    queries=queries,
                    iteration=iteration,
                    num_queries=num_queries,
                    results=results,
                    rubric_focus=rubric_focus,
                )

    async def _generate_followup_queries(
        self,
        original_queries: List[str],
        accumulated_content: str,
        num_queries: int = 3,
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate follow-up queries using the followup_query_agent with rubric-based adaptations."""
        try:
            # Enhance content with rubric-based guidance
            enhanced_content = accumulated_content
            if rubric_focus:
                focus_guidance = self._create_rubric_focus_guidance(rubric_focus)
                enhanced_content += f"\n\nFOCUS AREAS NEEDED: {focus_guidance}"

            followup_input = FollowUpQueryAgentInputSchema(
                original_queries=original_queries,
                content=enhanced_content,
                num_queries=num_queries,
            )

            followup_result: FollowUpQueryAgentOutputSchema = (
                await self.followup_query_agent.arun(
                    followup_input,
                )
            )

            if self.debug:
                logger.debug(
                    f"Follow-up reasoning: {followup_result.reasoning}",
                )
                logger.debug(
                    f"Identified gaps: {followup_result.original_query_gaps}",
                )
                if rubric_focus:
                    logger.debug(f"Applied rubric focus: {rubric_focus}")

            # Apply rubric-based query modifications
            if rubric_focus:
                adapted_queries = self._adapt_queries_for_rubrics(
                    followup_result.followup_queries,
                    rubric_focus,
                )
                return adapted_queries

            return followup_result.followup_queries

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning(
                f"Error generating follow-up queries. Error => {str(e)}",
            )
            return original_queries  # Fallback to original queries
        return original_queries

    async def _generate_initial_queries(
        self,
        queries: List[str],
        iteration: int,
        num_queries: int = 3,
        results: Optional[List[SearchResultItem]] = None,
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate initial queries using the query_agent with rubric-based adaptations."""
        context = ""
        if results:
            titles = [r.title for r in results]
            context = f"Previous searches found: {', '.join(titles)}"

        query_instruction = f"""
        Iteration {iteration} queries : {queries}
        Context/results so far: {context}
        """.strip()

        # Add rubric-based focus guidance
        if rubric_focus:
            focus_guidance = self._create_rubric_focus_guidance(rubric_focus)
            query_instruction += f"\n\nFOCUS AREAS NEEDED: {focus_guidance}"

        res = QueryAgentOutputSchema(queries=queries)
        try:
            res = await self.query_agent.arun(
                QueryAgentInputSchema(
                    num_queries=num_queries,
                    query=query_instruction,
                ),
            )

            # Apply rubric-based query modifications
            if rubric_focus:
                adapted_queries = self._adapt_queries_for_rubrics(
                    res.queries,
                    rubric_focus,
                )
                res.queries = adapted_queries

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning(
                f"Error generating initial queries. Error => {str(e)}",
            )
        return res.queries

    def _create_rubric_focus_guidance(self, rubric_focus: List[str]) -> str:
        """Create human-readable guidance for rubric focus areas."""
        guidance_map = {
            QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value: "Make queries more specific to the target topic and domain",
            QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value: "Look for systematic reviews, meta-analyses, and comprehensive surveys",
            QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value: "Focus on high-impact peer-reviewed journals and quality publications",
            QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value: "Find papers with strong methodological approaches and validation",
            QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value: "Prioritize recent publications (last 2-3 years)",
            QueryFocusStrategy.ADJUST_QUERY_SCOPE.value: "Refine the scope to match the research question boundaries",
        }

        guidance_list = [guidance_map.get(focus, focus) for focus in rubric_focus]
        return "; ".join(guidance_list)

    def _prioritize_rubric_focus(self, rubric_focus: List[str]) -> List[str]:
        """Prioritize rubric focus areas to avoid query over-complexity."""
        if not rubric_focus:
            return []

        # Priority order for rubric fixes (most critical first)
        priority_order = [
            QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value,  # Most important - improves relevance
            QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value,  # High impact - improves quality
            QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value,  # Good for depth
            QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value,  # Simple but effective
            QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value,  # Specific improvement
            QueryFocusStrategy.ADJUST_QUERY_SCOPE.value,  # General fallback
        ]

        # Return top 2-3 priorities to avoid complexity
        prioritized = []
        for priority in priority_order:
            if priority in rubric_focus:
                prioritized.append(priority)
                if len(prioritized) >= 2:  # Limit to 2 adaptations per query
                    break

        return prioritized

    def _adapt_queries_for_rubrics(
        self,
        queries: List[str],
        rubric_focus: List[str],
    ) -> List[str]:
        """Apply rubric-specific adaptations to queries with smart prioritization."""
        # Prioritize focus areas to avoid over-complexity
        prioritized_focus = self._prioritize_rubric_focus(rubric_focus)

        if self.debug:
            logger.debug(
                f"Prioritized rubric focus (from {len(rubric_focus)} to {len(prioritized_focus)}): {prioritized_focus}",
            )

        adapted_queries = []
        for i, query in enumerate(queries):
            # Apply different adaptations to different queries for variety
            focus_for_this_query = (
                prioritized_focus[i % len(prioritized_focus)]
                if prioritized_focus
                else None
            )

            if focus_for_this_query:
                adapted_query = self._apply_single_focus_adaptation(
                    query,
                    focus_for_this_query,
                )
                adapted_queries.append(adapted_query)
            else:
                adapted_queries.append(query)  # Keep original if no focus

        # Always include at least one original query as fallback
        if queries[0] not in adapted_queries:
            adapted_queries.append(queries[0])

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in adapted_queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)

        return unique_queries

    def _apply_single_focus_adaptation(self, query: str, focus: str) -> str:
        """Apply a single, clean adaptation based on rubric focus."""
        if focus == QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value:
            # Make query more specific without complex syntax
            return f"{query} specific detailed"

        elif focus == QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value:
            return f"{query} review OR survey OR meta-analysis"

        elif focus == QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value:
            return f"{query} journal peer-reviewed"

        elif focus == QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value:
            return f"{query} methodology approach"

        elif focus == QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value:
            return f"{query} 2020..2025"

        elif focus == QueryFocusStrategy.ADJUST_QUERY_SCOPE.value:
            return f'"{query}" specific'

        return query

    def _learn_from_iteration(
        self,
        iteration: int,
        queries: List[str],
        rubric_analysis: "_RubricAnalysis",
    ):
        """Simple learning: just track rubric history for trend analysis."""
        self.rubric_history.append(
            {
                "iteration": iteration,
                "analysis": rubric_analysis,
                "query": " AND ".join(queries),
            },
        )

    def _calculate_dynamic_batch_size(self, iteration: int, previous_criteria) -> int:
        """Calculate adaptive batch size based on rubric performance."""
        base_size = self.max_results_per_iteration

        # First iteration uses base size
        if iteration <= 1:
            return base_size

        # If we have rubric analysis from previous iteration
        if (
            hasattr(previous_criteria, "rubric_analysis")
            and previous_criteria.rubric_analysis
        ):
            rubric_count = previous_criteria.rubric_analysis.positive_rubric_count

            # If doing very poorly (0-1 positive rubrics), search more aggressively
            if rubric_count <= 1:
                return min(base_size * 2, 20)  # Double the search, cap at 20

            # If doing poorly (2 positive rubrics), search slightly more
            elif rubric_count == 2:
                return int(base_size * 1.5)

            # If doing well (4+ positive rubrics), can be more conservative
            elif rubric_count >= 4:
                return max(base_size // 2, 5)  # Half the search, minimum 5

        return base_size

    async def _arun(
        self,
        params: SearchToolInputSchema,
        **kwargs: Any,
    ) -> SearchToolOutputSchema:
        desired_max_results = params.max_results

        iteration = 0
        all_results = []
        current_results = []
        content_so_far = ""

        while not (
            criteria := await self._should_stop(
                iteration=iteration,
                all_results=all_results,
                current_results=current_results,
                max_results=desired_max_results,
                query=" AND ".join(params.queries),
            )
        ).stop_now:
            logger.debug(f"Stopping Criteria :: {criteria}")
            iteration += 1

            remaining_needed = desired_max_results - len(all_results)
            # Dynamic batch sizing: adjust based on previous rubric performance
            dynamic_batch_size = self._calculate_dynamic_batch_size(iteration, criteria)
            search_limit = min(remaining_needed, dynamic_batch_size)

            current_queries = params.queries
            if iteration > 0:
                # Get rubric focus from previous iteration's stopping criteria
                rubric_focus = None
                if (
                    hasattr(criteria, "recommended_query_focus")
                    and criteria.recommended_query_focus
                ):
                    rubric_focus = criteria.recommended_query_focus

                logger.debug(f"Rubric focus for iteration {iteration}: {rubric_focus}")

                current_queries = await self._generate_queries(
                    iteration=iteration,
                    num_queries=3,
                    queries=params.queries,
                    results=all_results,
                    accumulated_content=content_so_far,  # Pass accumulated content
                    rubric_focus=rubric_focus,  # Pass rubric focus for adaptive queries
                )

            if self.debug:
                logger.info(f"🔄 ITERATION {iteration} - REFORMULATED QUERIES:")
                for i, query in enumerate(current_queries, 1):
                    is_original = query in params.queries
                    marker = "🎯" if is_original else "🔄"
                    logger.info(f"  {i}. {marker} '{query}'")
                if rubric_focus:
                    logger.info(f"📋 Applied rubric focus: {rubric_focus}")
            
            logger.debug(
                f"Generated queries (iteration {iteration}): {current_queries}",
            )
            search_input = SearchToolInputSchema(
                queries=current_queries,
                max_results=search_limit,
                category=params.category,
            )
            search_result = await self.search_tool.arun(
                self.search_tool.input_schema(**search_input.model_dump()),
            )

            current_results = self._deduplicate_results(
                new_results=search_result.results,
                existing_results=all_results,
            )
            
            # Apply per-link relevancy assessment if enabled
            if self.link_relevancy_assessor and current_results:
                if self.debug:
                    logger.debug(f"Assessing relevancy for {len(current_results)} new search results")
                
                # Determine if current queries are reformulated
                is_reformulated = iteration > 1 and current_queries != params.queries
                reformulated_query = current_queries[0] if is_reformulated and current_queries else None
                
                if self.debug and reformulated_query:
                    logger.debug(f"Using reformulated query for comparison: '{reformulated_query}' vs original: '{' AND '.join(params.queries)}'")
                
                assessor_input = LinkRelevancyAssessorInputSchema(
                    search_results=current_results,
                    original_query=" AND ".join(params.queries),
                    reformulated_query=reformulated_query,
                    domain_context=f"Agentic search iteration {iteration} with rubric focus" if rubric_focus else None,
                )
                
                try:
                    assessment_output = await self.link_relevancy_assessor.arun(assessor_input)
                    
                    if self.debug:
                        logger.debug(f"Relevancy assessment summary: {assessment_output.assessment_summary}")
                    
                    # Use filtered results based on relevancy
                    current_results = assessment_output.filtered_results
                    
                except Exception as e:
                    logger.warning(f"Error in relevancy assessment: {e}")
                    # Continue with original results if assessment fails

            # Fallback mechanism: If adapted queries return no results, try original queries
            if (
                not current_results
                and iteration > 1
                and current_queries != params.queries
            ):
                if self.debug:
                    logger.debug(
                        f"No results from adapted queries, falling back to original queries: {params.queries}",
                    )

                fallback_input = SearchToolInputSchema(
                    queries=params.queries,  # Use original queries
                    max_results=search_limit,
                    category=params.category,
                )

                fallback_result = await self.search_tool.arun(
                    self.search_tool.input_schema(**fallback_input.model_dump()),
                )

                current_results = self._deduplicate_results(
                    new_results=fallback_result.results,
                    existing_results=all_results,
                )

                if current_results and self.debug:
                    logger.debug(
                        f"Fallback successful: found {len(current_results)} results",
                    )

            all_results.extend(current_results)

            # Update accumulated content after each iteration
            new_content = self._accumulate_content(current_results)
            if new_content:
                content_so_far += "\n" + new_content if content_so_far else new_content

            # Learn from this iteration if we have rubric analysis
            if hasattr(criteria, "rubric_analysis") and criteria.rubric_analysis:
                self._learn_from_iteration(
                    iteration,
                    current_queries,
                    criteria.rubric_analysis,
                )

            if self.debug:
                logger.debug(
                    f"Content accumulated so far (chars): {len(content_so_far)}",
                )
                logger.debug(
                    f"Current iteration results: {len(current_results)}",
                )

        logger.debug(f"Final Stopping Criteria :: {criteria}")
        return SearchToolOutputSchema(results=all_results, category=params.category)


class DeepLitSearchToolConfig(SearchToolConfig):
    """
    Configuration for the DeepLitSearchTool that implements multi-agent deep research.
    """

    # Model configurations
    use_mini_model: bool = Field(
        default=True,
        description="Use o3-mini-deep-research (faster) vs full o3 model",
    )

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

    debug: bool = Field(
        default=False,
        description="Enable debug logging",
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


class DeepLitSearchTool(AgenticSearchTool):
    """
    Advanced literature search tool implementing multi-agent deep research pattern.

    This tool orchestrates multiple agents to:
    1. Triage and clarify research queries
    2. Build detailed research instructions
    3. Perform iterative deep research with quality checks
    4. Produce comprehensive, well-structured research reports

    The implementation follows the OpenAI Deep Research pattern but is adapted
    to work within the akd framework using existing tools and agents.
    """

    config_schema = DeepLitSearchToolConfig

    def __init__(
        self,
        config: DeepLitSearchToolConfig | None = None,
        search_tool: SearchTool | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        triage_agent=None,
        clarifying_agent=None,
        instruction_agent=None,
        research_agent=None,
        debug: bool = False,
    ) -> None:
        """Initialize the DeepLitSearchTool with agents and configuration."""
        super().__init__(config=config, debug=debug)

        # Initialize search tools
        self.search_tool = search_tool or SearxNGSearchTool()
        self.semantic_scholar_tool = (
            SemanticScholarSearchTool() if self.use_semantic_scholar else None
        )

        # Initialize relevancy agent
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()
        
        # Initialize link relevancy assessor if enabled
        if self.enable_per_link_assessment:
            assessor_config = LinkRelevancyAssessorConfig(
                min_relevancy_score=self.min_relevancy_score,
                full_content_threshold=self.full_content_threshold,
                debug=debug,
            )
            self.link_relevancy_assessor = LinkRelevancyAssessor(
                config=assessor_config,
                relevancy_agent=self.relevancy_agent,
                debug=debug,
            )
        else:
            self.link_relevancy_assessor = None
        
        # Initialize scrapers for full content fetching
        if self.enable_full_content_scraping:
            self.web_scraper = SimpleWebScraper(debug=debug)
            self.pdf_scraper = SimplePDFScraper(debug=debug)
        else:
            self.web_scraper = None
            self.pdf_scraper = None

        # Import agents here to avoid circular imports
        from akd.agents.deep_research import (
            ClarifyingAgent,
            DeepResearchAgent,
            InstructionBuilderAgent,
            TriageAgent,
        )

        # Initialize deep research agents
        self.triage_agent = triage_agent or TriageAgent(debug=debug)
        self.clarifying_agent = clarifying_agent or ClarifyingAgent(debug=debug)
        self.instruction_agent = instruction_agent or InstructionBuilderAgent(
            debug=debug
        )
        self.research_agent = research_agent or DeepResearchAgent(debug=debug)

        # Track research state
        self.research_history = []
        self.clarification_history = []

    async def _handle_clarification(
        self,
        query: str,
        mock_answers: Optional[Dict[str, str]] = None,
    ) -> tuple[str, List[str]]:
        """
        Handle the clarification process with the user.

        Args:
            query: The original research query
            mock_answers: Optional mock answers for testing

        Returns:
            Tuple of (enriched_query, clarifications)
        """
        from akd.agents.schemas import ClarifyingAgentInputSchema

        clarifying_input = ClarifyingAgentInputSchema(query=query)
        clarifying_output = await self.clarifying_agent.arun(clarifying_input)

        if self.debug:
            logger.debug(f"Clarifying questions: {clarifying_output.questions}")

        # In a real implementation, this would interact with the user
        # For now, we'll use mock answers or default responses
        clarifications = []
        for question in clarifying_output.questions:
            answer = (mock_answers or {}).get(question, "No specific preference")
            clarifications.append(f"{question}: {answer}")

        # Create enriched query with clarifications
        enriched_query = f"{query}\n\nAdditional context:\n" + "\n".join(clarifications)

        return enriched_query, clarifications

    async def _build_research_instructions(
        self,
        query: str,
        clarifications: Optional[List[str]] = None,
    ) -> str:
        """
        Build detailed research instructions from query and clarifications.

        Args:
            query: The research query (possibly enriched)
            clarifications: List of clarification responses

        Returns:
            Detailed research instructions
        """
        from akd.agents.schemas import InstructionBuilderInputSchema

        instruction_input = InstructionBuilderInputSchema(
            query=query,
            clarifications=clarifications,
        )

        instruction_output = await self.instruction_agent.arun(instruction_input)

        if self.debug:
            logger.debug(
                f"Research instructions: {instruction_output.research_instructions}"
            )
            logger.debug(f"Focus areas: {instruction_output.focus_areas}")

        return instruction_output.research_instructions

    async def _perform_deep_research(
        self,
        instructions: str,
        original_query: str,
    ) -> "DeepResearchOutputSchema":
        """
        Perform the actual deep research using iterative search and synthesis.

        This method coordinates search tools, relevancy checking, and the
        research agent to produce comprehensive results.

        Args:
            instructions: Detailed research instructions
            original_query: The original user query

        Returns:
            Deep research output with report and metadata
        """

        # Initialize research tracking
        all_results = []
        iterations = 0
        quality_scores = []
        research_trace = []

        # Initial search queries from instructions
        initial_queries = await self._generate_initial_queries(instructions)

        while iterations < self.max_research_iterations:
            iterations += 1
            research_trace.append(
                f"Iteration {iterations}: Searching with queries: {initial_queries}"
            )

            # Perform searches
            search_results = await self._execute_searches(
                initial_queries, 
                original_query, 
                is_reformulated=(iterations > 1)  # Queries are reformulated after first iteration
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
                    f"quality score: {quality_score:.2f}"
                )

                # Check if we've reached quality threshold
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality >= self.quality_threshold and len(all_results) >= 10:
                    research_trace.append(
                        f"Stopping: Quality threshold reached ({avg_quality:.2f})"
                    )
                    break

            # Generate refined queries for next iteration
            if iterations < self.max_research_iterations:
                initial_queries = await self._generate_refined_queries(
                    initial_queries,
                    all_results,
                    instructions,
                )

        # Synthesize final research report
        research_input = DeepResearchInputSchema(
            research_instructions=instructions,
            original_query=original_query,
            max_iterations=iterations,
            quality_threshold=self.quality_threshold,
        )

        # For now, create a structured output
        # In a full implementation, this would use the research agent
        research_output = await self._synthesize_research(
            all_results,
            research_input,
            quality_scores,
            research_trace,
        )

        return research_output

    async def _generate_initial_queries(self, instructions: str) -> List[str]:
        """Generate initial search queries from research instructions."""
        from akd.agents.query import QueryAgent
        from akd.agents.schemas import QueryAgentInputSchema

        query_agent = QueryAgent()
        query_input = QueryAgentInputSchema(
            query=instructions,
            num_queries=5,  # More queries for comprehensive coverage
        )

        query_output = await query_agent.arun(query_input)
        
        if self.debug:
            logger.info(f"🧠 DeepLitSearchTool - INITIAL QUERIES GENERATED:")
            for i, query in enumerate(query_output.queries, 1):
                logger.info(f"  {i}. '{query}'")
        
        return query_output.queries

    async def _execute_searches(
        self, 
        queries: List[str], 
        original_query: str = None,
        is_reformulated: bool = False
    ) -> List[SearchResultItem]:
        """Execute searches using available search tools."""
        all_results = []

        # Search with primary tool
        search_input = SearchToolInputSchema(
            queries=queries,
            max_results=20,
            category="science",
        )

        # Convert to appropriate schema based on search tool type
        if hasattr(self.search_tool, 'input_schema') and self.search_tool.input_schema.__name__ == 'SemanticScholarSearchToolInputSchema':
            primary_input = SemanticScholarSearchToolInputSchema(
                queries=queries,
                max_results=20,
                category="science",
            )
        elif hasattr(self.search_tool, 'input_schema') and self.search_tool.input_schema.__name__ == 'SearxNGSearchToolInputSchema':
            from .searxng_search import SearxNGSearchToolInputSchema
            primary_input = SearxNGSearchToolInputSchema(
                queries=queries,
                max_results=20,
                category="science",
            )
        else:
            primary_input = search_input

        primary_results = await self.search_tool.arun(primary_input)
        all_results.extend(primary_results.results)

        # Search with Semantic Scholar if enabled
        if self.semantic_scholar_tool and self.use_semantic_scholar:
            # Convert to SemanticScholarSearchToolInputSchema
            ss_input = SemanticScholarSearchToolInputSchema(
                queries=queries,
                max_results=20,
                category="science",
            )
            ss_results = await self.semantic_scholar_tool.arun(ss_input)
            all_results.extend(ss_results.results)

        # Apply per-link relevancy assessment if enabled
        if self.link_relevancy_assessor and all_results:
            if self.debug:
                logger.debug(f"Assessing relevancy for {len(all_results)} search results")
            
            # Determine reformulated query if current queries differ from original
            # This enables original vs reformulated query comparison to determine
            # which query version produces better relevancy alignment for each source
            reformulated_query = None
            if is_reformulated and original_query:
                # Use the primary refined query as reformulated query
                reformulated_query = queries[0] if queries and queries[0] != original_query else None
                
                if self.debug and reformulated_query:
                    logger.debug(f"Using reformulated query for comparison: '{reformulated_query}' vs original: '{original_query}'")
            
            assessor_input = LinkRelevancyAssessorInputSchema(
                search_results=all_results,
                original_query=original_query or queries[0],  # Fallback to first query
                reformulated_query=reformulated_query,
                domain_context=f"Research iteration with {len(queries)} query variations" if len(queries) > 1 else None,
            )
            
            try:
                assessment_output = await self.link_relevancy_assessor.arun(assessor_input)
                
                if self.debug:
                    logger.debug(f"Relevancy assessment summary: {assessment_output.assessment_summary}")
                
                # Use filtered results based on relevancy
                return assessment_output.filtered_results
                
            except Exception as e:
                logger.warning(f"Error in relevancy assessment: {e}")
                # Fall back to original results if assessment fails
                return all_results
        
        # Fetch full content for high-relevancy results if enabled
        if self.web_scraper and all_results:
            all_results = await self._fetch_full_content_for_high_relevancy(all_results)
        
        return all_results

    async def _fetch_full_content_for_high_relevancy(
        self,
        results: List[SearchResultItem],
    ) -> List[SearchResultItem]:
        """Fetch full content for results marked as high-relevancy."""
        high_relevancy_results = [r for r in results if r.should_fetch_full_content]
        
        if not high_relevancy_results:
            return results
        
        if self.debug:
            logger.debug(f"Fetching full content for {len(high_relevancy_results)} high-relevancy results")
        
        for result in high_relevancy_results:
            try:
                # Try PDF first if available
                if result.pdf_url and self.pdf_scraper:
                    try:
                        pdf_input = ScraperToolInputSchema(url=str(result.pdf_url))
                        pdf_content = await self.pdf_scraper.arun(pdf_input)
                        if pdf_content.content and len(pdf_content.content) > 500:
                            result.content = pdf_content.content
                            if self.debug:
                                logger.debug(f"Fetched PDF content for {result.url} ({len(result.content)} chars)")
                            continue
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"PDF scraping failed for {result.pdf_url}: {e}")
                
                # Fall back to web scraping
                if self.web_scraper:
                    web_input = ScraperToolInputSchema(url=str(result.url))
                    web_content = await self.web_scraper.arun(web_input)
                    if web_content.content and len(web_content.content) > len(result.content or ""):
                        result.content = web_content.content
                        if self.debug:
                            logger.debug(f"Fetched web content for {result.url} ({len(result.content)} chars)")
                
            except Exception as e:
                if self.debug:
                    logger.debug(f"Content fetching failed for {result.url}: {e}")
                # Keep original content on failure
                continue
        
        return results

    def _deduplicate_results(
        self,
        new_results: List[SearchResultItem],
        existing_results: List[SearchResultItem],
    ) -> List[SearchResultItem]:
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
            ]
        )

        rubric_input = MultiRubricRelevancyInputSchema(
            content=content,
            query=query,
        )

        rubric_output = await self.relevancy_agent.arun(rubric_input)

        # Calculate quality score from rubrics
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
            ]
        )

        return positive_count / 6.0

    async def _generate_refined_queries(
        self,
        previous_queries: List[str],
        results: List[SearchResultItem],
        instructions: str,
    ) -> List[str]:
        """Generate refined queries based on current results."""
        from akd.agents.query import FollowUpQueryAgent
        from akd.agents.schemas import FollowUpQueryAgentInputSchema

        # Create content summary from results
        content = "\n\n".join(
            [
                f"Title: {r.title}\nSummary: {r.content[:200]}..."
                for r in results[-10:]  # Use recent results
            ]
        )

        followup_agent = FollowUpQueryAgent()
        # Enhance content with research instructions context
        enhanced_content = f"Research Instructions: {instructions}\n\nCurrent Results:\n{content}"
        
        followup_input = FollowUpQueryAgentInputSchema(
            original_queries=previous_queries,
            content=enhanced_content,
            num_queries=3,
        )

        followup_output = await followup_agent.arun(followup_input)
        
        if self.debug:
            logger.info(f"🔄 DeepLitSearchTool - REFINED QUERIES GENERATED:")
            for i, query in enumerate(followup_output.followup_queries, 1):
                is_original = query in previous_queries
                marker = "🎯" if is_original else "🔄"
                logger.info(f"  {i}. {marker} '{query}'")
        
        return followup_output.followup_queries

    async def _synthesize_research(
        self,
        results: List[SearchResultItem],
        research_input: DeepResearchInputSchema,
        quality_scores: List[float],
        research_trace: List[str],
    ) -> "DeepResearchOutputSchema":
        """Synthesize research results into a comprehensive report."""
        from akd.agents.schemas import DeepResearchOutputSchema

        # Group results by relevance score if available
        if results and results[0].relevancy_score is not None:
            # Sort by relevancy score (highest first)
            sorted_results = sorted(
                results,
                key=lambda r: r.relevancy_score or 0.0,
                reverse=True,
            )
            high_quality_results = sorted_results[:20]
            
            # Separate into quality tiers
            high_relevancy = [r for r in results if (r.relevancy_score or 0) >= 0.7]
            medium_relevancy = [r for r in results if 0.4 <= (r.relevancy_score or 0) < 0.7]
            low_relevancy = [r for r in results if (r.relevancy_score or 0) < 0.4]
        else:
            # Fall back to original ordering
            high_quality_results = results[:20]
            high_relevancy = []
            medium_relevancy = []
            low_relevancy = []

        # Extract key findings
        key_findings = []
        for result in high_quality_results[:10]:
            if result.content:
                # Extract first significant sentence as a finding
                sentences = result.content.split(". ")
                if sentences:
                    key_findings.append(sentences[0] + ".")

        # Calculate relevancy statistics
        relevancy_stats = {}
        if results and results[0].relevancy_score is not None:
            scores = [r.relevancy_score for r in results if r.relevancy_score is not None]
            relevancy_stats = {
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "high_relevancy_count": len(high_relevancy),
                "medium_relevancy_count": len(medium_relevancy),
                "low_relevancy_count": len(low_relevancy),
                "full_content_fetched": len([r for r in results if r.should_fetch_full_content]),
            }

        # Create research report structure
        report_sections = [
            "# Research Report",
            "\n## Executive Summary",
            f"This research on '{research_input.original_query}' analyzed {len(results)} sources "
            f"across {len(research_trace)} iterations.",
        ]
        
        # Add relevancy and query alignment summary if available
        if relevancy_stats:
            report_sections.extend([
                "\n**Quality Assessment:**",
                f"- Average relevancy score: {relevancy_stats['avg_score']:.2f}/1.0",
                f"- High relevancy sources: {relevancy_stats['high_relevancy_count']}",
                f"- Medium relevancy sources: {relevancy_stats['medium_relevancy_count']}",
                f"- Full content analyzed: {relevancy_stats['full_content_fetched']} sources",
            ])
            
            # Add query alignment summary
            query_alignment_results = [r for r in results if r.query_alignment_details]
            if query_alignment_results:
                reformulated_better = len([r for r in query_alignment_results 
                                         if r.query_alignment_details.get("best_query") == "reformulated"])
                original_better = len([r for r in query_alignment_results 
                                     if r.query_alignment_details.get("best_query") == "original"])
                
                report_sections.extend([
                    "\n**Query Alignment Analysis:**",
                    f"- Sources better aligned with original query: {original_better} 🎯",
                    f"- Sources better aligned with reformulated queries: {reformulated_better} 🔄",
                    f"- Query refinement effectiveness: {reformulated_better / len(query_alignment_results) * 100:.1f}%" if query_alignment_results else "",
                ])
        
        report_sections.append("\n## Key Findings")

        for i, finding in enumerate(key_findings[:5], 1):
            report_sections.append(f"{i}. {finding}")

        report_sections.extend(
            [
                "\n## Detailed Analysis",
                "Based on the comprehensive literature review, the following themes emerged:",
                "\n## Sources Consulted",
            ]
        )

        # Add top sources with relevancy information
        sources = []
        citations = []
        for result in high_quality_results:
            sources.append(str(result.url))
            
            # Add relevancy metadata to citations
            citation = {
                "title": result.title or "Untitled",
                "url": str(result.url),
                "excerpt": (result.content[:200] if result.content else "") + "...",
            }
            
            if result.relevancy_score is not None:
                citation["relevancy_score"] = result.relevancy_score
                citation["full_content_fetched"] = result.should_fetch_full_content
                
                # Add query alignment details if available
                if result.query_alignment_details:
                    citation["query_alignment"] = result.query_alignment_details
            
            citations.append(citation)
            
            # Format source entry with relevancy score
            relevancy_indicator = ""
            if result.relevancy_score is not None:
                score = result.relevancy_score
                if score >= 0.7:
                    relevancy_indicator = " 🟢"  # High relevancy
                elif score >= 0.4:
                    relevancy_indicator = " 🟡"  # Medium relevancy
                else:
                    relevancy_indicator = " 🔴"  # Low relevancy
                relevancy_indicator += f" ({score:.2f})"
            
            full_content_indicator = " 📄" if result.should_fetch_full_content else ""
            
            # Add query alignment indicator
            query_alignment_indicator = ""
            if result.query_alignment_details:
                if result.query_alignment_details.get("best_query") == "reformulated":
                    query_alignment_indicator = " 🔄"  # Better aligned with reformulated query
                elif result.query_alignment_details.get("best_query") == "original":
                    query_alignment_indicator = " 🎯"  # Better aligned with original query
            
            report_sections.append(
                f"- [{result.title}]({result.url}){relevancy_indicator}{full_content_indicator}{query_alignment_indicator}"
            )

        report_sections.extend(
            [
                "\n## Research Quality",
                f"Average quality score: {sum(quality_scores) / len(quality_scores):.2f}"
                if quality_scores
                else "No quality scores available",
                "\n## Limitations and Gaps",
                "- Limited to publicly available sources",
                "- Time constraints may have limited depth of analysis",
            ]
        )

        research_report = "\n".join(report_sections)

        # Create output
        output = DeepResearchOutputSchema(
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
            iterations_performed=len(research_trace),
            research_trace=research_trace,
        )

        return output

    async def _arun(
        self,
        params: SearchToolInputSchema,
        **kwargs: Any,
    ) -> SearchToolOutputSchema:
        """
        Run the DeepLitSearchTool with multi-agent orchestration.

        This implements the full deep research pipeline:
        1. Triage the query
        2. Clarify if needed
        3. Build research instructions
        4. Perform deep research
        5. Return structured results
        """
        from akd.agents.schemas import TriageAgentInputSchema

        original_query = " ".join(params.queries)  # Combine queries

        # Step 1: Triage the query
        triage_input = TriageAgentInputSchema(query=original_query)
        triage_output = await self.triage_agent.arun(triage_input)

        if self.debug:
            logger.debug(f"Triage decision: {triage_output.routing_decision}")
            logger.debug(f"Reasoning: {triage_output.reasoning}")

        # Step 2: Handle based on triage decision
        enriched_query = original_query
        clarifications = None

        if triage_output.needs_clarification and self.auto_clarify:
            enriched_query, clarifications = await self._handle_clarification(
                original_query,
                kwargs.get("mock_answers"),
            )

        # Step 3: Build research instructions
        instructions = await self._build_research_instructions(
            enriched_query,
            clarifications,
        )

        # Step 4: Perform deep research
        research_output = await self._perform_deep_research(
            instructions,
            original_query,
        )

        # Step 5: Convert research output to SearchToolOutputSchema
        # Extract search results from the research for compatibility
        results = []
        for citation in research_output.citations or []:
            # Create SearchResultItem and preserve relevancy metadata
            result_item = SearchResultItem(
                url=citation["url"],
                title=citation["title"] or "Untitled",  # Ensure title is never None
                content=citation["excerpt"],
                query=original_query or "Unknown query",  # Ensure query is never None
                category="science",
            )
            
            # Preserve relevancy information if available
            if "relevancy_score" in citation:
                result_item.relevancy_score = citation["relevancy_score"]
            if "full_content_fetched" in citation:
                result_item.should_fetch_full_content = citation["full_content_fetched"]
            if "query_alignment" in citation:
                result_item.query_alignment_details = citation["query_alignment"]
            
            results.append(result_item)

        # Add research report as a special result
        if results:
            results.insert(
                0,
                SearchResultItem(
                    url="deep-research://report",
                    title="Deep Research Report",
                    content=research_output.research_report,
                    query=original_query,
                    category="research",
                    extra={
                        "key_findings": research_output.key_findings,
                        "quality_score": research_output.evidence_quality_score,
                        "iterations": research_output.iterations_performed,
                    },
                ),
            )

        return SearchToolOutputSchema(
            results=results,
            category=params.category,
        )
