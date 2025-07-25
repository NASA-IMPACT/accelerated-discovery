from enum import Enum
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd.agents._base import BaseAgent, BaseAgentConfig
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
from akd.structures import SearchResultItem
from akd.tools.search import (
    SearchTool,
    SearchToolInputSchema,
    SearchToolOutputSchema,
    SearxNGSearchTool,
)


class QueryFocusStrategy(str, Enum):
    """Query adaptation strategies based on rubric analysis."""

    REFINE_TOPIC_SPECIFICITY = "refine_topic_specificity"
    SEARCH_COMPREHENSIVE_REVIEWS = "search_comprehensive_reviews"
    TARGET_PEER_REVIEWED_SOURCES = "target_peer_reviewed_sources"
    SEARCH_METHODOLOGICAL_PAPERS = "search_methodological_papers"
    ADD_RECENT_YEAR_FILTERS = "add_recent_year_filters"
    ADJUST_QUERY_SCOPE = "adjust_query_scope"


class SearchAgent(BaseAgent, SearchTool):
    """Base agent for performing literature searches using a search tool."""

    pass


class ControlledSearchAgentConfig(BaseAgentConfig):
    """
    Configuration for the ControlledSearchAgent.
    This agent uses multi-rubric analysis and agentic decision-making
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
    # Overall assessment thresholds
    strong_assessment_threshold: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Minimum positive rubrics (out of 6) for 'strong' assessment.",
    )
    moderate_assessment_threshold: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Minimum positive rubrics (out of 6) for 'moderate' assessment.",
    )
    # Progress thresholds for stopping decision
    min_result_percentage: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Minimum percentage of requested results before allowing stop.",
    )
    min_absolute_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Minimum absolute number of results before allowing stop.",
    )
    high_progress_threshold: float = Field(
        default=0.8,
        ge=0.6,
        le=1.0,
        description="Progress threshold for 'high progress' quality adjustments.",
    )
    low_progress_threshold: float = Field(
        default=0.5,
        ge=0.2,
        le=0.7,
        description="Progress threshold for 'low progress' quality adjustments.",
    )
    # Quality adjustments for stopping decision
    min_quality_when_high_progress: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Minimum quality threshold when high progress.",
    )
    max_quality_when_low_progress: int = Field(
        default=5,
        ge=3,
        le=6,
        description="Maximum quality threshold when low progress.",
    )
    # Safety limits for stopping decision
    overflow_multiplier: float = Field(
        default=1.5,
        ge=1.2,
        le=3.0,
        description="Stop when results exceed target by this multiplier.",
    )
    critical_rubrics_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Progress threshold for critical rubrics check.",
    )
    # Critical rubrics set
    critical_rubrics: set[str] = Field(
        default_factory=lambda: {"topic_alignment", "evidence_quality"},
        description="Rubrics considered critical for quality.",
    )


class ControlledSearchAgent(SearchAgent):
    """
    Agent for performing controlled agentic literature searches
    using multi-rubric analysis and agentic decision-making.
    This agent iteratively refines search queries based on rubric assessments
    and dynamically decides when to stop searching based on quality and quantity of results.

    Note:
        - It's not stateless. Meaning: we track the history of rubrics
          and decisions made during the search process.
    """

    config_schema = ControlledSearchAgentConfig

    # Total number of rubrics used in quality assessment
    _TOTAL_RUBRICS = 6

    class _RubricAnalysis(BaseModel):
        """Analysis of multi-rubric assessment for agentic decision making."""

        positive_rubric_count: int = Field(default=0)
        weak_rubrics: List[str] = Field(default_factory=list)
        strong_rubrics: List[str] = Field(default_factory=list)

        overall_assessment: str = Field(default="")
        reasoning_steps: List[str] = Field(default_factory=list)

    class _StoppingCriteria(BaseModel):
        stop_now: bool = Field(default=False)
        reasoning_trace: str = Field(default="")
        rubric_analysis: Optional["ControlledSearchAgent._RubricAnalysis"] = Field(
            default=None,
        )
        recommended_query_focus: List[str] = Field(default_factory=list)

    def __init__(
        self,
        config: ControlledSearchAgentConfig | None = None,
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

    async def get_response_async(
        self,
        response_model: type[SearchToolInputSchema] | None = None,
    ) -> SearchToolOutputSchema:
        """
        This agent doesn't use the standard LLM response pattern since it orchestrates
        multiple other agents and tools for agentic search behavior.
        The actual search logic is implemented in _arun.
        """
        raise NotImplementedError(
            "ControlledSearchAgent uses _arun for orchestration, not get_response_async",
        )

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

        # Check each rubric against its positive value
        rubric_checks = [
            (
                "topic_alignment",
                rubric_output.topic_alignment == TopicAlignmentLabel.ALIGNED,
            ),
            (
                "content_depth",
                rubric_output.content_depth == ContentDepthLabel.COMPREHENSIVE,
            ),
            (
                "recency_relevance",
                rubric_output.recency_relevance == RecencyRelevanceLabel.CURRENT,
            ),
            (
                "methodological_relevance",
                rubric_output.methodological_relevance
                == MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            ),
            (
                "evidence_quality",
                rubric_output.evidence_quality
                == EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            ),
            (
                "scope_relevance",
                rubric_output.scope_relevance == ScopeRelevanceLabel.IN_SCOPE,
            ),
        ]

        # Categorize rubrics
        for rubric_name, is_positive in rubric_checks:
            if is_positive:
                analysis.strong_rubrics.append(rubric_name)
            else:
                analysis.weak_rubrics.append(rubric_name)

        analysis.positive_rubric_count = len(analysis.strong_rubrics)

        # Overall assessment using configurable thresholds
        if analysis.positive_rubric_count >= self.config.strong_assessment_threshold:
            analysis.overall_assessment = "strong"
        elif (
            analysis.positive_rubric_count >= self.config.moderate_assessment_threshold
        ):
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
        quality_score = rubric_analysis.positive_rubric_count / self._TOTAL_RUBRICS

        # Dynamic minimum results threshold (adaptive based on quality)
        min_results_threshold = max(
            desired_max_results
            * self.config.min_result_percentage,  # At least configurable % of requested
            min(
                self.config.min_absolute_results,
                desired_max_results,
            ),  # But at least configurable minimum or total requested if less
        )

        # Never stop if we have too few results
        if current_result_count < min_results_threshold:
            return False, (
                f"CONTINUE: Need more results ({current_result_count}/{min_results_threshold:.0f} minimum)"
            )

        # Dynamic quality threshold - lower if we have many results, higher if few
        quality_threshold = self.min_positive_rubrics
        if (
            result_progress > self.config.high_progress_threshold
        ):  # If we have most requested results
            quality_threshold = max(
                self.config.min_quality_when_high_progress,
                self.min_positive_rubrics - 1,
            )  # Lower quality bar
        elif (
            result_progress < self.config.low_progress_threshold
        ):  # If we have few results
            quality_threshold = min(
                self.config.max_quality_when_low_progress,
                self.min_positive_rubrics + 1,
            )  # Higher quality bar

        # Adaptive stopping based on quality + quantity balance
        if rubric_analysis.positive_rubric_count >= quality_threshold:
            # Good quality achieved
            if current_result_count >= desired_max_results:
                return (
                    True,
                    f"STOP: Target reached ({current_result_count}/{desired_max_results}) + quality good ({rubric_analysis.positive_rubric_count}/{self._TOTAL_RUBRICS})",
                )
            elif (
                result_progress >= self.early_stop_result_progress
                and quality_score >= self.early_stop_quality_score
            ):  # Configurable early stopping thresholds
                return (
                    True,
                    f"STOP: Sufficient results ({current_result_count}/{desired_max_results}, {result_progress:.1%}) + excellent quality ({rubric_analysis.positive_rubric_count}/{self._TOTAL_RUBRICS}, {quality_score:.1%})",
                )

        # Force stop if we've exceeded target significantly (search overflow protection)
        if (
            current_result_count
            >= desired_max_results * self.config.overflow_multiplier
        ):
            return (
                True,
                f"STOP: Exceeded target ({current_result_count}/{desired_max_results}) - preventing overflow",
            )

        # Critical rubrics check - but balanced with results progress
        critical_rubrics = self.config.critical_rubrics
        weak_critical = critical_rubrics.intersection(set(rubric_analysis.weak_rubrics))

        if (
            weak_critical and result_progress < self.config.critical_rubrics_threshold
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
                elif result_progress < self.config.low_progress_threshold:
                    # Force continue if we don't have enough results yet
                    return False, (
                        f"CONTINUE: Stagnation detected but insufficient results ({current_result_count}/{desired_max_results}) - need at least 50%"
                    )

        # Continue searching - provide specific guidance
        if result_progress < self.config.low_progress_threshold:
            return (
                False,
                f"CONTINUE: Need more results ({current_result_count}/{desired_max_results}) + improving quality ({rubric_analysis.positive_rubric_count}/{self._TOTAL_RUBRICS})",
            )
        else:
            return (
                False,
                f"CONTINUE: Refining quality ({rubric_analysis.positive_rubric_count}/{self._TOTAL_RUBRICS}) with {current_result_count}/{desired_max_results} results",
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
    ) -> "_StoppingCriteria":
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
