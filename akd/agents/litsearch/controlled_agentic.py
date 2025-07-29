"""
Controlled Agentic Literature Search Agent

This agent performs controlled agentic literature searches using multi-rubric analysis
and agentic decision-making to iteratively refine search queries based on rubric assessments.
"""

from __future__ import annotations

from typing import Any, List

from loguru import logger
from pydantic import Field

from akd.agents.query import FollowUpQueryAgent, QueryAgent
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
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
)
from akd.tools.search._base import QueryFocusStrategy
from akd.tools.search.searxng_search import SearxNGSearchTool

from ._base import (
    LitBaseAgent,
    LitSearchAgentConfig,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
    RubricAnalysis,
    StoppingCriteria,
)


class ControlledAgenticLitSearchAgentConfig(LitSearchAgentConfig):
    """
    Configuration for the ControlledAgenticLitSearchAgent.
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


class ControlledAgenticLitSearchAgent(LitBaseAgent):
    """
    Agent for performing controlled agentic literature searches
    using multi-rubric analysis and agentic decision-making.

    This agent iteratively refines search queries based on rubric assessments
    and dynamically decides when to stop searching based on quality and quantity of results.

    Note:
        - It's not stateless. Meaning: we track the history of rubrics
          and decisions made during the search process.
    """

    config_schema = ControlledAgenticLitSearchAgentConfig

    def __init__(
        self,
        config: ControlledAgenticLitSearchAgentConfig | None = None,
        search_tool=None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            config=config or ControlledAgenticLitSearchAgentConfig(), debug=debug
        )

        self.search_tool = search_tool or SearxNGSearchTool()
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()
        self.query_agent = query_agent or QueryAgent()
        self.followup_query_agent = followup_query_agent or FollowUpQueryAgent()

        # Track rubric patterns for agentic learning
        self.rubric_history = []

        # Initialize link relevancy assessor if enabled
        if self.config.enable_per_link_assessment:
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
    ) -> RubricAnalysis:
        """Analyze multi-rubric output to determine positive/negative assessments."""
        analysis = RubricAnalysis()

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
        rubric_analysis: RubricAnalysis,
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
        quality_threshold = self.config.min_positive_rubrics
        if result_progress > 0.8:  # If we have most requested results
            quality_threshold = max(
                2,
                self.config.min_positive_rubrics - 1,
            )  # Lower quality bar
        elif result_progress < 0.5:  # If we have few results
            quality_threshold = min(
                5,
                self.config.min_positive_rubrics + 1,
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
                result_progress >= self.config.early_stop_result_progress
                and quality_score >= self.config.early_stop_quality_score
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
        if len(self.rubric_history) >= self.config.rubric_improvement_threshold:
            recent_scores = [
                h["analysis"].positive_rubric_count
                for h in self.rubric_history[
                    -self.config.rubric_improvement_threshold :
                ]
            ]
            if all(
                score <= rubric_analysis.positive_rubric_count
                for score in recent_scores
            ):
                # Only stop for stagnation if we have BOTH reasonable quantity AND excellent quality
                # Never stop for stagnation if we have less than 50% of requested results
                if (
                    result_progress >= self.config.stagnation_result_progress
                    and quality_score >= self.config.stagnation_quality_score
                ):
                    return True, (
                        f"STOP: No improvement in {self.config.rubric_improvement_threshold} iterations + sufficient results ({current_result_count}/{desired_max_results}, {result_progress:.1%}) + excellent quality ({quality_score:.1%})"
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

    async def _should_stop(
        self,
        iteration: int,
        query: str,
        all_results: list,
        current_results: list,
        max_results: int,
    ) -> StoppingCriteria:
        criteria = StoppingCriteria(
            stop_now=False,
            reasoning_trace=f"Iteration {iteration}/{self.config.max_iteration}",
        )

        # Basic stopping conditions
        if iteration >= self.config.max_iteration:
            criteria.stop_now = True
            criteria.reasoning_trace = (
                f"Max iterations reached ({iteration}/{self.config.max_iteration})"
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
                logger.warning(f"Error in multi-rubric analysis: {e}")
                # Fallback to continue searching if analysis fails
                criteria.stop_now = False
                criteria.reasoning_trace = (
                    "Multi-rubric analysis failed, continuing search"
                )

        return criteria

    def _generate_query_focus_recommendations(
        self,
        rubric_analysis: RubricAnalysis,
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

    def _learn_from_iteration(
        self,
        iteration: int,
        queries: List[str],
        rubric_analysis: RubricAnalysis,
    ):
        """Simple learning: just track rubric history for trend analysis."""
        self.rubric_history.append(
            {
                "iteration": iteration,
                "analysis": rubric_analysis,
                "query": " AND ".join(queries),
            },
        )

    async def _arun(
        self,
        params: LitSearchAgentInputSchema,
        **kwargs: Any,
    ) -> LitSearchAgentOutputSchema:
        desired_max_results = params.max_results
        queries = [params.query]  # Convert single query to list for compatibility

        iteration = 0
        all_results = []
        current_results = []

        while not (
            criteria := await self._should_stop(
                iteration=iteration,
                all_results=all_results,
                current_results=current_results,
                max_results=desired_max_results,
                query=params.query,
            )
        ).stop_now:
            logger.debug(f"Stopping Criteria :: {criteria}")
            iteration += 1

            if self.debug:
                logger.info(f"🔄 ITERATION {iteration}")

            # For this simplified version, we'll use the basic search functionality
            # The full implementation would include all the query generation and adaptation logic
            from akd.tools.search._base import SearchToolInputSchema

            search_input = SearchToolInputSchema(
                queries=queries,
                max_results=min(
                    desired_max_results - len(all_results),
                    self.config.max_results_per_iteration,
                ),
                category=params.category,
            )

            search_result = await self.search_tool.arun(
                self.search_tool.input_schema(**search_input.model_dump()),
            )

            current_results = self._deduplicate_results(
                new_results=search_result.results,
                existing_results=all_results,
            )

            all_results.extend(current_results)

            # Learn from this iteration if we have rubric analysis
            if hasattr(criteria, "rubric_analysis") and criteria.rubric_analysis:
                self._learn_from_iteration(
                    iteration,
                    queries,
                    criteria.rubric_analysis,
                )

        logger.debug(f"Final Stopping Criteria :: {criteria}")

        # Convert SearchResultItem objects to dictionaries for output
        results_as_dicts = []
        for result in all_results:
            results_as_dicts.append(
                {
                    "url": str(result.url),
                    "title": result.title,
                    "content": result.content,
                    "category": result.category,
                }
            )

        return LitSearchAgentOutputSchema(
            results=results_as_dicts,
            category=params.category,
            iterations_performed=iteration,
        )
