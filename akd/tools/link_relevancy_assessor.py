"""
Link Relevancy Assessor Tool for evaluating individual search result relevancy.

This tool processes individual SearchResultItems and enriches them with detailed
relevancy metadata using the MultiRubricRelevancyAgent.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.agents.relevancy import (
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
    MultiRubricRelevancyOutputSchema,
)
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig


class ScoringWeights(BaseModel):
    """Configuration for relevancy scoring weights."""

    topic_alignment_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for topic alignment scoring",
    )
    content_depth_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for content depth scoring",
    )
    evidence_quality_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for evidence quality scoring",
    )
    methodological_relevance_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for methodological relevance scoring",
    )
    recency_relevance_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for recency relevance scoring",
    )
    scope_relevance_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for scope relevance scoring",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that weights sum to 1.0."""
        total = (
            self.topic_alignment_weight
            + self.content_depth_weight
            + self.evidence_quality_weight
            + self.methodological_relevance_weight
            + self.recency_relevance_weight
            + self.scope_relevance_weight
        )
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")


class LinkRelevancyAssessorConfig(BaseToolConfig):
    """Configuration for the LinkRelevancyAssessor tool."""

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
    assessment_batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of links to assess in parallel",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of relevancy assessments",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )
    scoring_weights: ScoringWeights = Field(
        default_factory=ScoringWeights,
        description="Scoring weights for relevancy calculation",
    )


class LinkRelevancyAssessorInputSchema(InputSchema):
    """Input schema for the LinkRelevancyAssessor tool."""

    search_results: List[SearchResultItem] = Field(
        ...,
        description="List of search results to assess for relevancy",
    )
    original_query: str = Field(
        ...,
        description="The original search query",
    )
    reformulated_query: Optional[str] = Field(
        None,
        description="Reformulated query if different from original",
    )
    domain_context: Optional[str] = Field(
        None,
        description="Additional domain context for better relevancy assessment",
    )


class LinkRelevancyAssessorOutputSchema(OutputSchema):
    """Output schema for the LinkRelevancyAssessor tool."""

    assessed_results: List[SearchResultItem] = Field(
        ...,
        description="Search results enriched with relevancy metadata",
    )
    filtered_results: List[SearchResultItem] = Field(
        ...,
        description="Results filtered by minimum relevancy threshold",
    )
    high_relevancy_results: List[SearchResultItem] = Field(
        ...,
        description="Results that should have full content fetched",
    )
    assessment_summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics of the relevancy assessment",
    )


class LinkRelevancyAssessor(BaseTool):
    """
    Tool for assessing individual link relevancy using multi-rubric analysis.

    This tool processes SearchResultItems individually to provide detailed
    relevancy scoring and metadata, enabling smarter content filtering and
    prioritization.
    """

    input_schema = LinkRelevancyAssessorInputSchema
    output_schema = LinkRelevancyAssessorOutputSchema
    config_schema = LinkRelevancyAssessorConfig

    def __init__(
        self,
        config: Optional[LinkRelevancyAssessorConfig] = None,
        relevancy_agent: Optional[MultiRubricRelevancyAgent] = None,
        debug: bool = False,
    ):
        """Initialize the LinkRelevancyAssessor."""
        config = config or LinkRelevancyAssessorConfig(debug=debug)
        super().__init__(config=config, debug=debug)

        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()
        self._assessment_cache: Dict[str, MultiRubricRelevancyOutputSchema] = {}

    def _create_cache_key(self, query: str, content: str) -> str:
        """Create a cache key for relevancy assessments."""
        import hashlib

        key_string = f"{query}:{content[:200]}"  # Use first 200 chars for cache key
        return hashlib.md5(key_string.encode()).hexdigest()

    def _calculate_relevancy_score(
        self,
        assessment: MultiRubricRelevancyOutputSchema,
    ) -> float:
        """Calculate a numeric relevancy score from multi-rubric assessment."""
        weights = self.config.scoring_weights

        # Map enum values to scores
        topic_score = 1.0 if assessment.topic_alignment.value == "aligned" else 0.0
        depth_score = 1.0 if assessment.content_depth.value == "comprehensive" else 0.0
        evidence_score = (
            1.0 if assessment.evidence_quality.value == "high_quality_evidence" else 0.0
        )
        method_score = (
            1.0
            if assessment.methodological_relevance.value == "methodologically_sound"
            else 0.0
        )
        recency_score = 1.0 if assessment.recency_relevance.value == "current" else 0.0
        scope_score = 1.0 if assessment.scope_relevance.value == "in_scope" else 0.0

        # Calculate weighted scores
        weighted_scores = [
            topic_score * weights.topic_alignment_weight,
            depth_score * weights.content_depth_weight,
            evidence_score * weights.evidence_quality_weight,
            method_score * weights.methodological_relevance_weight,
            recency_score * weights.recency_relevance_weight,
            scope_score * weights.scope_relevance_weight,
        ]

        total_score = sum(weighted_scores)

        if self.debug:
            logger.debug("🧮 RELEVANCY SCORE BREAKDOWN:")
            logger.debug(
                f"  📍 Topic alignment: {assessment.topic_alignment.value} -> {topic_score:.2f} * {weights.topic_alignment_weight:.2f} = {weighted_scores[0]:.3f}",
            )
            logger.debug(
                f"  📊 Content depth: {assessment.content_depth.value} -> {depth_score:.2f} * {weights.content_depth_weight:.2f} = {weighted_scores[1]:.3f}",
            )
            logger.debug(
                f"  🏆 Evidence quality: {assessment.evidence_quality.value} -> {evidence_score:.2f} * {weights.evidence_quality_weight:.2f} = {weighted_scores[2]:.3f}",
            )
            logger.debug(
                f"  🔬 Methodological: {assessment.methodological_relevance.value} -> {method_score:.2f} * {weights.methodological_relevance_weight:.2f} = {weighted_scores[3]:.3f}",
            )
            logger.debug(
                f"  ⏰ Recency: {assessment.recency_relevance.value} -> {recency_score:.2f} * {weights.recency_relevance_weight:.2f} = {weighted_scores[4]:.3f}",
            )
            logger.debug(
                f"  🎯 Scope: {assessment.scope_relevance.value} -> {scope_score:.2f} * {weights.scope_relevance_weight:.2f} = {weighted_scores[5]:.3f}",
            )
            logger.debug(f"  ✅ TOTAL SCORE: {total_score:.3f}")

            # Log reasoning steps for problematic scores
            if total_score >= 0.7:  # High score - check if it should be high
                logger.debug(f"  💭 REASONING: {assessment.reasoning_steps}")

        return total_score

    async def _assess_single_result(
        self,
        result: SearchResultItem,
        query: str,
        domain_context: Optional[str] = None,
    ) -> SearchResultItem:
        """Assess relevancy for a single search result."""
        if not result.content:
            # No content to assess, assign low relevancy
            result.score = 0.1
            result.should_fetch_full_content = False
            return result

        # Check cache first
        cache_key = self._create_cache_key(query, result.content)

        if self.enable_caching and cache_key in self._assessment_cache:
            assessment = self._assessment_cache[cache_key]
            if self.debug:
                logger.debug(f"Using cached assessment for {result.url}")
        else:
            # Perform new assessment
            try:
                assessment_input = MultiRubricRelevancyInputSchema(
                    query=query,
                    content=f"Title: {result.title}\nContent: {result.content}",
                    domain_context=domain_context,
                )

                assessment = await self.relevancy_agent.arun(assessment_input)

                # Cache the assessment
                if self.enable_caching:
                    self._assessment_cache[cache_key] = assessment

            except Exception as e:
                logger.warning(f"Error assessing relevancy for {result.url}: {e}")
                # Assign default low relevancy on error
                result.score = 0.2
                result.should_fetch_full_content = False
                return result

        # Calculate numeric score
        relevancy_score = self._calculate_relevancy_score(assessment)

        # Update result with relevancy metadata
        result.score = relevancy_score
        result.relevancy_assessment = assessment.model_dump()
        result.should_fetch_full_content = (
            relevancy_score >= self.full_content_threshold
        )

        if self.debug:
            logger.debug(
                f"Assessed {result.url}: score={relevancy_score:.2f}, "
                f"should_fetch={result.should_fetch_full_content}",
            )

        return result

    async def _assess_query_alignment(
        self,
        results: List[SearchResultItem],
        original_query: str,
        reformulated_query: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> List[SearchResultItem]:
        """Assess results against both original and reformulated queries."""

        # Process in batches for efficiency
        assessed_results = []

        for i in range(0, len(results), self.assessment_batch_size):
            batch = results[i : i + self.assessment_batch_size]

            # Assess against original query
            original_tasks = [
                self._assess_single_result(result, original_query, domain_context)
                for result in batch
            ]

            batch_results = await asyncio.gather(*original_tasks)

            # If reformulated query exists, assess against it too
            if reformulated_query and reformulated_query != original_query:
                reformulated_tasks = [
                    self._assess_single_result(
                        result,
                        reformulated_query,
                        domain_context,
                    )
                    for result in batch_results
                ]

                reformulated_results = await asyncio.gather(*reformulated_tasks)

                # Compare scores and keep better alignment
                for orig_result, reform_result in zip(
                    batch_results,
                    reformulated_results,
                ):
                    if reform_result.relevancy_score > orig_result.relevancy_score:
                        # Use reformulated query assessment
                        orig_result.score = reform_result.relevancy_score
                        orig_result.relevancy_assessment = (
                            reform_result.relevancy_assessment
                        )
                        orig_result.should_fetch_full_content = (
                            reform_result.should_fetch_full_content
                        )
                        orig_result.query_alignment_details = {
                            "best_query": "reformulated",
                            "original_score": orig_result.relevancy_score,
                            "reformulated_score": reform_result.relevancy_score,
                        }
                    else:
                        orig_result.query_alignment_details = {
                            "best_query": "original",
                            "original_score": orig_result.relevancy_score,
                            "reformulated_score": reform_result.relevancy_score,
                        }

            assessed_results.extend(batch_results)

            # Small delay between batches
            if i + self.assessment_batch_size < len(results):
                await asyncio.sleep(0.1)

        return assessed_results

    def _create_assessment_summary(
        self,
        results: List[SearchResultItem],
    ) -> Dict[str, Any]:
        """Create summary statistics of the relevancy assessment."""
        if not results:
            return {
                "total_results": 0,
                "assessed_results": 0,
                "avg_relevancy_score": 0.0,
                "high_relevancy_count": 0,
                "filtered_count": 0,
            }

        assessed_results = [r for r in results if r.relevancy_score is not None]
        scores = [
            r.relevancy_score for r in assessed_results if r.relevancy_score is not None
        ]

        return {
            "total_results": len(results),
            "assessed_results": len(assessed_results),
            "avg_relevancy_score": sum(scores) / len(scores) if scores else 0.0,
            "min_relevancy_score": min(scores) if scores else 0.0,
            "max_relevancy_score": max(scores) if scores else 0.0,
            "high_relevancy_count": len(
                [r for r in results if r.should_fetch_full_content],
            ),
            "filtered_count": len(
                [
                    r
                    for r in assessed_results
                    if r.relevancy_score >= self.min_relevancy_score
                ],
            ),
            "cache_hits": len(self._assessment_cache) if self.enable_caching else 0,
        }

    async def _arun(
        self,
        params: LinkRelevancyAssessorInputSchema,
        **kwargs,
    ) -> LinkRelevancyAssessorOutputSchema:
        """Run the link relevancy assessment."""

        if self.debug:
            logger.info(f"Assessing relevancy for {len(params.search_results)} results")

        # Assess all results
        assessed_results = await self._assess_query_alignment(
            results=params.search_results,
            original_query=params.original_query,
            reformulated_query=params.reformulated_query,
            domain_context=params.domain_context,
        )

        # Filter results by minimum relevancy threshold
        filtered_results = [
            result
            for result in assessed_results
            if result.relevancy_score is not None
            and result.relevancy_score >= self.min_relevancy_score
        ]

        # Identify high-relevancy results for full content fetching
        high_relevancy_results = [
            result for result in assessed_results if result.should_fetch_full_content
        ]

        # Create assessment summary
        assessment_summary = self._create_assessment_summary(assessed_results)

        if self.debug:
            logger.info(f"Assessment complete: {assessment_summary}")

        return LinkRelevancyAssessorOutputSchema(
            assessed_results=assessed_results,
            filtered_results=filtered_results,
            high_relevancy_results=high_relevancy_results,
            assessment_summary=assessment_summary,
        )
