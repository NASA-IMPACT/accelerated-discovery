import itertools
from typing import List, Optional, Self

from atomic_agents.lib.base.base_tool import BaseToolConfig
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from akd.agents.relevancy import (
    ContentDepthLabel,
    EvidenceQualityLabel,
    MethodologicalRelevanceLabel,
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
    MultiRubricRelevancyOutputSchema,
    RecencyRelevanceLabel,
    RelevancyAgent,
    RelevancyAgentInputSchema,
    RelevancyAgentOutputSchema,
    ScopeRelevanceLabel,
    TopicAlignmentLabel,
)
from akd.structures import RelevancyLabel

from ._base import BaseIOSchema, BaseTool


class RelevancyCheckerInputSchema(RelevancyAgentInputSchema):
    """Input schema for the RelevancyChecker."""

    pass


class _RelevancyCheckerSwappedInputSchema(BaseIOSchema):
    """Schema with swapped field order"""

    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )
    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )


class RelevancyCheckerOutputSchema(BaseIOSchema):
    """Output schema for the RelevancyChecker."""

    score: float = Field(
        ...,
        description="The relevance score between the query and the content.",
        ge=0.0,
        le=1.0,
    )
    reasoning_steps: List[str] = Field(
        ...,
        description="The reasoning steps leading to the relevance check.",
    )


class RelevancyCheckerConfig(BaseToolConfig):
    """Configuration for the RelevancyChecker."""

    model_config = {"arbitrary_types_allowed": True}

    debug: bool = Field(
        default=True,
        description="Boolean flag for debug mode",
    )
    n_iter: int = Field(
        default=2,
        description="Number of iterations for the relevancy check.",
    )
    swapping: bool = Field(
        default=True,
        description="Boolean flag for swapping query and content.",
    )
    agent: RelevancyAgent = Field(
        default_factory=RelevancyAgent,
        description="Relevancy agent to use for the check.",
    )


class RelevancyChecker(BaseTool):
    input_schema = RelevancyCheckerInputSchema
    output_schema = RelevancyCheckerOutputSchema

    def __init__(
        self,
        config: Optional[RelevancyCheckerConfig] = None,
        debug: bool = False,
    ) -> None:
        config = config or RelevancyCheckerConfig()
        super().__init__(config, debug)

    async def arun(
        self,
        params: RelevancyCheckerInputSchema,
    ) -> RelevancyCheckerOutputSchema:
        logger.info(f"Running relevancy check for query: {params.query}")
        outputs = await self._run(params, n_iter=self.config.n_iter)
        if self.config.swapping:
            logger.info("Running swapping pass. Query and Content swapped")
            param_swapped = _RelevancyCheckerSwappedInputSchema(
                content=params.content,
                query=params.query,
            )
            outputs.extend(await self._run(param_swapped, n_iter=self.config.n_iter))
        return await self._ensemble(outputs)

    async def _run(
        self,
        params: RelevancyCheckerInputSchema,
        n_iter: int = 3,
    ) -> List[RelevancyAgentOutputSchema]:
        outputs = []
        for i in range(n_iter):
            output = self.config.agent.run(params)
            if self.debug:
                logger.debug(f"Relevancy check {i + 1}: {output}")
            outputs.append(output)
            self.config.agent.reset_memory()
        return outputs

    async def _ensemble(
        self,
        outputs: List[RelevancyAgentOutputSchema],
    ) -> RelevancyCheckerOutputSchema:
        logger.info(f"Ensembling {len(outputs)} outputs")
        relevance = len(
            list(filter(lambda r: r.label == RelevancyLabel.RELEVANT, outputs)),
        ) / len(outputs)
        return RelevancyCheckerOutputSchema(
            score=relevance,
            reasoning_steps=list(
                itertools.chain(*map(lambda r: r.reasoning_steps, outputs)),
            ),
        )


# ---- Enhanced Relevancy Checker ----


class RubricWeights(BaseModel):
    """
    Configurable weights for different relevancy aspects.
    These weights determine the importance of each aspect during scoring.
    The weights must sum to 1.0, and each weight must be between 0.0 and 1.0.
    The default weights are set to reflect a balanced approach, but they can be adjusted
    based on usecases.
    """

    topic_alignment: float = Field(
        default=0.3,
        description="Weight for topic alignment assessment",
        ge=0.0,
        le=1.0,
    )
    content_depth: float = Field(
        default=0.2,
        description="Weight for content depth assessment",
        ge=0.0,
        le=1.0,
    )
    recency_relevance: float = Field(
        default=0.15,
        description="Weight for recency relevance assessment",
        ge=0.0,
        le=1.0,
    )
    methodological_relevance: float = Field(
        default=0.15,
        description="Weight for methodological relevance assessment",
        ge=0.0,
        le=1.0,
    )
    evidence_quality: float = Field(
        default=0.1,
        description="Weight for evidence quality assessment",
        ge=0.0,
        le=1.0,
    )
    scope_relevance: float = Field(
        default=0.1,
        description="Weight for scope relevance assessment",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> Self:
        total = (
            self.topic_alignment
            + self.content_depth
            + self.recency_relevance
            + self.methodological_relevance
            + self.evidence_quality
            + self.scope_relevance
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self


class RubricScoringConfig(BaseModel):
    """Configurable scoring for rubric labels"""

    topic_alignment: dict[str, float] = Field(
        default_factory=lambda: {
            TopicAlignmentLabel.ALIGNED: 1.0,
            TopicAlignmentLabel.NOT_ALIGNED: 0.0,
        },
        description="Scores for topic alignment labels",
    )

    content_depth: dict[str, float] = Field(
        default_factory=lambda: {
            ContentDepthLabel.COMPREHENSIVE: 1.0,
            ContentDepthLabel.SURFACE_LEVEL: 0.3,
        },
        description="Scores for content depth labels",
    )

    recency_relevance: dict[str, float] = Field(
        default_factory=lambda: {
            RecencyRelevanceLabel.CURRENT: 1.0,
            RecencyRelevanceLabel.OUTDATED: 0.2,
        },
        description="Scores for recency relevance labels",
    )

    methodological_relevance: dict[str, float] = Field(
        default_factory=lambda: {
            MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND: 1.0,
            MethodologicalRelevanceLabel.METHODOLOGICALLY_WEAK: 0.1,
        },
        description="Scores for methodological relevance labels",
    )

    evidence_quality: dict[str, float] = Field(
        default_factory=lambda: {
            EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE: 1.0,
            EvidenceQualityLabel.LOW_QUALITY_EVIDENCE: 0.2,
        },
        description="Scores for evidence quality labels",
    )

    scope_relevance: dict[str, float] = Field(
        default_factory=lambda: {
            ScopeRelevanceLabel.IN_SCOPE: 1.0,
            ScopeRelevanceLabel.OUT_OF_SCOPE: 0.0,
        },
        description="Scores for scope relevance labels",
    )


class EnhancedRelevancyCheckerConfig(BaseToolConfig):
    """Configuration for the Enhanced RelevancyChecker."""

    model_config = {"arbitrary_types_allowed": True}

    debug: bool = Field(
        default=False,
        description="Boolean flag for debug mode",
    )
    n_iter: int = Field(
        default=2,
        description="Number of iterations for the relevancy check",
    )
    swapping: bool = Field(
        default=False,
        description="Boolean flag for swapping query and content",
    )
    rubric_weights: RubricWeights = Field(
        default_factory=RubricWeights,
        description="Weights for different relevancy rubrics",
    )

    rubric_scoring_config: RubricScoringConfig = Field(
        default_factory=RubricScoringConfig,
        description="Numerical scores assigned to each rubric metric",
    )
    relevance_threshold: float = Field(
        default=0.6,
        description="Threshold for considering content relevant",
        ge=0.0,
        le=1.0,
    )
    agent: MultiRubricRelevancyAgent = Field(
        default_factory=MultiRubricRelevancyAgent,
        description="Multi-rubric relevancy agent",
    )


class EnhancedRelevancyCheckerOutputSchema(BaseIOSchema):
    """Output schema for the Enhanced RelevancyChecker."""

    score: float = Field(
        ...,
        description="The computed relevance score between the query and the content",
        ge=0.0,
        le=1.0,
    )
    is_relevant: bool = Field(
        ...,
        description="Boolean indicating if content meets relevance threshold",
    )
    rubric_scores: dict[str, float] = Field(
        ...,
        description="Individual scores for each rubric dimension",
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assessment based on consistency across iterations",
        ge=0.0,
        le=1.0,
    )
    reasoning_steps: List[str] = Field(
        ...,
        description="The reasoning steps leading to the relevance check",
    )


class EnhancedRelevancyChecker(
    BaseTool[MultiRubricRelevancyInputSchema, EnhancedRelevancyCheckerOutputSchema],
):
    input_schema = MultiRubricRelevancyInputSchema
    output_schema = EnhancedRelevancyCheckerOutputSchema

    class _MultiRubricRelevancyInputSchema(BaseIOSchema):
        """Input schema for multi-rubric relevancy agent"""

        query: str = Field(
            ...,
            description="The query to check for relevance.",
        )
        content: str = Field(
            ...,
            description="The content to check for relevance.",
        )
        domain_context: Optional[str] = Field(
            None,
            description="Additional domain context for better assessment",
        )

    def __init__(
        self,
        config: Optional[EnhancedRelevancyCheckerConfig] = None,
        debug: bool = False,
    ) -> None:
        config = config or EnhancedRelevancyCheckerConfig()
        super().__init__(config, debug)

    async def arun(
        self,
        param: MultiRubricRelevancyInputSchema,
    ) -> EnhancedRelevancyCheckerOutputSchema:
        logger.info(
            f"Running enhanced relevancy check for query: {param.query}",
        )

        # Run multiple iterations
        outputs = await self._run(param, n_iter=self.config.n_iter)

        # Optional swapping pass
        if self.config.swapping:
            logger.info("Running swapping pass - Query and Content swapped")
            param_swapped = self._MultiRubricRelevancyInputSchema(
                content=param.content,
                query=param.query,
                domain_context=param.domain_context,
            )
            swapped_outputs = await self._run(
                param_swapped,
                n_iter=self.config.n_iter,
            )
            outputs.extend(swapped_outputs)

        return await self._ensemble_with_rubrics(outputs)

    async def _run(
        self,
        param: MultiRubricRelevancyInputSchema,
        n_iter: int = 3,
    ) -> List[MultiRubricRelevancyOutputSchema]:
        outputs = []
        for i in range(n_iter):
            output = self.config.agent.run(param)
            if self.debug:
                logger.debug(f"Multi-rubric relevancy check {i + 1}: {output}")
            outputs.append(output)
            self.config.agent.reset_memory()
        return outputs

    def _calculate_rubric_score(
        self,
        rubric_result: str,
        rubric_type: str,
    ) -> float:
        """Convert rubric labels to numerical scores"""
        return self.rubric_scoring_config.get(rubric_type, {}).get(
            rubric_result,
            0.0,
        )

    async def _ensemble_with_rubrics(
        self,
        outputs: List[MultiRubricRelevancyOutputSchema],
    ) -> EnhancedRelevancyCheckerOutputSchema:
        logger.info(f"Ensembling {len(outputs)} multi-rubric outputs")

        # Calculate average scores for each rubric
        rubric_fields = [
            "topic_alignment",
            "content_depth",
            "recency_relevance",
            "methodological_relevance",
            "evidence_quality",
            "scope_relevance",
        ]

        rubric_scores = {}
        for field in rubric_fields:
            field_scores = []
            for output in outputs:
                field_value = getattr(output, field)
                score = self._calculate_rubric_score(field_value, field)
                field_scores.append(score)
            rubric_scores[field] = sum(field_scores) / len(field_scores)

        # Calculate weighted overall score
        weights = self.config.rubric_weights
        overall_score = (
            rubric_scores["topic_alignment"] * weights.topic_alignment
            + rubric_scores["content_depth"] * weights.content_depth
            + rubric_scores["recency_relevance"] * weights.recency_relevance
            + rubric_scores["methodological_relevance"]
            * weights.methodological_relevance
            + rubric_scores["evidence_quality"] * weights.evidence_quality
            + rubric_scores["scope_relevance"] * weights.scope_relevance
        )

        # Calculate confidence based on consistency across iterations
        all_scores = []
        for output in outputs:
            output_score = sum(
                [
                    self._calculate_rubric_score(getattr(output, field), field)
                    * getattr(weights, field)
                    for field in rubric_fields
                ],
            )
            all_scores.append(output_score)

        # Confidence is inverse of variance (higher consistency = higher confidence)
        if len(all_scores) > 1:
            variance = sum((s - overall_score) ** 2 for s in all_scores) / len(
                all_scores,
            )
            confidence = max(
                0.0,
                1.0 - (variance * 4),
            )  # Scale variance to 0-1 range
        else:
            confidence = 0.5  # Default confidence for single iteration

        # Determine if content is relevant based on threshold
        is_relevant = overall_score >= self.config.relevance_threshold

        # Combine all reasoning steps
        all_reasoning = list(
            itertools.chain(*[output.reasoning_steps for output in outputs]),
        )

        return EnhancedRelevancyCheckerOutputSchema(
            score=overall_score,
            is_relevant=is_relevant,
            rubric_scores=rubric_scores,
            confidence=confidence,
            reasoning_steps=all_reasoning,
        )
