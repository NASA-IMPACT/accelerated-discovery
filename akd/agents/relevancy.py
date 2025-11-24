from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent


class RelevancyLabel(str, Enum):
    RELEVANT = "Relevant"
    NOT_RELEVANT = "Not Relevant"


class RelevancyAgentInputSchema(InputSchema):
    """Input schema for relevancy agent"""

    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )
    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )


class RelevancyAgentOutputSchema(OutputSchema):
    """Output schema for relevancy agent"""

    label: RelevancyLabel = Field(
        ...,
        description=("The label indicating the relevance between the query and the content."),
    )
    reasoning_steps: List[str] = Field(
        ...,
        description=("Very concise/step-by-step reasoning steps leading to the relevance check."),
    )


class RelevancyAgent(
    LiteLLMInstructorBaseAgent[RelevancyAgentInputSchema, RelevancyAgentOutputSchema],
):
    input_schema = RelevancyAgentInputSchema
    output_schema = RelevancyAgentOutputSchema


# ---- Enhanced Way ---


class EnhancedRelevancyLabel(str, Enum):
    """Enhanced relevancy labels with more granularity"""

    HIGHLY_RELEVANT = "highly_relevant"
    MODERATELY_RELEVANT = "moderately_relevant"
    TANGENTIALLY_RELEVANT = "tangentially_relevant"
    NOT_RELEVANT = "not_relevant"


class TopicAlignmentLabel(str, Enum):
    ALIGNED = "aligned"
    NOT_ALIGNED = "not_aligned"


class ContentDepthLabel(str, Enum):
    COMPREHENSIVE = "comprehensive"
    SURFACE_LEVEL = "surface_level"


class RecencyRelevanceLabel(str, Enum):
    CURRENT = "current"
    OUTDATED = "outdated"


class MethodologicalRelevanceLabel(str, Enum):
    METHODOLOGICALLY_SOUND = "methodologically_sound"
    METHODOLOGICALLY_WEAK = "methodologically_weak"


class EvidenceQualityLabel(str, Enum):
    HIGH_QUALITY_EVIDENCE = "high_quality_evidence"
    MEDIUM_QUALITY_EVIDENCE = "medium_quality_evidence"
    LOW_QUALITY_EVIDENCE = "low_quality_evidence"


class ScopeRelevanceLabel(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


class MultiRubricRelevancyInputSchema(InputSchema):
    """Input schema for multi-rubric relevancy agent"""

    query: str = Field(..., description="The query to check for relevance.")
    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )
    domain_context: Optional[str] = Field(
        None,
        description="Additional domain context for better assessment",
    )


class MultiRubricRelevancyOutputSchema(OutputSchema):
    """Output schema for multi-rubric relevancy agent"""

    # Individual rubric assessments
    topic_alignment: TopicAlignmentLabel = Field(
        ...,
        description="Whether the content aligns with the query topic",
    )
    content_depth: ContentDepthLabel = Field(
        ...,
        description="Whether the content provides comprehensive coverage",
    )
    recency_relevance: RecencyRelevanceLabel = Field(
        ...,
        description="Whether the content is current and up-to-date",
    )
    methodological_relevance: MethodologicalRelevanceLabel = Field(
        ...,
        description="Whether the methodology/approach is sound",
    )
    evidence_quality: EvidenceQualityLabel = Field(
        ...,
        description="Whether the evidence presented is high quality",
    )
    scope_relevance: ScopeRelevanceLabel = Field(
        ...,
        description="Whether the content scope matches query requirements",
    )

    # Overall assessment
    overall_relevance: EnhancedRelevancyLabel = Field(
        ...,
        description="Overall relevance assessment",
    )

    reasoning_steps: List[str] = Field(
        ...,
        description="Step-by-step reasoning for each rubric assessment",
    )


class MultiRubricRelevancyAgent(
    LiteLLMInstructorBaseAgent[
        MultiRubricRelevancyInputSchema,
        MultiRubricRelevancyOutputSchema,
    ],
):
    input_schema = MultiRubricRelevancyInputSchema
    output_schema = MultiRubricRelevancyOutputSchema


# Relevancy Scoring Agent


class ScoringCategory(BaseModel):
    """Scoring category with name, description, and numeric value."""

    name: str = Field(
        ..., description="Category name (e.g., 'Fully Satisfies', 'Partially Satisfies', 'Does Not Satisfy')"
    )
    description: str = Field(..., description="Detailed description of what this category represents")
    value: float = Field(..., description="Numeric score for this category")


class CriterionEvaluation(BaseModel):
    """Evaluation result for a single criterion."""

    criterion_name: str = Field(..., description="Name of the criterion being evaluated")
    category: str = Field(
        ...,
        description="The selected scoring category (e.g., 'Fully Satisfies', 'Partially Satisfies', 'Does Not Satisfy')",
    )
    score: float = Field(..., description="Numeric score assigned based on the selected category")
    reasoning: str = Field(
        ..., description="Detailed explanation for why this score was assigned, with specific evidence from the content"
    )


class RelevanceCriterion(BaseModel):
    """A single relevance criterion for evaluating repositories."""

    name: str = Field(
        ..., description="Short identifier in snake_case (e.g., 'data_processing', 'machine_learning_models')"
    )
    description: str = Field(
        ..., description="Clear description of what makes a repository relevant for this criterion"
    )
    is_required: bool = Field(..., description="Whether this is a required criterion or nice-to-have")


class RelevancyScoringAgentInputSchema(InputSchema):
    """Input schema for the Relevancy Check Agent."""

    query: str = Field(..., description="The original search query")
    content: str = Field(
        ...,
        description=(
            "Content to evaluate against criteria. This is a concatenated string "
            "containing all relevant repository information (title, description, README, etc.)"
        ),
    )
    required_criteria: List[RelevanceCriterion] = Field(
        ..., description="Required relevance criteria that must be addressed"
    )
    nice_to_have_criteria: List[RelevanceCriterion] = Field(
        default_factory=list, description="Optional nice-to-have criteria for bonus scoring"
    )


class RelevancyScoringAgentOutputSchema(OutputSchema):
    """
    Output schema for relevancy evaluation results.

    This schema contains evaluations for both required and nice-to-have criteria,
    with scores, categories, and reasoning for each.
    """

    required_criteria_evaluations: List[CriterionEvaluation] = Field(
        ..., description="Evaluation results for all required criteria"
    )
    nice_to_have_criteria_evaluations: List[CriterionEvaluation] = Field(
        default_factory=list, description="Evaluation results for all nice-to-have criteria"
    )
    overall_assessment: str = Field(
        ...,
        description=(
            "Concise feedback for query refinement explaining what aspects of the query "
            "were well-matched, what was missing or poorly matched, and suggestions for "
            "improving retrieval. Focus on actionable insights about query-content alignment."
        ),
    )


class RelevancyScoringAgent(
    LiteLLMInstructorBaseAgent[RelevancyScoringAgentInputSchema, RelevancyScoringAgentOutputSchema]
):
    """
    Agent that evaluates content relevance against dynamically generated criteria.

    This agent takes query-specific relevance criteria (from DynamicRelevanceCriteriaAgent)
    and evaluates repository content to determine how well it satisfies each criterion.

    The agent uses a three-level scoring rubric for each criterion:
    - Fully Satisfies (3.0): Content clearly and comprehensively addresses the criterion
    - Partially Satisfies (1.5): Content somewhat addresses the criterion but incompletely
    - Does Not Satisfy (0.0): Content does not address the criterion

    Key features:
    - Evaluates both required and nice-to-have criteria
    - Provides detailed reasoning for each score
    - Single evaluation call assesses all criteria efficiently
    - Evidence-based assessment with specific content references

    The output can be used for:
    - Filtering out irrelevant repositories
    - Ranking repositories by relevance (scoring done downstream)
    - Explaining why repositories were selected or rejected
    - Tuning search and retrieval systems
    """

    input_schema = RelevancyScoringAgentInputSchema
    output_schema = RelevancyScoringAgentOutputSchema
