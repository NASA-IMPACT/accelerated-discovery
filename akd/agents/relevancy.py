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


# Dynamic Criteria Generation Agent


class RelevanceCriterion(BaseModel):
    """A single relevance criterion for evaluating repositories."""

    name: str = Field(
        ..., description="Short identifier in snake_case (e.g., 'data_processing', 'machine_learning_models')"
    )
    description: str = Field(
        ..., description="Clear description of what makes a repository relevant for this criterion"
    )


class DynamicRelevanceCriteriaAgentInputSchema(InputSchema):
    """Input schema for the Dynamic Relevance Criteria Agent."""

    query: str = Field(..., description="The search query for which to generate relevance criteria")
    context: Optional[str] = Field(
        default=None, description="Optional additional context about the search domain or requirements"
    )


class DynamicRelevanceCriteriaAgentOutputSchema(OutputSchema):
    """
    Output schema for dynamically generated relevance criteria.

    This schema represents query-specific criteria that will be used to evaluate
    and rank code repositories based on their relevance to the search query.
    """

    required_relevance_criteria: List[RelevanceCriterion] = Field(
        ...,
        description=(
            "Core criteria that repositories MUST address to be considered relevant. "
            "These represent the essential requirements extracted from the query."
        ),
    )
    nice_to_have_relevance_criteria: List[RelevanceCriterion] = Field(
        default_factory=list,
        description=(
            "Optional bonus criteria for additional value. "
            "Repositories satisfying these provide extra relevance beyond core requirements."
        ),
    )
    query_intent_summary: str = Field(..., description="Brief summary of the overall query intent and search objective")
    reasoning_steps: List[str] = Field(
        ..., description="Step-by-step reasoning for the generation of the relevance criteria"
    )


class DynamicRelevanceCriteriaAgent(
    LiteLLMInstructorBaseAgent[DynamicRelevanceCriteriaAgentInputSchema, DynamicRelevanceCriteriaAgentOutputSchema]
):
    """
    Agent that dynamically generates query-specific relevance criteria for code repository search.

    This agent analyzes search queries to extract both required and optional relevance criteria
    that capture the query's intent and domain-specific requirements. Rather than applying a
    fixed rubric, it creates tailored evaluation criteria for each unique query.

    For code repository search, criteria might include:
    - Implementation of specific algorithms or methodologies
    - Addressing particular scientific domains or use cases
    - Providing certain functionality or features
    - Demonstrating relevant technical approaches
    - Supporting specific data types or formats
    - Integration with particular tools or frameworks

    The system distinguishes between:
    - Required criteria: Repositories must address these to be considered relevant
    - Nice-to-have criteria: Provide bonus scoring for additional value

    This dynamic approach enables the relevance assessment framework to adapt to each
    query's nuances, such as whether a repository should focus on data processing versus
    modeling, visualization versus analysis, or simulation versus observational data handling.
    """

    input_schema = DynamicRelevanceCriteriaAgentInputSchema
    output_schema = DynamicRelevanceCriteriaAgentOutputSchema
