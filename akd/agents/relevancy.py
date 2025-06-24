from enum import Enum
from typing import List, Optional

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.structures import RelevancyLabel


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
        description=(
            "The label indicating the relevance between the query and the content."
        ),
    )
    reasoning_steps: List[str] = Field(
        ...,
        description=(
            "Very concise/step-by-step reasoning steps leading to the relevance check."
        ),
    )


class RelevancyAgent(
    InstructorBaseAgent[RelevancyAgentInputSchema, RelevancyAgentOutputSchema],
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
    InstructorBaseAgent[
        MultiRubricRelevancyInputSchema,
        MultiRubricRelevancyOutputSchema,
    ],
):
    input_schema = MultiRubricRelevancyInputSchema
    output_schema = MultiRubricRelevancyOutputSchema
