"""
Shared schemas for agents to avoid circular imports.

This module contains input/output schemas that are used across multiple modules,
separated from the agent implementations to prevent circular dependencies.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field
from akd._base import InputSchema, OutputSchema


class QueryAgentInputSchema(InputSchema):
    """This is the input schema for the QueryAgent."""

    query: str = Field(
        ...,
        description="A detailed query/instruction or request to "
        "generate search engine queries for.",
    )
    num_queries: int = Field(
        default=3,
        description="The number of search queries to generate.",
    )


class QueryAgentOutputSchema(OutputSchema):
    """
    Schema for output queries  for information, news,
    references, and other content.

    Returns a list of search results with a short
    description or content snippet and URLs for further exploration
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "science"]] = Field(
        "science",
        description="Category of the search queries.",
    )


class FollowUpQueryAgentInputSchema(InputSchema):
    """This is the input schema for the FollowUpQueryAgent."""

    original_queries: List[str] = Field(
        ...,
        description="The original search queries that were used to retrieve content.",
    )

    content: str = Field(
        ...,
        description="The text content obtained from the original queries "
        "that will be used to generate follow-up queries.",
    )

    num_queries: int = Field(
        default=3,
        description="The number of follow-up search queries to generate.",
    )

    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas or aspects to focus on for follow-up queries. "
        "If not provided, the agent will identify gaps and interesting areas automatically.",
    )


class FollowUpQueryAgentOutputSchema(OutputSchema):
    """
    Schema for output follow-up queries based on original queries and content.
    Returns a list of refined search queries that dig deeper into the topic
    or explore related areas not covered in the original content.
    """

    followup_queries: List[str] = Field(
        ...,
        description="List of follow-up search queries based on the original content.",
    )

    category: Optional[Literal["general", "science", "research", "clarification"]] = (
        Field(
            default="general",
            description="Category of the follow-up search queries.",
        )
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why these follow-up queries were generated "
        "and what gaps or areas they aim to address.",
    )

    original_query_gaps: Optional[List[str]] = Field(
        default=None,
        description="Identified gaps or areas that weren't fully covered "
        "in the original content.",
    )


# Deep Research Agent Schemas


class ClarifyingAgentInputSchema(InputSchema):
    """Input schema for the ClarifyingAgent that asks follow-up questions."""

    query: str = Field(
        ...,
        description="The user's research query that may need clarification.",
    )
    
    context: Optional[str] = Field(
        default=None,
        description="Additional context about the research request if available.",
    )


class ClarifyingAgentOutputSchema(OutputSchema):
    """Output schema for the ClarifyingAgent with clarifying questions."""

    questions: List[str] = Field(
        ...,
        description="List of 2-3 clarifying questions to gather more context.",
        min_length=1,
        max_length=3,
    )
    
    needs_clarification: bool = Field(
        default=True,
        description="Whether the query needs clarification.",
    )


class InstructionBuilderInputSchema(InputSchema):
    """Input schema for the InstructionBuilderAgent that creates research instructions."""

    query: str = Field(
        ...,
        description="The user's research query (potentially enriched with clarifications).",
    )
    
    clarifications: Optional[List[str]] = Field(
        default=None,
        description="User responses to clarifying questions if any.",
    )
    
    preferences: Optional[Dict[str, str]] = Field(
        default=None,
        description="User preferences for the research (format, depth, sources, etc.).",
    )


class InstructionBuilderOutputSchema(OutputSchema):
    """Output schema for the InstructionBuilderAgent with detailed research instructions."""

    research_instructions: str = Field(
        ...,
        description="Detailed, structured research instructions for the deep research agent.",
    )
    
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="List of expected outputs/deliverables from the research.",
    )
    
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Key areas to focus on during research.",
    )
    
    output_format: Optional[str] = Field(
        default=None,
        description="Requested output format (report, table, comparison, etc.).",
    )


class TriageAgentInputSchema(InputSchema):
    """Input schema for the TriageAgent that routes queries."""

    query: str = Field(
        ...,
        description="The user's research query to triage.",
    )
    
    has_context: bool = Field(
        default=False,
        description="Whether the query already has sufficient context.",
    )


class TriageAgentOutputSchema(OutputSchema):
    """Output schema for the TriageAgent routing decision."""

    needs_clarification: bool = Field(
        ...,
        description="Whether the query needs clarification before research.",
    )
    
    routing_decision: Literal["clarify", "instruct", "direct_research"] = Field(
        ...,
        description="Where to route the query next.",
    )
    
    reasoning: str = Field(
        ...,
        description="Brief explanation of the routing decision.",
    )


class DeepResearchInputSchema(InputSchema):
    """Input schema for the DeepResearchAgent."""

    research_instructions: str = Field(
        ...,
        description="Detailed research instructions to execute.",
    )
    
    original_query: str = Field(
        ...,
        description="The original user query for reference.",
    )
    
    max_iterations: int = Field(
        default=5,
        description="Maximum number of research iterations.",
    )
    
    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Quality threshold for stopping research (0-1).",
    )
    
    output_format: Optional[str] = Field(
        default=None,
        description="Requested output format for the research.",
    )


class DeepResearchOutputSchema(OutputSchema):
    """Output schema for the DeepResearchAgent with comprehensive research results."""

    research_report: str = Field(
        ...,
        description="The comprehensive research report based on findings.",
    )
    
    key_findings: List[str] = Field(
        ...,
        description="List of key findings from the research.",
    )
    
    sources_consulted: List[str] = Field(
        ...,
        description="List of sources consulted during research.",
    )
    
    evidence_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score of the evidence found (0-1).",
    )
    
    gaps_identified: Optional[List[str]] = Field(
        default=None,
        description="Any gaps or limitations identified in the research.",
    )
    
    citations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of citations with title, url, relevant excerpts, and relevancy metadata.",
    )
    
    iterations_performed: int = Field(
        ...,
        description="Number of research iterations performed.",
    )
    
    research_trace: Optional[List[str]] = Field(
        default=None,
        description="Trace of research steps taken.",
    )


class ReportQualityValidatorInputSchema(InputSchema):
    """Input schema for report quality validation."""
    
    original_query: str = Field(..., description="Original research query")
    research_report: str = Field(..., description="Research report to validate")
    search_quality_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Quality metrics from search process"
    )
    weak_rubrics: List[str] = Field(
        default_factory=list, description="Weak rubrics from search analysis"
    )
    strong_rubrics: List[str] = Field(
        default_factory=list, description="Strong rubrics from search analysis"
    )
    evidence_quality_score: float = Field(
        default=0.5, description="Evidence quality score from search"
    )
    sources_consulted: List[str] = Field(
        default_factory=list, description="Sources used in research"
    )


class QualityDimension(BaseModel):
    """Individual quality assessment dimension."""
    
    dimension: str = Field(..., description="Quality dimension name")
    score: float = Field(..., description="Score for this dimension (0.0-1.0)")
    reasoning: str = Field(..., description="Detailed reasoning for the score")
    specific_issues: List[str] = Field(
        default_factory=list, description="Specific issues identified"
    )


class ReportQualityValidatorOutputSchema(OutputSchema):
    """Output schema for report quality validation."""
    
    validation_passed: bool = Field(..., description="Whether report passes quality validation")
    overall_quality_score: float = Field(..., description="Overall quality score (0.0-1.0)")
    
    # Quality dimensions
    query_alignment_score: float = Field(..., description="How well report addresses original query")
    structure_completeness_score: float = Field(..., description="Completeness of report structure")
    evidence_consistency_score: float = Field(..., description="Consistency between evidence and claims")
    gap_acknowledgment_score: float = Field(..., description="Whether report acknowledges limitations")
    
    # Detailed analysis
    quality_dimensions: List[QualityDimension] = Field(
        default_factory=list, description="Detailed quality assessment"
    )
    
    # Recommendations
    improvement_recommendations: List[str] = Field(
        default_factory=list, description="Specific recommendations for improvement"
    )
    missing_elements: List[str] = Field(
        default_factory=list, description="Elements missing from report"
    )
    
    # Decision factors
    trigger_deep_research: bool = Field(..., description="Whether to trigger DeepResearchAgent")
    decision_reasoning: str = Field(..., description="Reasoning for the decision")