"""
Report Quality Validator Agent for evaluating research report quality against original queries.

This agent provides agentic validation of research reports using multiple quality dimensions
without requiring additional searches, avoiding chicken-and-egg problems.
"""

from typing import List, Optional
from pydantic import Field

from ._base import BaseAgentConfig, InstructorBaseAgent
from .schemas import (
    ReportQualityValidatorInputSchema,
    ReportQualityValidatorOutputSchema,
    QualityDimension,
)


class ReportQualityValidatorAgentConfig(BaseAgentConfig):
    """Configuration for the ReportQualityValidatorAgent."""

    model_name: str = "gpt-4o"
    temperature: float = 0.1  # Low temperature for consistent evaluation
    
    # Quality thresholds
    minimum_overall_score: float = Field(
        default=0.6, description="Minimum overall quality score to pass validation"
    )
    query_alignment_threshold: float = Field(
        default=0.7, description="Minimum query alignment score"
    )
    structure_completeness_threshold: float = Field(
        default=0.5, description="Minimum structure completeness score"
    )
    evidence_consistency_threshold: float = Field(
        default=0.6, description="Minimum evidence consistency score"
    )
    
    # Required report sections
    required_sections: List[str] = Field(
        default_factory=lambda: [
            "executive summary", "key findings", "analysis", 
            "evidence quality", "limitations", "recommendations"
        ],
        description="Required sections in research report"
    )


class ReportQualityValidatorAgent(
    InstructorBaseAgent[ReportQualityValidatorInputSchema, ReportQualityValidatorOutputSchema]
):
    """
    Agent that rigorously evaluates research report quality against original queries.
    
    This agent provides multi-dimensional quality assessment without requiring additional
    searches, using sophisticated analytical criteria to determine if a report adequately
    addresses the original research question.
    """

    input_schema = ReportQualityValidatorInputSchema
    output_schema = ReportQualityValidatorOutputSchema
    config_schema = ReportQualityValidatorAgentConfig

    def __init__(
        self,
        config: Optional[ReportQualityValidatorAgentConfig] = None,
        debug: bool = False,
    ):
        """Initialize the ReportQualityValidatorAgent."""
        config = config or ReportQualityValidatorAgentConfig()
        super().__init__(config, debug)

    async def _arun(
        self, 
        params: ReportQualityValidatorInputSchema
    ) -> ReportQualityValidatorOutputSchema:
        """
        Evaluate research report quality against the original query.
        
        Args:
            params: Input parameters containing original query, report, and context
            
        Returns:
            Comprehensive quality assessment with scores and recommendations
        """
        
        # Build comprehensive evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(params)
        
        # Generate structured quality assessment
        messages = [
            {
                "role": "system", 
                "content": """You are an expert research quality evaluator. Assess research reports against original queries using multiple quality dimensions. Be precise and objective in your analysis."""
            },
            {"role": "user", "content": evaluation_prompt}
        ]
        
        response = await self._generate_structured_response(
            messages=messages,
            response_model=ReportQualityValidatorOutputSchema
        )
        
        # Apply validation logic and determine final outcome
        validated_response = self._apply_validation_logic(response, params)
        
        return validated_response

    def _build_evaluation_prompt(self, params: ReportQualityValidatorInputSchema) -> str:
        """Build comprehensive context for quality evaluation."""
        
        prompt_sections = [
            "=== EVALUATION TASK ===",
            "Evaluate this research report against the original query using rigorous quality criteria.",
            "",
            "=== ORIGINAL RESEARCH QUERY ===",
            params.original_query,
            "",
            "=== RESEARCH REPORT TO EVALUATE ===",
            params.research_report,
            "",
            "=== SEARCH QUALITY CONTEXT ===",
            f"Evidence Quality Score: {params.evidence_quality_score}",
            f"Weak Rubrics from Search: {', '.join(params.weak_rubrics) if params.weak_rubrics else 'None'}",
            f"Strong Rubrics from Search: {', '.join(params.strong_rubrics) if params.strong_rubrics else 'None'}",
            f"Sources Consulted: {len(params.sources_consulted)}",
            "",
            "=== QUALITY DIMENSIONS TO ASSESS ===",
            "",
            "1. QUERY ALIGNMENT (0.0-1.0):",
            "   - Does the report directly address the original query?",
            "   - Are all aspects of the query covered?",
            "   - Is the focus maintained throughout?",
            "",
            "2. STRUCTURE COMPLETENESS (0.0-1.0):",
            f"   - Required sections: {', '.join(self.config.required_sections)}",
            "   - Is the report well-organized?",
            "   - Are all expected elements present?",
            "",
            "3. EVIDENCE CONSISTENCY (0.0-1.0):",
            "   - Are claims supported by evidence?",
            "   - Is evidence properly attributed?",
            "   - Are conclusions justified by findings?",
            "",
            "4. GAP ACKNOWLEDGMENT (0.0-1.0):",
            "   - Does the report acknowledge limitations?",
            "   - Are weak rubrics from search addressed?",
            "   - Are uncertainties clearly stated?",
            "",
            "=== INSTRUCTIONS ===",
            "Provide detailed assessment with specific examples.",
            "Consider the search quality context in your evaluation.",
            "Be precise about what triggers the need for DeepResearchAgent.",
            "Focus on actionable feedback for improvement.",
        ]
        
        return "\n".join(prompt_sections)

    def _apply_validation_logic(
        self,
        assessment: ReportQualityValidatorOutputSchema,
        params: ReportQualityValidatorInputSchema
    ) -> ReportQualityValidatorOutputSchema:
        """Apply configuration-based validation logic to determine final outcome."""
        
        # Calculate weighted overall score
        dimension_weights = {
            "query_alignment": 0.4,
            "structure_completeness": 0.2,
            "evidence_consistency": 0.3,
            "gap_acknowledgment": 0.1
        }
        
        weighted_score = (
            assessment.query_alignment_score * dimension_weights["query_alignment"] +
            assessment.structure_completeness_score * dimension_weights["structure_completeness"] +
            assessment.evidence_consistency_score * dimension_weights["evidence_consistency"] +
            assessment.gap_acknowledgment_score * dimension_weights["gap_acknowledgment"]
        )
        
        assessment.overall_quality_score = weighted_score
        
        # Check individual thresholds
        passes_query_alignment = assessment.query_alignment_score >= self.config.query_alignment_threshold
        passes_structure = assessment.structure_completeness_score >= self.config.structure_completeness_threshold
        passes_evidence = assessment.evidence_consistency_score >= self.config.evidence_consistency_threshold
        passes_overall = weighted_score >= self.config.minimum_overall_score
        
        # Consider search quality context
        search_quality_adequate = params.evidence_quality_score >= 0.6
        has_critical_weak_rubrics = any(
            weak in ["topic_alignment", "evidence_quality"] 
            for weak in params.weak_rubrics
        )
        
        # Decision logic
        validation_passed = (
            passes_query_alignment and 
            passes_structure and 
            passes_evidence and 
            passes_overall and
            search_quality_adequate and
            not has_critical_weak_rubrics
        )
        
        assessment.validation_passed = validation_passed
        assessment.trigger_deep_research = not validation_passed
        
        # Build decision reasoning
        failure_reasons = []
        if not passes_query_alignment:
            failure_reasons.append(f"Query alignment too low ({assessment.query_alignment_score:.2f} < {self.config.query_alignment_threshold})")
        if not passes_structure:
            failure_reasons.append(f"Structure incomplete ({assessment.structure_completeness_score:.2f} < {self.config.structure_completeness_threshold})")
        if not passes_evidence:
            failure_reasons.append(f"Evidence consistency poor ({assessment.evidence_consistency_score:.2f} < {self.config.evidence_consistency_threshold})")
        if not passes_overall:
            failure_reasons.append(f"Overall score insufficient ({weighted_score:.2f} < {self.config.minimum_overall_score})")
        if not search_quality_adequate:
            failure_reasons.append(f"Search quality inadequate ({params.evidence_quality_score:.2f} < 0.6)")
        if has_critical_weak_rubrics:
            failure_reasons.append(f"Critical weak rubrics: {[r for r in params.weak_rubrics if r in ['topic_alignment', 'evidence_quality']]}")
        
        if failure_reasons:
            assessment.decision_reasoning = f"Triggering DeepResearchAgent due to: {'; '.join(failure_reasons)}"
            assessment.improvement_recommendations.extend([
                f"CRITICAL: {reason}" for reason in failure_reasons
            ])
        else:
            assessment.decision_reasoning = "Report meets all quality criteria - no further research needed"
        
        return assessment