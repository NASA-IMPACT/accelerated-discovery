"""
Base classes and shared utilities for literature search agents.
"""

from typing import List
from pydantic import BaseModel, Field
from akd.agents._base import BaseAgentConfig
from akd._base import InputSchema, OutputSchema, AbstractBase


class LitSearchAgentInputSchema(InputSchema):
    """Base input schema for literature search agents."""
    
    query: str = Field(..., description="Research query to search for")
    max_results: int = Field(default=20, description="Maximum number of results to return")
    category: str = Field(default="science", description="Search category")


class LitSearchAgentOutputSchema(OutputSchema):
    """Base output schema for literature search agents."""
    
    results: List[dict] = Field(..., description="List of search results")
    category: str = Field(..., description="Search category")
    iterations_performed: int = Field(default=1, description="Number of search iterations performed")


class LitSearchAgentConfig(BaseAgentConfig):
    """Base configuration for literature search agents."""
    
    debug: bool = Field(default=False, description="Enable debug logging")
    max_iterations: int = Field(default=5, description="Maximum search iterations")


class RubricAnalysis(BaseModel):
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


class StoppingCriteria(BaseModel):
    """Criteria for determining when to stop iterative search."""
    
    stop_now: bool = Field(default=False)
    reasoning_trace: str = Field(default="")
    rubric_analysis: RubricAnalysis = Field(default_factory=RubricAnalysis)
    recommended_query_focus: List[str] = Field(default_factory=list)


class LitBaseAgent(AbstractBase[LitSearchAgentInputSchema, LitSearchAgentOutputSchema]):
    """
    Abstract base class for literature search agents.
    
    Provides common functionality for all literature search agents including:
    - Standard input/output schemas
    - Common configuration handling
    - Shared utility methods for literature search workflows
    - Consistent error handling and logging patterns
    """
    
    input_schema = LitSearchAgentInputSchema
    output_schema = LitSearchAgentOutputSchema
    config_schema = LitSearchAgentConfig
    
    def __init__(self, config: LitSearchAgentConfig = None, debug: bool = False, **kwargs):
        """Initialize the literature search agent with common setup."""
        super().__init__(config=config, debug=debug, **kwargs)
    
    def _validate_query(self, query: str) -> str:
        """Validate and clean the input query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()
    
    def _should_continue_search(self, iteration: int, quality_score: float = 0.0) -> bool:
        """Determine if search should continue based on iteration and quality."""
        max_iterations = getattr(self.config, 'max_iterations', 5)
        quality_threshold = getattr(self.config, 'quality_threshold', 0.7)
        
        if iteration >= max_iterations:
            return False
        if quality_score >= quality_threshold:
            return False
        return True
    
    def _format_search_summary(self, total_results: int, iterations: int, quality_score: float = 0.0) -> str:
        """Format a standardized search summary."""
        return f"Literature search completed: {total_results} results in {iterations} iterations (quality: {quality_score:.2f})"