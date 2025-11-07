"""
Base classes and shared utilities for literature search agents.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig
from akd.structures import SearchResultItem

from .answer import QuestionAnsweringAgent, QuestionAnsweringAgentOutputSchema


class SearchMode(str, Enum):
    """Search mode determining the depth and breadth of search."""

    FAST = "fast"  # 10 results - quick overview
    MEDIUM = "medium"  # 20 results - balanced search
    LONG = "long"  # 50 results - comprehensive search
    EXTENSIVE = "extensive"  # 100 results - exhaustive search

    def to_max_results(self) -> int:
        """Convert search mode to maximum number of results."""
        mapping = {
            SearchMode.FAST: 10,
            SearchMode.MEDIUM: 50,
            SearchMode.LONG: 100,
            SearchMode.EXTENSIVE: 200,
        }
        return mapping[self]


class SearchAgentInputSchema(InputSchema):
    """Base input schema for literature search agents."""

    query: str = Field(..., description="Query to search for. Can be question or subject/topic of interest.")
    search_mode: SearchMode = Field(
        default=SearchMode.MEDIUM,
        description="Search mode determining depth and breadth of search",
    )
    additional_context: str | None = Field(default=None, description="Additional context for the search agent")


class SearchAgentOutputSchema(OutputSchema):
    """Base output schema for literature search agents."""

    __response_field__ = "report"

    answer: str = Field(..., description="Concise shortform answer to the research query in few sentences.")
    report: str | None = Field(default=None, description="Detailed report pertaining to the research query.")
    results: list[SearchResultItem] = Field(..., description="List of search results")
    iterations_performed: int = Field(
        default=1,
        description="Number of search iterations performed",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata and synthesis information",
    )


class SearchAgentConfig(BaseAgentConfig):
    """Base configuration for literature search agents."""

    debug: bool = Field(default=False, description="Enable debug logging")
    max_iterations: int = Field(
        default=5,
        description="Maximum number of search iterations",
    )
    # used to limit results per iteration
    max_results: int = Field(
        default=50,
        description="Maximum number of search results to retrieve by the agent (hard limit). This is not used for capping search tool results, which is controlled by SearchMode or 'max_results' from kwargs.",
    )


class SearchAgent[TInput: SearchAgentInputSchema, TOutput: SearchAgentOutputSchema](
    BaseAgent[TInput, TOutput],
):
    """Base agent for performing literature searches using a search tool."""

    input_schema = SearchAgentInputSchema
    output_schema = SearchAgentOutputSchema
    config_schema = SearchAgentConfig

    def __init__(
        self,
        answer_agent: QuestionAnsweringAgent | None = None,
        config: SearchAgentConfig | None = None,
        debug: bool = False,
    ):
        super().__init__(config=config, debug=debug)
        self.answer_agent = answer_agent or QuestionAnsweringAgent()

    async def get_response_async(
        self,
        *args,
        **kwargs,
    ) -> TOutput:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Optional[OutputSchema]):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            OutputSchema: The response from the language model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def _generate_answer(
        self,
        query: str,
        search_results: List[SearchResultItem],
        additional_context: str | None = None,
        **kwargs,
    ) -> QuestionAnsweringAgentOutputSchema:
        """
        Generate a concise shortform answer from search results.

        Subclasses must implement this method to provide custom answer
        generation logic (e.g., using an LLM).

        Args:
            query: The original research query
            results: List of search results
            **kwargs: Additional keyword arguments (e.g., additional_context)

        Returns:
            A concise shortform answer to the query
        """
        return await self.answer_agent.arun(
            self.answer_agent.input_schema(
                query=query,
                search_results=search_results,
                additional_context=additional_context,
            ),
        )

    @abstractmethod
    async def _generate_report(
        self,
        query: str,
        results: List[SearchResultItem],
        **kwargs,
    ) -> str:
        """
        Generate a detailed research report from search results.

        Subclasses must implement this method to provide custom report
        generation logic (e.g., using an LLM or synthesis agent).

        Args:
            query: The original research query
            results: List of search results
            **kwargs: Additional keyword arguments (e.g., additional_context, research_report)

        Returns:
            A detailed research report
        """
        raise NotImplementedError("Subclasses must implement _generate_report()")


class LitSearchAgentInputSchema(SearchAgentInputSchema):
    """Base input schema for literature search agents."""

    pass


class LitSearchAgentOutputSchema(SearchAgentOutputSchema):
    """Base output schema for literature search agents."""

    report: str | None = Field(
        default=None,
        description="Synthesized research report from the literature search",
    )


class LitSearchAgentConfig(SearchAgentConfig):
    """Base configuration for literature search agents."""

    pass


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


class LitBaseAgent(SearchAgent[LitSearchAgentInputSchema, LitSearchAgentOutputSchema]):
    """
    Abstract base class for literature search agents.

    Provides common functionality for all literature search agents including:
    - Standard input/output schemas
    - Common configuration handling
    - Shared utility methods for literature search workflows
    - Consistent error handling and logging patterns
    """

    input_schema = SearchAgentInputSchema
    output_schema = SearchAgentOutputSchema
    config_schema = SearchAgentConfig

    def _validate_query(self, query: str) -> str:
        """Validate and clean the input query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()

    def _should_continue_search(
        self,
        iteration: int,
        quality_score: float = 0.0,
    ) -> bool:
        """Determine if search should continue based on iteration and quality."""
        max_iterations = getattr(self.config, "max_iterations", 5)
        quality_threshold = getattr(self.config, "quality_threshold", 0.7)

        if iteration >= max_iterations:
            return False
        if quality_score >= quality_threshold:
            return False
        return True

    def _format_search_summary(
        self,
        total_results: int,
        iterations: int,
        quality_score: float = 0.0,
    ) -> str:
        """Format a standardized search summary."""
        return f"Literature search completed: {total_results} results in {iterations} iterations (quality: {quality_score:.2f})"
