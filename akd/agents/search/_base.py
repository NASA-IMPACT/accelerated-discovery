"""
Base classes and shared utilities for literature search agents.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd._base.streaming import RunningEvent, RunningEventData
from akd._base.tool_calling import RunContext
from akd.agents._base import BaseAgent, BaseAgentConfig
from akd.structures import SearchResult
from akd.tools.reranker import (
    RerankerTool,
    RerankerToolConfig,
    RerankerType,
    create_reranker,
)

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
    results: list[SearchResult] = Field(..., description="List of search results")
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
        description="Maximum number of search results to retrieve by the agent (hard limit). This is not used for capping search tool results, which is controlled by SearchMode or 'search_max_results' from kwargs.",
    )

    # Reranker configuration
    reranker_type: RerankerType = Field(
        default="none",
        description="The type of reranker to use for combining results from multiple search tools.",
    )
    reranker_config: RerankerToolConfig = Field(
        default_factory=lambda: RerankerToolConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L12-v2",
        ),
        description="Configuration for the reranker tool.",
    )


class SearchAgent[TInput: SearchAgentInputSchema, TOutput: SearchAgentOutputSchema](
    BaseAgent[TInput, TOutput],
):
    """
    Base agent for performing literature searches using a search tool.

    Notes:
    - By default `answer` is auto-generated using `akd.agents.search.answer.QuestionAnsweringAgent`.
    - Subclasses must implement `_generate_report()` to provide custom report generation logic.
    """

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
        self.reranker: RerankerTool = create_reranker(
            reranker_type=self.config.reranker_type,
            config=self.config.reranker_config,
            debug=self.debug,
        )

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
        search_results: list[SearchResult],
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
        results: list[SearchResult],
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

    def _emit_step_event(
        self,
        step: str,
        message: str,
        run_context: RunContext,
        step_index: int | None = None,
        total_steps: int | None = None,
        substep: str | None = None,
        **data_kwargs: Any,
    ) -> RunningEvent:
        """Create a RUNNING event for a pipeline step.

        Args:
            step: Step identifier (e.g., "triage", "research.search")
            message: Human-readable progress message
            run_context: Execution context with run_id, etc.
            step_index: Current step number (1-based)
            total_steps: Total number of main steps
            substep: Sub-step identifier for nested progress
            **data_kwargs: Additional data to include in event payload

        Returns:
            RunningEvent with step information
        """
        data = {
            "step": step,
            **data_kwargs,
        }
        if step_index is not None:
            data["step_index"] = step_index
        if total_steps is not None:
            data["total_steps"] = total_steps
        if substep is not None:
            data["substep"] = substep

        return RunningEvent(
            source=self.__class__.__name__,
            message=message,
            data=RunningEventData(**data),
            run_context=run_context,
        )

    def _format_search_summary(
        self,
        total_results: int,
        iterations: int,
        quality_score: float = 0.0,
    ) -> str:
        """Format a standardized search summary."""
        return f"Literature search completed: {total_results} results in {iterations} iterations (quality: {quality_score:.2f})"
