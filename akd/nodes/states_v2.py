import operator
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from akd.structures import SearchResultItem
from akd.common_types import ToolType as Tool
from akd.utils import LANGCHAIN_CORE_INSTALLED

if TYPE_CHECKING or LANGCHAIN_CORE_INSTALLED:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
else:
    BaseMessage = BaseModel


class AgentStatus(str, Enum):
    """Status of an agent in the workflow"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep(BaseModel):
    """Represents a single step in the research plan"""

    step_id: str = Field(..., description="Unique identifier for the step")
    agent_type: str = Field(..., description="Type of agent to execute this step")
    description: str = Field(
        ...,
        description="Human-readable description of what this step does",
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inputs required for this step",
    )
    outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs produced by this step",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Step IDs that must complete before this step",
    )
    status: AgentStatus = Field(
        default=AgentStatus.PENDING,
        description="Current status of this step",
    )
    priority: int = Field(
        default=0,
        description="Priority level (higher = more important)",
    )
    estimated_duration: Optional[int] = Field(
        None,
        description="Estimated duration in minutes",
    )
    actual_duration: Optional[int] = Field(
        None,
        description="Actual duration in minutes",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if step failed",
    )
    human_review_required: bool = Field(
        default=False,
        description="Whether this step requires human review",
    )


class ResearchPlan(BaseModel):
    """Complete research plan with steps and metadata"""

    plan_id: str = Field(..., description="Unique identifier for the plan")
    title: str = Field(..., description="Title of the research plan")
    description: str = Field(..., description="Description of the research goals")
    steps: Dict[str, PlanStep] = Field(
        default_factory=dict,
        description="Steps in the plan",
    )
    current_step_id: Optional[str] = Field(None, description="Currently executing step")
    completed_steps: List[str] = Field(
        default_factory=list,
        description="List of completed step IDs",
    )
    blocked_steps: List[str] = Field(
        default_factory=list,
        description="List of blocked step IDs",
    )
    human_feedback: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Human feedback and instructions",
    )
    created_at: Optional[str] = Field(None, description="Plan creation timestamp")
    updated_at: Optional[str] = Field(None, description="Plan last update timestamp")

    def get_next_steps(self) -> List[PlanStep]:
        """Get all steps that are ready to execute (dependencies met, status pending)"""
        ready_steps = []
        for step in self.steps.values():
            if step.status == AgentStatus.PENDING and all(
                dep_id in self.completed_steps for dep_id in step.dependencies
            ):
                ready_steps.append(step)
        return sorted(ready_steps, key=lambda x: x.priority, reverse=True)


class HumanInteraction(BaseModel):
    """Represents a human interaction point in the workflow"""

    interaction_id: str = Field(
        ...,
        description="Unique identifier for the interaction",
    )
    interaction_type: Literal["approval", "feedback", "input", "decision", "review"] = (
        Field(..., description="Type of human interaction required")
    )
    prompt: str = Field(..., description="Prompt or question for the human")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context information for the human",
    )
    response: Optional[Dict[str, Any]] = Field(None, description="Human response")
    timestamp: Optional[str] = Field(None, description="Timestamp of the interaction")
    status: Literal["pending", "completed", "skipped"] = Field(default="pending")
    timeout_minutes: Optional[int] = Field(
        None,
        description="Timeout for this interaction",
    )


class LiteratureSearchState(BaseModel):
    """State specific to literature search operations"""

    search_queries: List[str] = Field(
        default_factory=list,
        description="Generated search queries",
    )
    search_results: List[SearchResultItem] = Field(
        default_factory=list,
        description="All search results",
    )
    filtered_results: List[SearchResultItem] = Field(
        default_factory=list,
        description="Relevancy-filtered results",
    )
    extraction_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted information",
    )
    search_iteration: int = Field(default=0, description="Current search iteration")
    relevancy_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevancy scores by URL",
    )
    search_strategy: str = Field(
        default="initial",
        description="Current search strategy",
    )


class ExtractionState(BaseModel):
    """State specific to structured extraction operations"""

    extraction_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for extraction",
    )
    raw_content: List[str] = Field(
        default_factory=list,
        description="Raw content to extract from",
    )
    extracted_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted structured data",
    )
    extraction_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores",
    )
    validation_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation results",
    )


class PlannerState(BaseModel):
    """State specific to the planner agent"""

    research_goal: str = Field(..., description="High-level research goal")
    research_domain: str = Field(default="general", description="Research domain/field")
    current_plan: Optional[ResearchPlan] = Field(
        None,
        description="Current active research plan",
    )
    plan_history: List[ResearchPlan] = Field(
        default_factory=list,
        description="History of plans",
    )
    available_agents: List[str] = Field(
        default_factory=list,
        description="Available agent types",
    )
    agent_capabilities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Capabilities of each agent type",
    )
    planning_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context for planning decisions",
    )


class ToolSearchResult(BaseModel):
    """
    Hold the tool search result
    """

    tool: Optional[Tool] = Field(
        ...,
        description="Tool found during search",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="When tool is found, what arguments extracted?",
    )
    result: Optional[Any] = Field(
        default=None,
        description="When tool is executed, set this field",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        return getattr(self.tool, "name", self.tool.__class__.__name__)


class NodeState(BaseModel):
    """Fields common to all node-like states."""

    if LANGCHAIN_CORE_INSTALLED:
        messages: Annotated[List[Union[BaseMessage, Dict[str, Any]]], add_messages] = (
            Field(default_factory=list)
        )
    else:
        messages: List[Union[BaseMessage, Dict[str, Any]]] = Field(default_factory=list)

    inputs: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


class SupervisorState(NodeState):
    """Extra fields that only the Supervisor cares about."""

    tool_calls: List[ToolSearchResult] = Field(default_factory=list)
    steps: Dict[str, Any] = Field(default_factory=dict)


class NodeTemplateState(NodeState):
    supervisor_state: SupervisorState = Field(default_factory=SupervisorState)
    input_guardrails: Dict[str, Any] = Field(default_factory=dict)
    output_guardrails: Dict[str, Any] = Field(default_factory=dict)


class GlobalWorkflowState(NodeState):
    """
    Enhanced global state for planner-driven multi-agent workflows.
    This is the main state that flows through the LangGraph.
    """

    # Core workflow state
    workflow_id: str = Field(
        ...,
        description="Unique identifier for this workflow instance",
    )
    workflow_status: Literal[
        "initializing",
        "planning",
        "executing",
        "waiting_human",
        "completed",
        "failed",
    ] = Field(default="initializing", description="Current workflow status")

    # Planner state
    planner_state: PlannerState = Field(..., description="State of the planner agent")

    # Agent-specific states
    literature_search_state: LiteratureSearchState = Field(
        default_factory=LiteratureSearchState,
    )
    extraction_state: ExtractionState = Field(default_factory=ExtractionState)

    # Node states (existing pattern)
    node_states: Dict[str, NodeTemplateState] = Field(default_factory=dict)

    # Human interaction
    if LANGCHAIN_CORE_INSTALLED:
        human_interactions: Annotated[List[HumanInteraction], operator.add] = Field(
            default_factory=list,
        )
    else:
        human_interactions: List[HumanInteraction] = Field(default_factory=list)

    pending_human_interaction: Optional[HumanInteraction] = Field(
        None,
        description="Current pending human interaction",
    )

    # Research artifacts
    research_artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated research artifacts and results",
    )

    # Workflow metadata
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of executed steps",
    )
    error_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered during execution",
    )

    # Context and configuration
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences and settings",
    )
    workflow_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow configuration",
    )

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the currently executing step"""
        if (
            self.planner_state.current_plan
            and self.planner_state.current_plan.current_step_id
        ):
            return self.planner_state.current_plan.steps.get(
                self.planner_state.current_plan.current_step_id,
            )
        return None

    def get_ready_steps(self) -> List[PlanStep]:
        """Get all steps ready for execution"""
        if self.planner_state.current_plan:
            return self.planner_state.current_plan.get_next_steps()
        return []

    def add_human_interaction(self, interaction: HumanInteraction) -> None:
        """Add a new human interaction"""
        self.human_interactions.append(interaction)
        if interaction.status == "pending":
            self.pending_human_interaction = interaction
            self.workflow_status = "waiting_human"

    def complete_human_interaction(
        self,
        interaction_id: str,
        response: Dict[str, Any],
    ) -> bool:
        """Complete a human interaction with response"""
        for interaction in self.human_interactions:
            if interaction.interaction_id == interaction_id:
                interaction.response = response
                interaction.status = "completed"
                if (
                    self.pending_human_interaction
                    and self.pending_human_interaction.interaction_id == interaction_id
                ):
                    self.pending_human_interaction = None
                    self.workflow_status = "executing"
                return True
        return False


# Legacy alias for backward compatibility
GlobalState = GlobalWorkflowState
