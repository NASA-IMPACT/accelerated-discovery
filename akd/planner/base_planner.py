"""
Abstract base classes for workflow planners.

This module defines the interfaces that workflow planners must implement.
Concrete implementations (like LLMWorkflowPlanner) will be provided in future PRs.
"""

from abc import ABC, abstractmethod
from typing import Optional, TypedDict

from .structures import WorkflowPlan


class PlannerContext(TypedDict, total=False):
    """Context dictionary for planner operations."""

    previous_results: dict[str, object]
    constraints: list[str]
    user_preferences: dict[str, str | int | float | bool]
    session_id: Optional[str]


class AbstractWorkflowPlanner(ABC):
    """
    Abstract base class for workflow planners.

    A workflow planner is responsible for understanding user requirements and
    generating WorkflowPlan objects that can be executed by WorkflowBuilder.

    Concrete implementations might use:
    - LLM-based conversational planning (PR #3)
    - Rule-based planning
    - Template-based planning
    - Hybrid approaches
    """

    @abstractmethod
    async def plan_workflow(
        self, user_query: str, context: Optional[PlannerContext] = None
    ) -> WorkflowPlan:
        """
        Generate a WorkflowPlan from a user query.

        Args:
            user_query: Natural language description of the research task
            context: Optional context (previous results, constraints, etc.)

        Returns:
            WorkflowPlan ready to be passed to WorkflowBuilder

        Example:
            >>> planner = LLMWorkflowPlanner()
            >>> plan = await planner.plan_workflow(
            ...     "Find papers on AlphaFold and analyze gaps"
            ... )
            >>> # plan.suggested_agents = [deep_search, gap_analysis]
        """
        pass

    @abstractmethod
    async def refine_plan(
        self, plan: WorkflowPlan, feedback: str
    ) -> WorkflowPlan:
        """
        Refine an existing plan based on user feedback.

        Args:
            plan: Existing WorkflowPlan to refine
            feedback: User feedback or requested changes

        Returns:
            Refined WorkflowPlan

        Example:
            >>> plan = await planner.plan_workflow("Find AlphaFold papers")
            >>> refined = await planner.refine_plan(
            ...     plan, "Add code search after literature search"
            ... )
            >>> # refined.suggested_agents = [deep_search, code_search, gap_analysis]
        """
        pass

    @abstractmethod
    def validate_plan(self, plan: WorkflowPlan) -> tuple[bool, list[str]]:
        """
        Validate a WorkflowPlan for correctness and completeness.

        Args:
            plan: WorkflowPlan to validate

        Returns:
            Tuple of (is_valid, list_of_issues)

        Example:
            >>> is_valid, issues = planner.validate_plan(plan)
            >>> if not is_valid:
            ...     print(f"Plan has issues: {issues}")
        """
        pass


class AbstractInteractivePlanner(AbstractWorkflowPlanner):
    """
    Abstract base class for interactive conversational planners.

    Interactive planners support multi-turn conversations with users to
    clarify requirements, ask questions, and iteratively build plans.

    This is the interface that LLMWorkflowPlanner (PR #3) will implement.
    """

    @abstractmethod
    def start_session(self) -> "AbstractPlannerSession":
        """
        Start a new interactive planning session.

        Returns:
            PlannerSession object for managing the conversation

        Example:
            >>> planner = LLMWorkflowPlanner()
            >>> session = planner.start_session()
            >>> await session.send_message("I need to find papers on AlphaFold")
        """
        pass


class AbstractPlannerSession(ABC):
    """
    Abstract base class for interactive planning sessions.

    A session manages a single planning conversation with a user,
    maintaining state across multiple turns.
    """

    @abstractmethod
    async def send_message(self, message: str) -> str:
        """
        Send a message to the planner and get a response.

        Args:
            message: User's message or response to a question

        Returns:
            Planner's response message

        Example:
            >>> response = await session.send_message("Find papers on AlphaFold")
            >>> # "I'll help you search for papers. How many results do you need?"
        """
        pass

    @abstractmethod
    async def get_current_plan(self) -> Optional[WorkflowPlan]:
        """
        Get the current state of the workflow plan.

        Returns:
            Current WorkflowPlan if available, None if still gathering requirements

        Example:
            >>> plan = await session.get_current_plan()
            >>> if plan:
            ...     print(f"Current plan has {len(plan.suggested_agents)} agents")
        """
        pass

    @abstractmethod
    async def approve_plan(self) -> WorkflowPlan:
        """
        Approve the current plan and finalize it.

        Returns:
            Final WorkflowPlan ready for execution

        Raises:
            ValueError: If no plan is available to approve

        Example:
            >>> plan = await session.approve_plan()
            >>> # Plan is now final and can be passed to WorkflowBuilder
        """
        pass

    @abstractmethod
    async def reject_plan(self, reason: str) -> str:
        """
        Reject the current plan and provide feedback.

        Args:
            reason: Reason for rejection or requested changes

        Returns:
            Planner's response after receiving feedback

        Example:
            >>> response = await session.reject_plan(
            ...     "Add code search before gap analysis"
            ... )
        """
        pass

    @abstractmethod
    def get_conversation_history(self) -> list[dict[str, str]]:
        """
        Get the full conversation history.

        Returns:
            List of message dictionaries with 'role' and 'content'

        Example:
            >>> history = session.get_conversation_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        pass


# Type aliases for convenience
PlannerSession = AbstractPlannerSession
WorkflowPlanner = AbstractWorkflowPlanner
InteractivePlanner = AbstractInteractivePlanner
