"""
Tests for abstract planner base classes.

This file tests that the abstract interfaces are properly defined.
Concrete implementations will be tested in PR #3.
"""

import pytest

from akd.planner import (
    AbstractInteractivePlanner,
    AbstractPlannerSession,
    AbstractWorkflowPlanner,
    PlannerConfig,
)
from akd.planner.structures import WorkflowPlan


class TestPlannerConfig:
    """Test PlannerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PlannerConfig()

        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_conversation_turns == 10
        assert config.auto_approve_high_confidence is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlannerConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_conversation_turns=5,
            auto_approve_high_confidence=True,
        )

        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_conversation_turns == 5
        assert config.auto_approve_high_confidence is True


class TestAbstractInterfaces:
    """Test that abstract interfaces are properly defined."""

    def test_abstract_workflow_planner_has_required_methods(self):
        """Test that AbstractWorkflowPlanner defines required methods."""
        required_methods = ["plan_workflow", "refine_plan", "validate_plan"]

        for method_name in required_methods:
            assert hasattr(AbstractWorkflowPlanner, method_name)
            method = getattr(AbstractWorkflowPlanner, method_name)
            # Check it's marked as abstract
            assert hasattr(method, "__isabstractmethod__")
            assert method.__isabstractmethod__ is True

    def test_abstract_interactive_planner_extends_workflow_planner(self):
        """Test that AbstractInteractivePlanner extends AbstractWorkflowPlanner."""
        assert issubclass(AbstractInteractivePlanner, AbstractWorkflowPlanner)

    def test_abstract_interactive_planner_has_session_method(self):
        """Test that AbstractInteractivePlanner has start_session method."""
        assert hasattr(AbstractInteractivePlanner, "start_session")
        method = getattr(AbstractInteractivePlanner, "start_session")
        assert hasattr(method, "__isabstractmethod__")
        assert method.__isabstractmethod__ is True

    def test_abstract_session_has_required_methods(self):
        """Test that AbstractPlannerSession defines required methods."""
        required_methods = [
            "send_message",
            "get_current_plan",
            "approve_plan",
            "reject_plan",
            "get_conversation_history",
        ]

        for method_name in required_methods:
            assert hasattr(AbstractPlannerSession, method_name)
            method = getattr(AbstractPlannerSession, method_name)
            assert hasattr(method, "__isabstractmethod__")
            assert method.__isabstractmethod__ is True

    def test_cannot_instantiate_abstract_planner(self):
        """Test that abstract classes cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractWorkflowPlanner()

        with pytest.raises(TypeError):
            AbstractInteractivePlanner()

        with pytest.raises(TypeError):
            AbstractPlannerSession()


class TestConcreteImplementationExample:
    """Test that a concrete implementation would work correctly."""

    def test_minimal_concrete_planner(self):
        """Test that a minimal concrete implementation can be created."""

        class MinimalPlanner(AbstractWorkflowPlanner):
            """Minimal concrete implementation for testing."""

            async def plan_workflow(self, user_query, context=None):
                return WorkflowPlan(
                    workflow_description="Test workflow",
                    research_goal="Test goal",
                    suggested_agents=[],
                )

            async def refine_plan(self, plan, feedback):
                return plan

            def validate_plan(self, plan):
                return (True, [])

        # Should be able to instantiate
        planner = MinimalPlanner()
        assert isinstance(planner, AbstractWorkflowPlanner)

    def test_minimal_interactive_planner(self):
        """Test that a minimal interactive planner can be created."""

        class MinimalSession(AbstractPlannerSession):
            """Minimal session for testing."""

            async def send_message(self, message):
                return "Test response"

            async def get_current_plan(self):
                return None

            async def approve_plan(self):
                return WorkflowPlan(
                    workflow_description="Test",
                    research_goal="Test",
                    suggested_agents=[],
                )

            async def reject_plan(self, reason):
                return "Plan rejected"

            def get_conversation_history(self):
                return []

        class MinimalInteractivePlanner(AbstractInteractivePlanner):
            """Minimal interactive planner for testing."""

            async def plan_workflow(self, user_query, context=None):
                return WorkflowPlan(
                    workflow_description="Test",
                    research_goal="Test",
                    suggested_agents=[],
                )

            async def refine_plan(self, plan, feedback):
                return plan

            def validate_plan(self, plan):
                return (True, [])

            def start_session(self):
                return MinimalSession()

        # Should be able to instantiate
        planner = MinimalInteractivePlanner()
        assert isinstance(planner, AbstractInteractivePlanner)
        assert isinstance(planner, AbstractWorkflowPlanner)

        # Should be able to start a session
        session = planner.start_session()
        assert isinstance(session, AbstractPlannerSession)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
