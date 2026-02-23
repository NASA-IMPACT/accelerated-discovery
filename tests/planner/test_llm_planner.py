"""
Unit tests for the LLM-based workflow planner.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from akd.planner.config import AgentRegistryConfig
from akd.planner.llm_planner import (
    LLMWorkflowPlanner,
    InteractivePlannerSession,
    ConversationPhase,
    PlannerResponse,
    create_planner,
    quick_plan,
)
from akd.planner.structures import WorkflowPlan, AgentSuggestion
from akd.planner.registry import AgentRegistry


@pytest.fixture
def mock_registry():
    """Create a mock agent registry."""
    registry = MagicMock()
    registry.get_enabled_agents.return_value = []
    return registry


@pytest.fixture
def reset_singleton():
    """Reset AgentRegistry singleton after each test."""
    yield
    AgentRegistry._reset_singleton()


class TestLLMWorkflowPlanner:
    """Test LLMWorkflowPlanner class."""

    @pytest.mark.asyncio
    async def test_planner_initialization(self, mock_registry, reset_singleton):
        """Test planner initialization with custom config."""
        config = AgentRegistryConfig(auto_discover=False)
        planner = LLMWorkflowPlanner(registry=mock_registry)

        assert planner.registry is mock_registry
        assert planner.builder is not None
        assert planner.mapping_registry is not None
        assert planner.mapping_generator is not None

    @pytest.mark.asyncio
    async def test_planner_has_system_prompt(self, mock_registry, reset_singleton):
        """Test that planner generates system prompt from registry."""
        planner = LLMWorkflowPlanner(registry=mock_registry)
        system_prompt = planner._get_planner_system_prompt()

        assert "workflow planner" in system_prompt.lower()
        assert "available agents" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_create_planner_function(self, reset_singleton):
        """Test create_planner convenience function."""
        with patch('akd.planner.llm_planner.get_agent_registry') as mock_get_registry:
            mock_get_registry.return_value = MagicMock()
            planner = await create_planner()

            assert isinstance(planner, LLMWorkflowPlanner)
            assert planner.registry is not None


class TestInteractivePlannerSession:
    """Test InteractivePlannerSession class."""

    @pytest.fixture
    def mock_planner(self, mock_registry):
        """Create a mock planner for session testing."""
        planner = LLMWorkflowPlanner(registry=mock_registry)
        return planner

    @pytest.mark.asyncio
    async def test_session_initialization(self, mock_planner):
        """Test session initialization."""
        session = InteractivePlannerSession(
            mock_planner,
            "Find papers on AlphaFold"
        )

        assert session.initial_request == "Find papers on AlphaFold"
        assert session.current_phase == ConversationPhase.INITIAL_REQUIREMENTS
        assert session.conversation_history == []
        assert session.workflow_plan is None
        assert session.final_workflow is None

    @pytest.mark.asyncio
    async def test_session_state_update(self, mock_planner):
        """Test that session updates state correctly."""
        session = InteractivePlannerSession(
            mock_planner,
            "Test request"
        )

        response = PlannerResponse(
            message="Test response",
            phase=ConversationPhase.AGENT_SELECTION,
            ready_to_generate=False
        )

        session._update_session_state("User message", response)

        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
        assert session.conversation_history[0]["content"] == "User message"
        assert session.conversation_history[1]["role"] == "assistant"
        assert session.conversation_history[1]["content"] == "Test response"
        assert session.current_phase == ConversationPhase.AGENT_SELECTION

    @pytest.mark.asyncio
    async def test_session_workflow_plan_capture(self, mock_planner):
        """Test that session captures workflow plan from response."""
        session = InteractivePlannerSession(
            mock_planner,
            "Test request"
        )

        workflow_plan = WorkflowPlan(
            workflow_description="Test workflow",
            research_goal="Test goal",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )

        response = PlannerResponse(
            message="Generated plan",
            phase=ConversationPhase.FINALIZATION,
            workflow_plan=workflow_plan,
            ready_to_generate=True
        )

        session._update_session_state("Generate", response)

        assert session.workflow_plan is not None
        assert session.workflow_plan.research_goal == "Test goal"

    @pytest.mark.asyncio
    async def test_session_conversation_summary(self, mock_planner):
        """Test conversation summary generation."""
        session = InteractivePlannerSession(
            mock_planner,
            "Find AlphaFold papers"
        )

        workflow_plan = WorkflowPlan(
            workflow_description="Literature search",
            research_goal="Find AlphaFold papers",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search papers",
                    confidence=0.9
                )
            ]
        )

        session.workflow_plan = workflow_plan
        summary = session.get_conversation_summary()

        assert "Find AlphaFold papers" in summary
        assert "Deep Search" in summary


class TestWorkflowPlan:
    """Test WorkflowPlan model."""

    def test_workflow_plan_creation(self):
        """Test creating a workflow plan."""
        plan = WorkflowPlan(
            workflow_description="Test workflow",
            research_goal="Test goal",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="test_agent",
                    agent_name="Test Agent",
                    reason="Testing",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results"]
                )
            ],
            workflow_steps=["Step 1", "Step 2"],
            potential_issues=["None"]
        )

        assert plan.research_goal == "Test goal"
        assert len(plan.suggested_agents) == 1
        assert plan.suggested_agents[0].agent_id == "test_agent"

    def test_workflow_plan_defaults(self):
        """Test workflow plan with default values."""
        plan = WorkflowPlan(
            workflow_description="Minimal plan",
            research_goal="Minimal goal"
        )

        assert plan.suggested_agents == []
        assert plan.workflow_steps == []
        assert plan.potential_issues == []


class TestAgentSuggestion:
    """Test AgentSuggestion model."""

    def test_agent_suggestion_creation(self):
        """Test creating an agent suggestion."""
        suggestion = AgentSuggestion(
            agent_id="deep_search",
            agent_name="Deep Search",
            reason="Literature search needed",
            confidence=0.95,
            required_inputs=["query", "max_results"],
            expected_outputs=["results", "synthesis"],
            depends_on=None
        )

        assert suggestion.agent_id == "deep_search"
        assert suggestion.confidence == 0.95
        assert len(suggestion.required_inputs) == 2
        assert len(suggestion.expected_outputs) == 2

    def test_agent_suggestion_with_dependencies(self):
        """Test agent suggestion with dependencies."""
        suggestion = AgentSuggestion(
            agent_id="gap_analysis",
            agent_name="Gap Analysis",
            reason="Analyze gaps",
            confidence=0.9,
            depends_on=["deep_search"]
        )

        assert suggestion.depends_on == ["deep_search"]


class TestPlannerIntegration:
    """Integration tests for planner with real components."""

    @pytest.mark.asyncio
    async def test_planner_with_real_registry(self, reset_singleton):
        """Test planner with real agent registry."""
        # Use real registry with limited agents
        config = AgentRegistryConfig(
            auto_discover=True,
            use_agents=["deep_search"]
        )
        registry = AgentRegistry(config)

        planner = LLMWorkflowPlanner(registry=registry)

        # Verify planner has access to registry agents
        agents = planner.registry.get_enabled_agents()
        assert len(agents) >= 1
        assert any(a.agent_id == "deep_search" for a in agents)

    @pytest.mark.asyncio
    async def test_quick_plan_function(self, reset_singleton):
        """Test quick_plan convenience function."""
        with patch('akd.planner.llm_planner.get_agent_registry') as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get_enabled_agents.return_value = []
            mock_get_registry.return_value = mock_registry

            session = await quick_plan("Find papers on protein folding")

            assert isinstance(session, InteractivePlannerSession)
            assert session.initial_request == "Find papers on protein folding"


class TestPlannerErrorHandling:
    """Test error handling in planner."""

    @pytest.mark.asyncio
    async def test_generate_workflow_without_plan(self, mock_registry):
        """Test that generate_workflow fails without a plan."""
        planner = LLMWorkflowPlanner(registry=mock_registry)
        session = InteractivePlannerSession(planner, "Test")

        with pytest.raises(ValueError, match="No workflow plan available"):
            await session.generate_workflow()

    @pytest.mark.asyncio
    async def test_generate_workflow_with_missing_agents(self, mock_registry):
        """Test workflow generation with missing agents."""
        mock_registry.get_agent.return_value = None

        planner = LLMWorkflowPlanner(registry=mock_registry)
        session = InteractivePlannerSession(planner, "Test")

        # Set a workflow plan with non-existent agent
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="nonexistent_agent",
                    agent_name="Nonexistent",
                    reason="Testing",
                    confidence=1.0
                )
            ]
        )

        with pytest.raises(ValueError, match="not found in registry"):
            await session.generate_workflow()


class TestSessionReadiness:
    """Test InteractivePlannerSession.is_ready_to_generate() deterministic check."""

    @pytest.fixture
    def mock_planner(self, mock_registry):
        """Create a mock planner for session testing."""
        planner = LLMWorkflowPlanner(registry=mock_registry)
        return planner

    def test_not_ready_without_plan(self, mock_planner):
        """No workflow plan: not ready."""
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = None
        assert session.is_ready_to_generate() is False

    def test_not_ready_with_empty_agents(self, mock_planner):
        """Workflow plan exists but has no agents: not ready."""
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Empty plan",
            research_goal="Test",
            suggested_agents=[],
        )
        assert session.is_ready_to_generate() is False

    def test_not_ready_without_research_goal(self, mock_planner):
        """Workflow plan has agents but no research goal: not ready."""
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )
        assert session.is_ready_to_generate() is False

    def test_not_ready_with_missing_registry_agent(self, mock_planner):
        """Agent in plan not found in registry: not ready."""
        mock_planner.registry.get_agent.return_value = None
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test goal",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="nonexistent_agent",
                    agent_name="Nonexistent",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )
        assert session.is_ready_to_generate() is False

    def test_ready_with_valid_plan(self, mock_planner):
        """Complete valid plan with all agents in registry: ready."""
        mock_planner.registry.get_agent.return_value = MagicMock()  # Agent exists
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Find papers on AlphaFold",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )
        assert session.is_ready_to_generate() is True

    def test_ready_with_multiple_agents(self, mock_planner):
        """Multiple agents, all in registry: ready."""
        mock_planner.registry.get_agent.return_value = MagicMock()
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Literature review",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search",
                    confidence=0.95,
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Analyze gaps",
                    confidence=0.9,
                    depends_on=["deep_search"],
                ),
            ],
        )
        assert session.is_ready_to_generate() is True

    def test_planner_response_no_longer_corrects_flags(self):
        """PlannerResponse no longer auto-corrects ready_to_generate.
        Whatever value is set stays — session overrides it later."""
        # ready=True with no plan — PlannerResponse should NOT correct it
        response = PlannerResponse(
            message="Workflow is ready",
            phase=ConversationPhase.FINALIZATION,
            workflow_plan=None,
            ready_to_generate=True,
        )
        # No validator → stays True (session will override before returning)
        assert response.ready_to_generate is True

        # ready=False with valid plan — PlannerResponse should NOT correct it
        plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )
        response2 = PlannerResponse(
            message="Workflow is ready and will be generated",
            phase=ConversationPhase.FINALIZATION,
            workflow_plan=plan,
            question=None,
            ready_to_generate=False,
        )
        # No validator: stays False (session will override before returning)
        assert response2.ready_to_generate is False

    def test_session_overrides_ready_when_no_plan(self, mock_planner):
        """LLM says ready=True but plan is None: session overrides to False.
        This is the exact bug that caused 'No workflow plan available' crashes."""
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = None

        response = PlannerResponse(
            message="Workflow is ready!",
            phase=ConversationPhase.FINALIZATION,
            workflow_plan=None,
            question=None,
            ready_to_generate=True,  # LLM incorrectly says ready
        )

        # Simulate what start()/respond() does after _update_session_state
        response.ready_to_generate = (
            session.is_ready_to_generate() and response.question is None
        )

        assert response.ready_to_generate is False

    def test_not_ready_when_question_pending(self, mock_planner):
        """Plan is structurally complete but LLM still asking a question: not ready."""
        from akd.planner.llm_planner import PlannerQuestion, PlannerQuestionType

        mock_planner.registry.get_agent.return_value = MagicMock()
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Find papers",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                )
            ],
        )
        # is_ready_to_generate() is True on its own
        assert session.is_ready_to_generate() is True

        # But combined with pending question → not ready
        question = PlannerQuestion(
            question="What time range?",
            question_type=PlannerQuestionType.OPEN_ENDED,
            context="Need clarification",
        )
        ready = session.is_ready_to_generate() and question is None
        assert ready is False

    def test_not_ready_with_partial_registry_match(self, mock_planner):
        """Two agents in plan, only one exists in registry: not ready."""
        mock_planner.registry.get_agent.side_effect = (
            lambda agent_id: MagicMock() if agent_id == "deep_search" else None
        )
        session = InteractivePlannerSession(mock_planner, "Test")
        session.workflow_plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Literature review",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search",
                    confidence=0.95,
                ),
                AgentSuggestion(
                    agent_id="nonexistent_agent",
                    agent_name="Nonexistent",
                    reason="Testing",
                    confidence=0.9,
                ),
            ],
        )
        assert session.is_ready_to_generate() is False


class TestConversationPhases:
    """Test conversation phase transitions."""

    def test_all_phases_defined(self):
        """Test that all conversation phases are defined."""
        phases = [
            ConversationPhase.INITIAL_REQUIREMENTS,
            ConversationPhase.GOAL_CLARIFICATION,
            ConversationPhase.AGENT_SELECTION,
            ConversationPhase.IO_SPECIFICATION,
            ConversationPhase.WORKFLOW_CONSTRUCTION,
            ConversationPhase.VALIDATION,
            ConversationPhase.FINALIZATION,
        ]

        assert len(phases) == 7
        # Verify all are unique
        assert len(set(phases)) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
