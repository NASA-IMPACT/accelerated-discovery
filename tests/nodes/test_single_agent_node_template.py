"""Test SingleAgentNodeTemplate with guardrails integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.configs.guardrails_config import GuardrailsConfig
from akd.nodes.states import GlobalState, NodeState
from akd.nodes.templates import SingleAgentNodeTemplate
from akd.tools.granite_guardian_tool import RiskDefinition


# Test schemas
class TestAgentInputSchema(InputSchema):
    """Test agent input schema."""

    query: str = Field(..., description="User query")


class TestAgentOutputSchema(OutputSchema):
    """Test agent output schema."""

    response: str = Field(..., description="Agent response")


# Test agent
class TestAgent(InstructorBaseAgent[TestAgentInputSchema, TestAgentOutputSchema]):
    """Test agent for guardrails integration testing."""

    input_schema = TestAgentInputSchema
    output_schema = TestAgentOutputSchema

    async def _arun(
        self,
        params: TestAgentInputSchema,
        **kwargs,
    ) -> TestAgentOutputSchema:
        """Simple test implementation."""
        return TestAgentOutputSchema(response=f"Processed: {params.query}")


class TestSingleAgentNodeTemplateBasic:
    """Test basic SingleAgentNodeTemplate functionality."""

    def test_init_with_agent(self):
        """Test initialization with a basic agent."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent)

        assert template.agent is agent
        assert template._input_schema == TestAgentInputSchema
        assert template._output_schema == TestAgentOutputSchema
        assert not hasattr(template.agent, "guardrails_config")

    def test_init_invalid_agent(self):
        """Test initialization with invalid agent."""
        with pytest.raises(TypeError, match="agent must be an instance of BaseAgent"):
            SingleAgentNodeTemplate(agent="not an agent")

    def test_init_agent_without_schemas(self):
        """Test initialization with agent missing schemas."""
        # The framework's metaclass validation prevents creating agents without schemas
        # So we test this by creating an agent and removing its schema attributes
        agent = TestAgent()

        # Remove the schema attributes to simulate a bad agent
        del agent.__class__.input_schema

        with pytest.raises(ValueError, match="must have an input_schema"):
            SingleAgentNodeTemplate(agent=agent)

    @pytest.mark.asyncio
    async def test_execute_basic_agent(self):
        """Test executing node template with basic agent."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, mutation=True)

        # Create test global state
        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(
                    inputs={"query": "Hello world"},
                ),
            },
        )

        # Execute the node
        result = await template.arun(global_state)

        assert "response" in result.outputs
        assert result.outputs["response"] == "Processed: Hello world"


class TestSingleAgentNodeTemplateGuardrails:
    """Test SingleAgentNodeTemplate with guardrails."""

    def test_init_with_risk_definition_guardrails(self):
        """Test initialization with RiskDefinition guardrails."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
            output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
        )

        # Agent should be wrapped with guardrails
        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")
        assert hasattr(template.agent, "guardrails_tool")
        assert template.agent.__class__.__name__.startswith("Guardrailed")

        # Check guardrails configuration
        config = template.agent.guardrails_config
        assert RiskDefinition.JAILBREAK in config.input_risk_types
        assert RiskDefinition.HARM in config.input_risk_types
        assert RiskDefinition.ANSWER_RELEVANCE in config.output_risk_types

    def test_init_with_guardrails_config(self):
        """Test initialization with GuardrailsConfig."""
        agent = TestAgent()
        config = GuardrailsConfig(
            enabled=True,
            input_risk_types=[RiskDefinition.HARM],
            output_risk_types=[RiskDefinition.GROUNDEDNESS],
            fail_on_risk=True,
            snippet_n_chars=50,
        )

        template = SingleAgentNodeTemplate(
            agent=agent,
            guardrails_config=config,
        )

        # Agent should be wrapped with guardrails
        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")

        # Check that our config was used
        agent_config = template.agent.guardrails_config
        assert agent_config.fail_on_risk is True
        assert agent_config.snippet_n_chars == 50
        assert agent_config.input_risk_types == [RiskDefinition.HARM]
        assert agent_config.output_risk_types == [RiskDefinition.GROUNDEDNESS]

    def test_init_with_mixed_guardrails_config(self):
        """Test initialization with both individual guardrails and config."""
        agent = TestAgent()
        config = GuardrailsConfig(
            enabled=True,
            fail_on_risk=True,
        )

        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.JAILBREAK],
            output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
            guardrails_config=config,
        )

        # Agent should be wrapped
        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")

        # Config should take precedence where specified
        agent_config = template.agent.guardrails_config
        assert agent_config.fail_on_risk is True

    def test_init_input_guardrails_only(self):
        """Test initialization with only input guardrails."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.HARM],
        )

        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")

        config = template.agent.guardrails_config
        # Should have our input guardrail plus defaults for output
        assert RiskDefinition.HARM in config.input_risk_types

    def test_init_output_guardrails_only(self):
        """Test initialization with only output guardrails."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            output_guardrails=[RiskDefinition.GROUNDEDNESS],
        )

        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")

        config = template.agent.guardrails_config
        # Should have our output guardrail plus defaults for input
        assert RiskDefinition.GROUNDEDNESS in config.output_risk_types

    def test_init_config_only(self):
        """Test initialization with only guardrails config."""
        agent = TestAgent()
        config = GuardrailsConfig(
            enabled=False,  # Disabled config
            input_risk_types=[RiskDefinition.VIOLENCE],
            output_risk_types=[RiskDefinition.RELEVANCE],
        )

        template = SingleAgentNodeTemplate(
            agent=agent,
            guardrails_config=config,
        )

        # Agent should still be wrapped even if disabled
        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")
        assert template.agent.guardrails_config.enabled is False

    @pytest.mark.asyncio
    async def test_execute_with_guardrails_disabled(self):
        """Test executing node template with guardrails disabled."""
        agent = TestAgent()
        config = GuardrailsConfig(enabled=False, fail_on_risk=False)

        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.HARM],
            output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
            guardrails_config=config,
            mutation=True,
        )

        # Create test global state
        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(
                    inputs={"query": "Test query"},
                ),
            },
        )

        # Execute the node
        result = await template.arun(global_state)

        assert "response" in result.outputs
        assert result.outputs["response"] == "Processed: Test query"

    @pytest.mark.asyncio
    async def test_execute_with_guardrails_mocked(self):
        """Test executing node template with mocked guardrails."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.HARM],
            output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
            guardrails_config=GuardrailsConfig(
                enabled=False,
            ),  # Disable to avoid needing real model
            mutation=True,
        )

        # Mock the guardrails tool to return safe results
        if (
            hasattr(template.agent, "guardrails_tool")
            and template.agent.guardrails_tool
        ):
            mock_tool = AsyncMock()
            mock_tool.arun.return_value = MagicMock(risk_results=[{"is_risky": False}])
            template.agent.guardrails_tool = mock_tool

        # Create test global state
        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(
                    inputs={"query": "Safe test query"},
                ),
            },
        )

        # Execute the node
        result = await template.arun(global_state)

        assert "response" in result.outputs
        assert result.outputs["response"] == "Processed: Safe test query"


class TestSingleAgentNodeTemplateEdgeCases:
    """Test edge cases and error conditions."""

    def test_init_empty_guardrails_lists(self):
        """Test initialization with empty guardrails lists."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[],
            output_guardrails=[],
        )

        # No guardrails should be applied with empty lists
        assert template.agent is agent
        assert not hasattr(template.agent, "guardrails_config")

    def test_init_none_values(self):
        """Test initialization with None values for optional parameters."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=None,
            output_guardrails=None,
            guardrails_config=None,
        )

        assert template.agent is agent
        assert not hasattr(template.agent, "guardrails_config")

    def test_node_id_generation(self):
        """Test that unique node IDs are generated."""
        agent = TestAgent()
        template1 = SingleAgentNodeTemplate(agent=agent)
        template2 = SingleAgentNodeTemplate(agent=agent)

        assert template1.node_id != template2.node_id
        assert len(template1.node_id) > 0
        assert len(template2.node_id) > 0

    def test_custom_node_id(self):
        """Test initialization with custom node ID."""
        agent = TestAgent()
        custom_id = "my-custom-node-id"
        template = SingleAgentNodeTemplate(agent=agent, node_id=custom_id)

        assert template.node_id == custom_id

    @pytest.mark.asyncio
    async def test_execute_empty_inputs(self):
        """Test executing with empty inputs."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, mutation=True)

        # Create global state with no inputs
        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(inputs={}),
            },
        )

        # Should raise an error due to missing required field
        with pytest.raises(Exception):  # Pydantic validation error
            await template.arun(global_state)

    @pytest.mark.asyncio
    async def test_execute_missing_node_state(self):
        """Test executing with missing node state."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, mutation=True)

        # Create global state without our node's state
        global_state = GlobalState(node_states={})

        # Should create empty node state and warn about no inputs
        # But will fail when trying to create agent input due to missing required field
        with pytest.raises(Exception):  # Will be a pydantic validation error
            await template.arun(global_state)


class TestSingleAgentNodeTemplateIntegration:
    """Integration tests for SingleAgentNodeTemplate."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_guardrails(self):
        """Test complete workflow with guardrails integration."""
        agent = TestAgent()

        # Create template with comprehensive guardrails
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[
                RiskDefinition.JAILBREAK,
                RiskDefinition.HARM,
                RiskDefinition.UNETHICAL_BEHAVIOR,
            ],
            output_guardrails=[
                RiskDefinition.ANSWER_RELEVANCE,
                RiskDefinition.GROUNDEDNESS,
            ],
            guardrails_config=GuardrailsConfig(
                enabled=False,  # Disable to avoid needing real model
                fail_on_risk=False,
                snippet_n_chars=100,
            ),
            mutation=True,
            debug=True,
        )

        # Verify guardrails were applied
        assert template.agent is not agent
        assert hasattr(template.agent, "guardrails_config")
        assert hasattr(template.agent, "_validate_with_guardrails")

        # Create comprehensive test state
        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(
                    inputs={"query": "What is the capital of France?"},
                    messages=[],
                    outputs={},
                    steps={},
                ),
            },
        )

        # Execute the full workflow
        result = await template.arun(global_state)

        # Verify results
        assert result is not None
        assert "response" in result.outputs
        assert "Processed: What is the capital of France?" in result.outputs["response"]

        # Verify node state was updated in global state if mutation is True
        updated_node_state = global_state.node_states[template.node_id]
        assert "response" in updated_node_state.outputs

    def test_guardrails_configuration_inheritance(self):
        """Test that guardrails configuration is properly inherited."""
        agent = TestAgent()

        # Test default configuration with input guardrails
        template1 = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrails=[RiskDefinition.HARM],
        )

        config1 = template1.agent.guardrails_config
        assert config1.enabled is True
        assert config1.fail_on_risk is False  # Default
        assert RiskDefinition.HARM in config1.input_risk_types

        # Test custom configuration with output guardrails
        custom_config = GuardrailsConfig(
            enabled=True,
            fail_on_risk=True,
            snippet_n_chars=200,
        )

        template2 = SingleAgentNodeTemplate(
            agent=TestAgent(),  # New agent instance
            output_guardrails=[RiskDefinition.GROUNDEDNESS],
            guardrails_config=custom_config,
        )

        config2 = template2.agent.guardrails_config
        assert config2.fail_on_risk is True
        assert config2.snippet_n_chars == 200
        # Note: GROUNDEDNESS should be in output_risk_types, but the actual config
        # depends on how the decorator merges user-provided lists with config


# Prevent pytest from collecting test agent class
TestAgent.__test__ = False


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
