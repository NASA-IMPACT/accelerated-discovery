"""Test SingleAgentNodeTemplate functionality."""

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.nodes.states import GlobalState, NodeState
from akd.nodes.templates import SingleAgentNodeTemplate


# Test schemas
class TestAgentInputSchema(InputSchema):
    """Test agent input schema."""

    query: str = Field(..., description="User query")


class TestAgentOutputSchema(OutputSchema):
    """Test agent output schema."""

    response: str = Field(..., description="Agent response")


# Test agent
class TestAgent(InstructorBaseAgent[TestAgentInputSchema, TestAgentOutputSchema]):
    """Test agent for testing."""

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

        assert template.agent is not None
        assert template._input_schema == TestAgentInputSchema
        assert template._output_schema == TestAgentOutputSchema

    def test_init_invalid_agent(self):
        """Test initialization with invalid agent."""
        with pytest.raises(TypeError, match="agent must be an instance of BaseAgent"):
            SingleAgentNodeTemplate(agent="not an agent")

    def test_init_agent_without_schemas(self):
        """Test initialization with agent missing schemas."""
        agent = TestAgent()
        agent.input_schema = None  # type: ignore

        with pytest.raises(ValueError, match="must have an input_schema"):
            SingleAgentNodeTemplate(agent=agent)

    @pytest.mark.asyncio
    async def test_execute_basic_agent(self):
        """Test executing node template with basic agent."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, mutation=False)

        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(
                    inputs={"query": "Hello world"},
                ),
            },
        )

        result = await template.arun(global_state)

        assert "response" in result.outputs
        assert result.outputs["response"] == "Processed: Hello world"


class TestSingleAgentNodeTemplateEdgeCases:
    """Test edge cases and error conditions."""

    def test_init_none_guardrails(self):
        """Test initialization with None values for guardrails."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            input_guardrail=None,
            output_guardrail=None,
        )

        assert template.agent is not None

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

        global_state = GlobalState(
            node_states={
                template.node_id: NodeState(inputs={}),
            },
        )

        with pytest.raises(Exception):  # Pydantic validation error
            await template.arun(global_state)

    @pytest.mark.asyncio
    async def test_execute_missing_node_state(self):
        """Test executing with missing node state."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, mutation=True)

        global_state = GlobalState(node_states={})

        with pytest.raises(Exception):
            await template.arun(global_state)


class TestSingleAgentNodeTemplateCrossNodeMapping:
    """Test cross-node input mapping with JSONPath functionality."""

    @pytest.mark.asyncio
    async def test_resolve_inputs_no_io_map(self):
        """Test _resolve_inputs with complete inputs and no io_map."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent)

        node_state = NodeState(inputs={"query": "test query"})
        global_state = GlobalState(node_states={"test": node_state})

        resolved = await template._resolve_inputs(node_state, global_state)
        assert resolved == {"query": "test query"}

    @pytest.mark.asyncio
    async def test_resolve_inputs_no_io_map_missing_required(self):
        """Test _resolve_inputs fails when required inputs missing and no io_map."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(agent=agent, node_id="test_node")

        node_state = NodeState(inputs={})
        global_state = GlobalState(node_states={"test_node": node_state})

        with pytest.raises(
            ValueError,
            match=r"(?s)Node.*missing required inputs.*Consider using io_map",
        ):
            await template._resolve_inputs(node_state, global_state)

    @pytest.mark.asyncio
    async def test_resolve_inputs_simple_jsonpath_mapping(self):
        """Test _resolve_inputs with simple JSONPath cross-node mapping."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            io_map={"query": "$.lit_search.inputs.query"},
            debug=True,
        )

        node_state = NodeState(inputs={})
        global_state = GlobalState(
            node_states={
                "current": node_state,
                "lit_search": NodeState(inputs={"query": "landslide nepal"}),
            },
        )

        resolved = await template._resolve_inputs(node_state, global_state)
        assert resolved["query"] == "landslide nepal"

    @pytest.mark.asyncio
    async def test_resolve_inputs_outputs_mapping(self):
        """Test _resolve_inputs mapping from outputs instead of inputs."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            io_map={"query": "$.search.outputs.final_query"},
            debug=True,
        )

        node_state = NodeState(inputs={})
        global_state = GlobalState(
            node_states={
                "current": node_state,
                "search": NodeState(outputs={"final_query": "earthquake detection"}),
            },
        )

        resolved = await template._resolve_inputs(node_state, global_state)
        assert resolved["query"] == "earthquake detection"

    @pytest.mark.asyncio
    async def test_resolve_inputs_no_override_existing(self):
        """Test _resolve_inputs never overrides existing node inputs."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            io_map={"query": "$.other.inputs.query"},
            debug=True,
        )

        node_state = NodeState(inputs={"query": "original query"})
        global_state = GlobalState(
            node_states={
                "current": node_state,
                "other": NodeState(inputs={"query": "should not override"}),
            },
        )

        resolved = await template._resolve_inputs(node_state, global_state)
        assert resolved["query"] == "original query"

    @pytest.mark.asyncio
    async def test_resolve_inputs_missing_source_node(self):
        """Test _resolve_inputs handles missing source node gracefully."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            io_map={"query": "$.nonexistent.inputs.query"},
            debug=True,
        )

        node_state = NodeState(inputs={})
        global_state = GlobalState(node_states={"current": node_state})

        with pytest.raises(ValueError, match="validation failed"):
            await template._resolve_inputs(node_state, global_state)

    @pytest.mark.asyncio
    async def test_resolve_inputs_nested_jsonpath(self):
        """Test _resolve_inputs with nested JSONPath expressions."""
        agent = TestAgent()
        template = SingleAgentNodeTemplate(
            agent=agent,
            io_map={"query": "$.config.inputs.search_params.query_text"},
            debug=True,
        )

        node_state = NodeState(inputs={})
        global_state = GlobalState(
            node_states={
                "current": node_state,
                "config": NodeState(
                    inputs={
                        "search_params": {
                            "query_text": "nested query value",
                        },
                    },
                ),
            },
        )

        resolved = await template._resolve_inputs(node_state, global_state)
        assert resolved["query"] == "nested query value"


# Prevent pytest from collecting test agent class
TestAgent.__test__ = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
