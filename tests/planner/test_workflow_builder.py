"""
Pytest tests for WorkflowBuilder module.

Tests workflow generation, validation, and JSON serialization.
"""

import json

import pytest

from akd.planner.format_builder import WorkflowFormat
from akd.planner.registry import AgentRegistry, get_agent_registry
from akd.planner.structures import AgentSuggestion, WorkflowPlan
from akd.planner.workflow_builder import WorkflowBuilder


@pytest.fixture
def agent_registry():
    """Shared agent registry fixture."""
    return get_agent_registry()


@pytest.fixture
def workflow_builder(agent_registry):
    """Shared workflow builder fixture."""
    return WorkflowBuilder(agent_registry)


@pytest.fixture
def reset_singleton():
    """Reset AgentRegistry singleton after each test."""
    yield
    AgentRegistry._reset_singleton()


@pytest.fixture
def simple_workflow_plan():
    """Simple two-agent workflow plan."""
    return WorkflowPlan(
        workflow_description="Search for papers and analyze gaps",
        research_goal="Find papers on AlphaFold and identify research gaps",
        suggested_agents=[
            AgentSuggestion(
                agent_id="deep_search",
                agent_name="Deep Search",
                reason="Search literature for papers",
                confidence=0.95,
            ),
            AgentSuggestion(
                agent_id="gap_analysis",
                agent_name="Gap Analysis",
                reason="Analyze papers to find research gaps",
                confidence=0.90,
            ),
        ],
        workflow_steps=[
            "Search literature for papers",
            "Analyze papers to identify gaps",
        ],
        potential_issues=[],
    )


class TestWorkflowBuilderCore:
    """Test core WorkflowBuilder functionality."""

    def test_builder_initialization(self, agent_registry):
        """Test that WorkflowBuilder initializes correctly."""
        builder = WorkflowBuilder(agent_registry)

        assert builder.registry is agent_registry
        assert builder.mapping_registry is not None

    def test_check_missing_agents(self, workflow_builder, simple_workflow_plan):
        """Test agent existence validation."""
        missing = workflow_builder.check_missing_agents(simple_workflow_plan)

        assert isinstance(missing, list)
        assert len(missing) == 0  # All agents should exist

    def test_check_agents_missing(self, workflow_builder):
        """Test detection of missing agents."""
        plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="nonexistent_agent",
                    agent_name="Nonexistent",
                    reason="Testing",
                    confidence=1.0,
                ),
            ],
        )

        missing = workflow_builder.check_missing_agents(plan)

        assert len(missing) == 1
        assert "nonexistent_agent" in missing


class TestWorkflowGeneration:
    """Test workflow generation."""

    def test_build_simple_workflow(self, workflow_builder, simple_workflow_plan):
        """Test building a simple two-agent workflow."""
        filled_inputs = {
            "deep_search": {
                "query": "AlphaFold protein structure prediction",
                "max_results": 20,
            },
            "gap_analysis": {
                # Will be filled by io_map at runtime
            },
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        assert isinstance(workflow, WorkflowFormat)
        assert len(workflow.nodes) == 2
        assert len(workflow.edges) == 3  # START->deep_search, deep_search->gap_analysis, gap_analysis->END

    def test_workflow_has_correct_nodes(self, workflow_builder, simple_workflow_plan):
        """Test that workflow contains correct agent nodes."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        node_types = [node.type for node in workflow.nodes]

        assert "deep_search" in node_types
        assert "gap_analysis" in node_types

    def test_workflow_has_io_map(self, workflow_builder, simple_workflow_plan):
        """Test that workflow generates io_map for data routing."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # gap_analysis should have io_map
        gap_node = next((n for n in workflow.nodes if n.type == "gap_analysis"), None)

        assert gap_node is not None
        assert gap_node.io_map is not None
        assert len(gap_node.io_map) > 0

    def test_workflow_edges(self, workflow_builder, simple_workflow_plan):
        """Test that workflow generates correct edges."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # Check START and END edges exist
        start_edges = [e for e in workflow.edges if e.from_node == "START"]
        end_edges = [e for e in workflow.edges if e.to_node == "END"]

        assert len(start_edges) == 1
        assert len(end_edges) == 1
        assert start_edges[0].to_node == "deep_search"
        assert end_edges[0].from_node == "gap_analysis"


class TestWorkflowSerialization:
    """Test workflow JSON serialization."""

    def test_workflow_to_json(self, workflow_builder, simple_workflow_plan):
        """Test that workflow can be serialized to JSON."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)
        workflow_json = workflow.to_json()

        # Validate JSON
        assert isinstance(workflow_json, str)
        data = json.loads(workflow_json)

        assert "workflow_type" in data
        assert "nodes" in data
        assert "edges" in data
        assert data["workflow_type"] == "AKDResearchWorkflow"

    def test_workflow_model_dump(self, workflow_builder, simple_workflow_plan):
        """Test that workflow can be dumped to dict."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)
        data = workflow.model_dump(by_alias=True, exclude_none=True)

        assert isinstance(data, dict)
        assert "workflow_type" in data
        assert "nodes" in data
        assert "edges" in data

    def test_workflow_save_to_file(self, workflow_builder, simple_workflow_plan, tmp_path):
        """Test saving workflow to file."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # Save to temporary file
        output_file = tmp_path / "test_workflow.json"
        workflow.save_to_file(str(output_file))

        # Verify file exists and is valid JSON
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["workflow_type"] == "AKDResearchWorkflow"
        assert len(data["nodes"]) == 2


class TestWorkflowInputHandling:
    """Test workflow input handling."""

    def test_empty_inputs(self, workflow_builder, simple_workflow_plan):
        """Test building workflow with empty inputs."""
        filled_inputs = {}

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # Should still build workflow, but nodes may have empty inputs
        assert len(workflow.nodes) == 2

    def test_partial_inputs(self, workflow_builder, simple_workflow_plan):
        """Test building workflow with partial inputs."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            # gap_analysis inputs omitted
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # Should build successfully
        assert len(workflow.nodes) == 2

        # deep_search node should have input
        deep_node = next((n for n in workflow.nodes if n.type == "deep_search"), None)
        assert deep_node is not None
        assert len(deep_node.input.fields) > 0

    def test_input_fields_preserved(self, workflow_builder, simple_workflow_plan):
        """Test that input fields are preserved in workflow."""
        filled_inputs = {
            "deep_search": {
                "query": "AlphaFold protein structure",
                "max_results": 20,
            },
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        deep_node = next((n for n in workflow.nodes if n.type == "deep_search"), None)
        assert deep_node is not None

        # Check that input fields contain the values
        input_dict = {}
        for field in deep_node.input.fields:
            if isinstance(field, dict):
                input_dict.update(field)

        assert "query" in input_dict
        assert "max_results" in input_dict
        assert input_dict["query"] == "AlphaFold protein structure"
        assert input_dict["max_results"] == 20


class TestWorkflowValidation:
    """Test workflow validation."""

    def test_empty_plan(self, workflow_builder):
        """Test building workflow with empty plan."""
        plan = WorkflowPlan(
            workflow_description="Empty",
            research_goal="Empty",
            suggested_agents=[],
        )

        workflow = workflow_builder.build(plan, {})

        assert len(workflow.nodes) == 0
        assert len(workflow.edges) == 0

    def test_single_agent_plan(self, workflow_builder):
        """Test building workflow with single agent."""
        plan = WorkflowPlan(
            workflow_description="Single agent",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Testing",
                    confidence=1.0,
                ),
            ],
        )

        filled_inputs = {"deep_search": {"query": "test"}}
        workflow = workflow_builder.build(plan, filled_inputs)

        assert len(workflow.nodes) == 1
        assert workflow.nodes[0].type == "deep_search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
