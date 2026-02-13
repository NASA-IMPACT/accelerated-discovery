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
        assert len(workflow.edges) == 3  # START->deep_search_1, deep_search_1->gap_analysis_1, gap_analysis_1->END

    def test_workflow_has_correct_nodes(self, workflow_builder, simple_workflow_plan):
        """Test that workflow contains correct agent nodes with unique IDs."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        node_types = [node.type_ for node in workflow.nodes]
        node_ids = [node.id for node in workflow.nodes]

        assert "deep_search" in node_types
        assert "gap_analysis" in node_types
        assert "deep_search_1" in node_ids
        assert "gap_analysis_1" in node_ids

    def test_workflow_has_io_map(self, workflow_builder, simple_workflow_plan):
        """Test that workflow generates io_map for data routing."""
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        workflow = workflow_builder.build(simple_workflow_plan, filled_inputs)

        # gap_analysis should have io_map
        gap_node = next((n for n in workflow.nodes if n.type_ == "gap_analysis"), None)

        assert gap_node is not None
        assert gap_node.id == "gap_analysis_1"
        assert gap_node.io_map is not None
        assert len(gap_node.io_map) > 0

    def test_workflow_edges(self, workflow_builder, simple_workflow_plan):
        """Test that workflow generates correct edges using node IDs."""
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
        assert start_edges[0].to_node == "deep_search_1"
        assert end_edges[0].from_node == "gap_analysis_1"

    def test_duplicate_agent_types_get_unique_ids(self, workflow_builder):
        """Test that duplicate agent types get unique IDs."""
        plan = WorkflowPlan(
            workflow_description="Dual search workflow",
            research_goal="Search twice",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="First search",
                    confidence=1.0,
                ),
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Second search",
                    confidence=1.0,
                ),
            ],
        )

        filled_inputs = {
            "deep_search": {"query": "test"},
        }

        workflow = workflow_builder.build(plan, filled_inputs)

        assert len(workflow.nodes) == 2
        assert workflow.nodes[0].id == "deep_search_1"
        assert workflow.nodes[1].id == "deep_search_2"
        assert workflow.nodes[0].type_ == "deep_search"
        assert workflow.nodes[1].type_ == "deep_search"

        # Edges should use unique IDs
        assert workflow.edges[0].to_node == "deep_search_1"
        assert workflow.edges[1].from_node == "deep_search_1"
        assert workflow.edges[1].to_node == "deep_search_2"
        assert workflow.edges[2].from_node == "deep_search_2"


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

        # Verify nodes have both id and type
        for node in data["nodes"]:
            assert "id" in node
            assert "type" in node

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
        deep_node = next((n for n in workflow.nodes if n.type_ == "deep_search"), None)
        assert deep_node is not None
        assert deep_node.id == "deep_search_1"
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

        deep_node = next((n for n in workflow.nodes if n.type_ == "deep_search"), None)
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
        assert workflow.nodes[0].type_ == "deep_search"
        assert workflow.nodes[0].id == "deep_search_1"


class TestJSONPathValidation:
    """Test JSONPath expression validation in WorkflowBuilder."""

    def test_validate_jsonpath_with_valid_identifiers(self, workflow_builder):
        """Test that valid identifiers (alphanumeric + underscore) pass validation."""
        # Test with simple alphanumeric
        result = workflow_builder._build_and_validate_jsonpath("agent_a_1", "field1")
        assert result == "$.agent_a_1.outputs.field1"

        # Test with underscores
        result = workflow_builder._build_and_validate_jsonpath("deep_search_agent_1", "research_results")
        assert result == "$.deep_search_agent_1.outputs.research_results"

        # Test with numbers
        result = workflow_builder._build_and_validate_jsonpath("agent123", "field456")
        assert result == "$.agent123.outputs.field456"

        # Test with mixed case
        result = workflow_builder._build_and_validate_jsonpath("AgentA", "FieldB")
        assert result == "$.AgentA.outputs.FieldB"

    def test_validate_jsonpath_with_hyphenated_node_id(self, workflow_builder):
        """Test that node IDs with hyphens are rejected."""
        with pytest.raises(ValueError, match="Invalid node_id for JSONPath"):
            workflow_builder._build_and_validate_jsonpath("test-agent", "field")

    def test_validate_jsonpath_with_hyphenated_field_name(self, workflow_builder):
        """Test that field names with hyphens are rejected."""
        with pytest.raises(ValueError, match="Invalid field_name for JSONPath"):
            workflow_builder._build_and_validate_jsonpath("agent_1", "test-field")

    def test_validate_jsonpath_with_special_characters_in_node_id(self, workflow_builder):
        """Test that node IDs with special characters are rejected."""
        invalid_node_ids = [
            "agent@123",  # @ symbol
            "agent.name",  # period
            "agent$id",  # dollar sign
            "agent[0]",  # brackets
            "agent/path",  # forward slash
            "agent\\path",  # backslash
            "agent id",  # space
            "agent\nagent",  # newline
        ]

        for node_id in invalid_node_ids:
            with pytest.raises(ValueError, match="Invalid node_id for JSONPath"):
                workflow_builder._build_and_validate_jsonpath(node_id, "field")

    def test_validate_jsonpath_with_special_characters_in_field_name(self, workflow_builder):
        """Test that field names with special characters are rejected."""
        invalid_field_names = [
            "field[0]",  # brackets (array access)
            "field.name",  # period
            "field@name",  # @ symbol
            "field name",  # space
            "field/name",  # forward slash
            "field\\name",  # backslash
        ]

        for field_name in invalid_field_names:
            with pytest.raises(ValueError, match="Invalid field_name for JSONPath"):
                workflow_builder._build_and_validate_jsonpath("agent_1", field_name)

    def test_validate_jsonpath_with_empty_node_id(self, workflow_builder):
        """Test that empty node IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid node_id for JSONPath"):
            workflow_builder._build_and_validate_jsonpath("", "field")

    def test_validate_jsonpath_with_empty_field_name(self, workflow_builder):
        """Test that empty field names are rejected."""
        with pytest.raises(ValueError, match="Invalid field_name for JSONPath"):
            workflow_builder._build_and_validate_jsonpath("agent_1", "")

    def test_validate_jsonpath_prevents_path_injection(self, workflow_builder):
        """Test that path traversal attempts are blocked."""
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "..",
            ".",
            "$/malicious/path",
        ]

        for malicious_input in malicious_inputs:
            # Test in node_id
            with pytest.raises(ValueError, match="Invalid node_id for JSONPath"):
                workflow_builder._build_and_validate_jsonpath(malicious_input, "field")

            # Test in field_name
            with pytest.raises(ValueError, match="Invalid field_name for JSONPath"):
                workflow_builder._build_and_validate_jsonpath("agent_1", malicious_input)

    def test_validate_jsonpath_with_very_long_identifiers(self, workflow_builder):
        """Test that very long but valid identifiers work."""
        long_node_id = "a" * 100
        long_field_name = "f" * 100

        result = workflow_builder._build_and_validate_jsonpath(long_node_id, long_field_name)
        assert result == f"$.{long_node_id}.outputs.{long_field_name}"

    def test_validate_jsonpath_integration_with_build(self, workflow_builder, agent_registry):
        """Test that JSONPath validation is applied during workflow building."""
        # This test verifies that the validation is actually used in the build process
        # by testing with valid identifiers that should work

        plan = WorkflowPlan(
            workflow_description="Test workflow",
            research_goal="Test JSONPath validation",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="First agent",
                    confidence=0.95,
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Second agent",
                    confidence=0.90,
                ),
            ],
        )

        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {},
        }

        # Should build successfully with valid identifiers
        workflow = workflow_builder.build(plan, filled_inputs)
        assert len(workflow.nodes) == 2

        # Check that io_map was built with validated JSONPath using node IDs
        gap_node = next(node for node in workflow.nodes if node.type_ == "gap_analysis")
        assert gap_node.id == "gap_analysis_1"
        assert gap_node.io_map is not None
        # JSONPath should reference the node ID, not the type
        for jsonpath in gap_node.io_map.values():
            assert jsonpath.startswith("$.deep_search_1.outputs.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
