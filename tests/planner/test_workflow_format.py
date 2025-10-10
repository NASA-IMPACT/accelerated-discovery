"""
Unit tests for WorkflowFormat validation.
"""

import pytest

from akd.planner.format_builder import WorkflowEdge, WorkflowFormat, WorkflowNode, WorkflowNodeIO


class TestWorkflowFormatValidation:
    """Test WorkflowFormat validation methods."""

    def test_validate_io_map_invalid_node_reference(self):
        """Test that validation detects invalid node references in io_map."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(
                    type="agent_b",
                    input=WorkflowNodeIO(),
                    io_map={"field1": "$.non_existent_agent.outputs.field"},
                ),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        issues = workflow.validate_io_map(strict=False)
        assert len(issues) > 0
        assert any("non_existent_agent" in issue for issue in issues)

    def test_validate_io_map_invalid_jsonpath_format(self):
        """Test that validation detects invalid JSONPath format."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(
                    type="agent_b",
                    input=WorkflowNodeIO(),
                    io_map={"field1": "invalid_path"},
                ),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        issues = workflow.validate_io_map(strict=False)
        assert len(issues) > 0
        assert any("must start with '$." in issue for issue in issues)

    def test_validate_io_map_valid_workflow(self):
        """Test that validation passes for valid io_map."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(
                    type="agent_b",
                    input=WorkflowNodeIO(),
                    io_map={"field1": "$.agent_a.outputs.result"},
                ),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        issues = workflow.validate_io_map(strict=False)
        assert len(issues) == 0

    def test_validate_edges_orphaned_node(self):
        """Test that validation detects orphaned nodes."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(type="agent_b", input=WorkflowNodeIO()),
                WorkflowNode(type="orphaned_agent", input=WorkflowNodeIO()),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        issues = workflow.validate_edges(strict=False)
        assert len(issues) > 0
        assert any("orphaned" in issue.lower() for issue in issues)

    def test_validate_edges_missing_start(self):
        """Test that validation detects missing START edge."""
        workflow = WorkflowFormat(
            nodes=[WorkflowNode(type="agent_a", input=WorkflowNodeIO())],
            edges=[WorkflowEdge(from_node="agent_a", to_node="END")],
        )

        issues = workflow.validate_edges(strict=False)
        assert len(issues) > 0
        assert any("START" in issue for issue in issues)

    def test_validate_edges_missing_end(self):
        """Test that validation detects missing END edge."""
        workflow = WorkflowFormat(
            nodes=[WorkflowNode(type="agent_a", input=WorkflowNodeIO())],
            edges=[WorkflowEdge(from_node="START", to_node="agent_a")],
        )

        issues = workflow.validate_edges(strict=False)
        assert len(issues) > 0
        assert any("END" in issue for issue in issues)

    def test_validate_edges_valid_workflow(self):
        """Test that validation passes for valid edges."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(type="agent_b", input=WorkflowNodeIO()),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        issues = workflow.validate_edges(strict=False)
        assert len(issues) == 0

    def test_validate_empty_workflow(self):
        """Test that validation detects empty workflows."""
        workflow = WorkflowFormat(nodes=[], edges=[])

        results = workflow.validate(strict=False)
        assert len(results["general"]) > 0
        assert any("no nodes" in issue.lower() for issue in results["general"])
        assert any("no edges" in issue.lower() for issue in results["general"])

    def test_validate_comprehensive(self):
        """Test comprehensive validation with multiple issues."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(
                    type="agent_b",
                    input=WorkflowNodeIO(),
                    io_map={"field1": "invalid"},  # Invalid JSONPath
                ),
            ],
            edges=[
                # Missing START edge
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        results = workflow.validate(strict=False)
        assert len(results["edges"]) > 0  # Missing START
        assert len(results["io_map"]) > 0  # Invalid JSONPath

    def test_validate_strict_mode_raises(self):
        """Test that strict mode raises ValueError on validation errors."""
        workflow = WorkflowFormat(nodes=[], edges=[])

        with pytest.raises(ValueError, match="Workflow validation failed"):
            workflow.validate(strict=True)

    def test_get_node_summary(self):
        """Test workflow summary generation."""
        workflow = WorkflowFormat(
            nodes=[
                WorkflowNode(type="agent_a", input=WorkflowNodeIO()),
                WorkflowNode(
                    type="agent_b",
                    input=WorkflowNodeIO(),
                    io_map={"field1": "$.agent_a.outputs.result"},
                ),
            ],
            edges=[
                WorkflowEdge(from_node="START", to_node="agent_a"),
                WorkflowEdge(from_node="agent_a", to_node="agent_b"),
                WorkflowEdge(from_node="agent_b", to_node="END"),
            ],
        )

        summary = workflow.get_node_summary()

        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 3
        assert summary["nodes_with_io_map"] == 1
        assert summary["total_io_map_entries"] == 1
        assert summary["node_types"] == ["agent_a", "agent_b"]
        assert summary["version"] == "1.0.0"
        assert summary["workflow_type"] == "AKDResearchWorkflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
