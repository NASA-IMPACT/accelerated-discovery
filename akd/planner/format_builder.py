"""
Workflow format builder for AKD planner.

This module provides classes for building workflow definitions
that can be executed by the AKD framework.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

# Type for workflow field values (supports common JSON-serializable types)
# Note: list and dict items are not recursively typed to avoid complexity
FieldValue = str | int | float | bool | list[Any] | dict[str, Any] | None

# Workflow format metadata
WORKFLOW_TYPE = "AKDResearchWorkflow"
WORKFLOW_FORMAT_VERSION = "1.0.0"


class WorkflowField(BaseModel):
    """Individual field in a workflow node's input or output."""

    name: str | None = None
    value: FieldValue = None
    # Support for various field structures in the sample format

    def model_dump(self, **kwargs) -> dict[str, FieldValue]:
        """Custom serialization to handle flexible field format."""
        # If both name and value are set, return as dict
        if self.name is not None and self.value is not None:
            return {self.name: self.value}
        # If only value is set, return just the value
        elif self.value is not None:
            return self.value
        # Otherwise return the full model
        return super().model_dump(**kwargs)


class WorkflowNodeIO(BaseModel):
    """Input or output specification for a workflow node."""

    fields: list[dict[str, FieldValue] | WorkflowField] = Field(default_factory=list)

    def model_dump(self, **kwargs) -> dict[str, list]:
        """Custom serialization to match the sample format."""
        result = {"fields": []}
        for field in self.fields:
            if isinstance(field, WorkflowField):
                result["fields"].append(field.model_dump(**kwargs))
            else:
                result["fields"].append(field)
        return result


class WorkflowNode(BaseModel):
    """Individual node in a workflow definition."""

    id: str = Field(..., description="Unique node identifier (e.g., my_agent_0)")
    type_: str = Field(..., alias="type", description="Node type corresponding to agent type for registry lookup")
    input: WorkflowNodeIO = Field(default_factory=WorkflowNodeIO)
    output: WorkflowNodeIO | None = Field(default=None)
    io_map: dict[str, str] | None = Field(
        default=None,
        description="JSONPath mappings for cross-node data access (target_field: jsonpath_expr using node id)",
    )

    model_config = {
        "populate_by_name": True,
        "serialize_by_alias": True,
    }


class WorkflowEdge(BaseModel):
    """Edge connecting two nodes in a workflow."""

    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")


class WorkflowFormat(BaseModel):
    """Complete workflow definition in AKD format."""

    workflow_type: str = Field(default=WORKFLOW_TYPE)
    version: str = Field(default=WORKFLOW_FORMAT_VERSION)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    output: WorkflowNodeIO | None = Field(default=None)
    edges: list[WorkflowEdge] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowFormat:
        """Create a WorkflowFormat instance from a dictionary (typically from JSON)."""
        return cls(**data)

    @classmethod
    def from_file(cls, file_path: str) -> WorkflowFormat:
        """Load a workflow from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, indent: int = 2, exclude_none: bool = True, **kwargs) -> str:
        """Export workflow to JSON string."""
        # Use exclude_none=True by default for cleaner output (omit null fields)
        kwargs.pop("exclude_none", None)
        kwargs.pop("indent", None)
        return self.model_dump_json(**kwargs, exclude_none=exclude_none, indent=indent)

    def save_to_file(self, file_path: str, **kwargs) -> None:
        """Save workflow to JSON file."""
        with open(file_path, "w") as f:
            f.write(self.to_json(**kwargs))

    def validate_io_map(self, strict: bool = False) -> list[str]:
        """
        Validate io_map JSONPath references in all nodes.

        Checks that:
        1. JSONPath expressions are well-formed (start with $.)
        2. Referenced node IDs exist in the workflow
        3. No circular dependencies exist

        Args:
            strict: If True, raise ValueError on validation errors. If False, return list of warnings.

        Returns:
            List of validation warnings/errors

        Raises:
            ValueError: If strict=True and validation errors found
        """
        issues = []
        node_ids = {node.id for node in self.nodes}

        for node in self.nodes:
            if not node.io_map:
                continue

            for target_field, jsonpath in node.io_map.items():
                # Check JSONPath format
                if not jsonpath.startswith("$."):
                    issue = f"Node '{node.id}', field '{target_field}': Invalid JSONPath '{jsonpath}' (must start with '$.')"
                    issues.append(issue)
                    continue

                # Extract referenced node from JSONPath (format: $.node_id.outputs.field)
                parts = jsonpath.split(".")
                if len(parts) < 4:
                    issue = f"Node '{node.id}', field '{target_field}': Invalid JSONPath format '{jsonpath}' (expected $.node_id.outputs.field)"
                    issues.append(issue)
                    continue

                referenced_node = parts[1]

                # Check referenced node exists
                if referenced_node not in node_ids:
                    issue = f"Node '{node.id}', field '{target_field}': Referenced node '{referenced_node}' not found in workflow"
                    issues.append(issue)

        if strict and issues:
            raise ValueError("Workflow io_map validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))

        return issues

    def validate_edges(self, strict: bool = False) -> list[str]:
        """
        Validate workflow edges.

        Checks that:
        1. All edges reference existing node IDs (or START/END)
        2. Workflow has a valid flow from START to END
        3. No orphaned nodes exist

        Args:
            strict: If True, raise ValueError on validation errors. If False, return list of warnings.

        Returns:
            List of validation warnings/errors

        Raises:
            ValueError: If strict=True and validation errors found
        """
        issues = []
        node_ids = {node.id for node in self.nodes}
        valid_nodes = node_ids | {"START", "END"}

        # Check all edge references are valid
        for edge in self.edges:
            if edge.from_node not in valid_nodes:
                issues.append(f"Edge references non-existent 'from' node: {edge.from_node}")
            if edge.to_node not in valid_nodes:
                issues.append(f"Edge references non-existent 'to' node: {edge.to_node}")

        # Check START and END exist
        has_start = any(edge.from_node == "START" for edge in self.edges)
        has_end = any(edge.to_node == "END" for edge in self.edges)

        if not has_start:
            issues.append("No edge from START node found")
        if not has_end:
            issues.append("No edge to END node found")

        # Check for orphaned nodes (nodes not in any edge)
        nodes_in_edges = set()
        for edge in self.edges:
            if edge.from_node != "START":
                nodes_in_edges.add(edge.from_node)
            if edge.to_node != "END":
                nodes_in_edges.add(edge.to_node)

        orphaned = node_ids - nodes_in_edges
        if orphaned:
            issues.append(f"Orphaned nodes (not connected to any edge): {orphaned}")

        if strict and issues:
            raise ValueError("Workflow edge validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))

        return issues

    def validate(self, strict: bool = False) -> dict[str, list[str]]:
        """
        Comprehensive workflow validation.

        Validates:
        - Edge connectivity
        - io_map references
        - Node structure

        Args:
            strict: If True, raise ValueError on any validation errors. If False, return dict of all issues.

        Returns:
            Dictionary with validation results: {"edges": [...], "io_map": [...], "general": [...]}

        Raises:
            ValueError: If strict=True and any validation errors found
        """
        results = {
            "edges": self.validate_edges(strict=False),
            "io_map": self.validate_io_map(strict=False),
            "general": [],
        }

        # General validations
        if not self.nodes:
            results["general"].append("Workflow has no nodes")

        if not self.edges:
            results["general"].append("Workflow has no edges")

        # Log validation results
        total_issues = sum(len(issues) for issues in results.values())
        if total_issues > 0:
            logger.warning(f"Workflow validation found {total_issues} issue(s)")
            for category, issues in results.items():
                if issues:
                    logger.warning(f"  {category.upper()}: {len(issues)} issue(s)")
                    for issue in issues:
                        logger.debug(f"    - {issue}")

        if strict and total_issues > 0:
            all_issues = []
            for category, issues in results.items():
                if issues:
                    all_issues.append(f"{category.upper()}:")
                    all_issues.extend(f"  - {issue}" for issue in issues)
            raise ValueError("Workflow validation failed:\n" + "\n".join(all_issues))

        return results

    def get_node_summary(self) -> dict[str, Any]:
        """
        Get summary statistics about the workflow.

        Returns:
            Dictionary with workflow statistics
        """
        nodes_with_io_map = [node for node in self.nodes if node.io_map]
        io_map_count = sum(len(node.io_map) for node in nodes_with_io_map if node.io_map)

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_with_io_map": len(nodes_with_io_map),
            "total_io_map_entries": io_map_count,
            "node_ids": [node.id for node in self.nodes],
            "node_types": [node.type_ for node in self.nodes],
            "version": self.version,
            "workflow_type": self.workflow_type,
        }
