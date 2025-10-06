"""
Workflow format builder for AKD planner.

This module provides classes for building workflow definitions
that can be executed by the AKD framework.
"""

from __future__ import annotations

import json
from typing import Union

from pydantic import BaseModel, Field

# Type for workflow field values (supports common JSON-serializable types)
FieldValue = Union[str, int, float, bool, list, dict, None]

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

    type: str = Field(..., description="Node type corresponding to agent type")
    input: WorkflowNodeIO = Field(default_factory=WorkflowNodeIO)
    output: WorkflowNodeIO | None = Field(default=None)
    io_map: dict[str, str] | None = Field(
        default=None,
        description="JSONPath mappings for cross-node data access (target_field: jsonpath_expr)",
    )


class WorkflowEdge(BaseModel):
    """Edge connecting two nodes in a workflow."""

    from_node: str = Field(..., alias="from", description="Source node ID")
    to_node: str = Field(..., alias="to", description="Target node ID")

    model_config = {
        "populate_by_name": True,
    }


class WorkflowFormat(BaseModel):
    """Complete workflow definition in AKD format."""

    workflow_type: str = Field(default=WORKFLOW_TYPE)
    version: str = Field(default=WORKFLOW_FORMAT_VERSION)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    output: WorkflowNodeIO | None = Field(default=None)
    edges: list[WorkflowEdge] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WorkflowFormat:
        """Create a WorkflowFormat instance from a dictionary."""
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
