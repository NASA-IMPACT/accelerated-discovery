"""
Workflow format builder for AKD planner.

This module provides classes for building workflow definitions
that can be executed by the AKD framework.
"""

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class WorkflowField(BaseModel):
    """Individual field in a workflow node's input or output."""

    name: Optional[str] = None
    value: Any = None
    # Support for various field structures in the sample format

    def model_dump(self, **kwargs) -> Dict[str, Any]:
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

    fields: List[Union[Dict[str, Any], WorkflowField]] = Field(default_factory=list)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
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
    output: Optional[WorkflowNodeIO] = Field(default=None)
    io_map: Optional[Dict[str, str]] = Field(
        default=None, description="JSONPath mappings for cross-node data access (target_field: jsonpath_expr)"
    )


class WorkflowEdge(BaseModel):
    """Edge connecting two nodes in a workflow."""

    from_node: str = Field(..., alias="from", description="Source node ID")
    to_node: str = Field(..., alias="to", description="Target node ID")

    class Config:
        populate_by_name = True


class WorkflowFormat(BaseModel):
    """Complete workflow definition in AKD format."""

    workflow_type: str = Field(default="AKDResearchWorkflow")
    version: str = Field(default="1.0.0")
    nodes: List[WorkflowNode] = Field(default_factory=list)
    output: Optional[WorkflowNodeIO] = Field(default=None)
    edges: List[WorkflowEdge] = Field(default_factory=list)

    def to_json(self, **kwargs) -> str:
        """Export workflow to JSON string."""
        # Use exclude_none=True by default for cleaner output (omit null fields)
        kwargs.setdefault("exclude_none", True)
        return json.dumps(self.model_dump(**kwargs), indent=2)

    def save_to_file(self, file_path: str, **kwargs) -> None:
        """Save workflow to JSON file."""
        with open(file_path, "w") as f:
            f.write(self.to_json(**kwargs))


def create_workflow_from_dict(data: Dict[str, Any]) -> WorkflowFormat:
    """Create a WorkflowFormat instance from a dictionary."""
    return WorkflowFormat(**data)


def load_workflow_from_file(file_path: str) -> WorkflowFormat:
    """Load a workflow from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return create_workflow_from_dict(data)
