from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from akd.structures import ToolSearchResult
from akd.utils import LANGCHAIN_CORE_INSTALLED

if TYPE_CHECKING or LANGCHAIN_CORE_INSTALLED:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = BaseModel


class NodeState(BaseModel):
    """Fields common to all node-like states."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    messages: List[Union[BaseMessage, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Message history for this node",
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for this node",
    )
    output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data from this node",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node metadata and configuration",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Node creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Node last update timestamp",
    )

    def update_timestamp(self) -> None:
        """Update the timestamp when state is modified."""
        self.updated_at = datetime.now()

    def merge_inputs(self, new_inputs: Dict[str, Any]) -> None:
        """Merge new inputs with existing inputs."""
        self.inputs.update(new_inputs)
        self.update_timestamp()

    def merge_output(self, new_output: Dict[str, Any]) -> None:
        """Merge new output with existing output."""
        self.output.update(new_output)
        self.update_timestamp()

    def add_message(self, message: Union[BaseMessage, Dict[str, Any]]) -> None:
        """Add a message to the message history."""
        self.messages.append(message)
        self.update_timestamp()

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.update_timestamp()


class SupervisorState(NodeState):
    """Extra fields that only the Supervisor cares about."""

    tool_calls: List[ToolSearchResult] = Field(
        default_factory=list,
        description="Tool execution results",
    )
    steps: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution steps and intermediate results",
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Current execution step",
    )
    execution_status: str = Field(
        default="pending",
        description="Execution status: pending, running, completed, failed",
    )
    error_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error information if execution failed",
    )

    @model_validator(mode="after")
    def validate_execution_status(self):
        """Validate execution status values."""
        valid_statuses = {"pending", "running", "completed", "failed"}
        if self.execution_status not in valid_statuses:
            raise ValueError(f"Invalid execution status: {self.execution_status}")
        return self

    def set_status(
        self,
        status: str,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update execution status."""
        self.execution_status = status
        if error_info:
            self.error_info = error_info
        self.update_timestamp()

    def add_tool_call(self, tool_result: ToolSearchResult) -> None:
        """Add a tool call result."""
        self.tool_calls.append(tool_result)
        self.update_timestamp()

    def update_steps(self, new_steps: Dict[str, Any]) -> None:
        """Update execution steps."""
        self.steps.update(new_steps)
        self.update_timestamp()


class NodeTemplateState(NodeState):
    """State for a node template including supervisor and guardrails."""

    supervisor_state: SupervisorState = Field(
        default_factory=SupervisorState,
        description="Supervisor execution state",
    )
    input_guardrails: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input validation results",
    )
    output_guardrails: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output validation results",
    )
    node_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this node",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of node IDs this node depends on",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing this node",
    )

    def is_ready_to_execute(self, global_state: Optional["GlobalState"] = None) -> bool:
        """Check if node is ready to execute based on dependencies."""
        if not self.dependencies:
            return True

        if global_state is None:
            return False

        for dep_id in self.dependencies:
            dep_state = global_state.node_states.get(dep_id)
            if (
                not dep_state
                or dep_state.supervisor_state.execution_status != "completed"
            ):
                return False
        return True

    def get_dependency_outputs(self, global_state: "GlobalState") -> Dict[str, Any]:
        """Get outputs from all dependencies."""
        dependency_outputs = {}
        for dep_id in self.dependencies:
            dep_state = global_state.node_states.get(dep_id)
            if dep_state and dep_state.output:
                dependency_outputs[dep_id] = dep_state.output
        return dependency_outputs

    def add_dependency(self, node_id: str) -> None:
        """Add a dependency to this node."""
        if node_id not in self.dependencies:
            self.dependencies.append(node_id)
            self.update_timestamp()

    def remove_dependency(self, node_id: str) -> None:
        """Remove a dependency from this node."""
        if node_id in self.dependencies:
            self.dependencies.remove(node_id)
            self.update_timestamp()

    def add_tag(self, tag: str) -> None:
        """Add a tag to this node."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.update_timestamp()

    def has_tag(self, tag: str) -> bool:
        """Check if node has a specific tag."""
        return tag in self.tags


class GlobalState(NodeState):
    """
    Global state of the system.
    """

    node_states: Dict[str, NodeTemplateState] = Field(
        default_factory=dict,
        description="State for each node in the workflow",
    )
    workflow_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-level metadata",
    )
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared context accessible to all nodes",
    )
    execution_plan: List[str] = Field(
        default_factory=list,
        description="Planned execution order of nodes",
    )
    interrupts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Interrupt points for human-in-the-loop",
    )
    checkpoints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Checkpoint data for state recovery",
    )

    def get_node_state(self, node_id: str) -> Optional[NodeTemplateState]:
        """Get state for a specific node."""
        return self.node_states.get(node_id)

    def create_node_state(
        self,
        node_id: str,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> NodeTemplateState:
        """Create a new node state."""
        if node_id in self.node_states:
            raise ValueError(f"Node {node_id} already exists")

        node_state = NodeTemplateState(
            node_id=node_id,
            inputs=initial_inputs or {},
        )
        self.node_states[node_id] = node_state
        self.update_timestamp()
        return node_state

    def update_node_state(self, node_id: str, node_state: NodeTemplateState) -> None:
        """Update state for a specific node."""
        self.node_states[node_id] = node_state
        self.update_timestamp()

    def remove_node_state(self, node_id: str) -> None:
        """Remove state for a specific node."""
        if node_id in self.node_states:
            del self.node_states[node_id]
            self.update_timestamp()

    def get_nodes_by_tag(self, tag: str) -> List[str]:
        """Get all node IDs with a specific tag."""
        return [
            node_id for node_id, state in self.node_states.items() if state.has_tag(tag)
        ]

    def get_completed_nodes(self) -> List[str]:
        """Get all completed node IDs."""
        return [
            node_id
            for node_id, state in self.node_states.items()
            if state.supervisor_state.execution_status == "completed"
        ]

    def get_failed_nodes(self) -> List[str]:
        """Get all failed node IDs."""
        return [
            node_id
            for node_id, state in self.node_states.items()
            if state.supervisor_state.execution_status == "failed"
        ]

    def get_ready_nodes(self) -> List[str]:
        """Get all nodes that are ready to execute."""
        return [
            node_id
            for node_id, state in self.node_states.items()
            if state.is_ready_to_execute(self)
            and state.supervisor_state.execution_status == "pending"
        ]

    def add_to_shared_context(self, key: str, value: Any) -> None:
        """Add data to shared context."""
        self.shared_context[key] = value
        self.update_timestamp()

    def get_from_shared_context(self, key: str, default: Any = None) -> Any:
        """Get data from shared context."""
        return self.shared_context.get(key, default)

    def set_checkpoint(self, checkpoint_id: str, data: Any) -> None:
        """Set a checkpoint for state recovery."""
        self.checkpoints[checkpoint_id] = {
            "data": data,
            "timestamp": datetime.now(),
        }
        self.update_timestamp()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """Get checkpoint data."""
        checkpoint = self.checkpoints.get(checkpoint_id)
        return checkpoint["data"] if checkpoint else None

    def set_interrupt(self, interrupt_id: str, data: Any) -> None:
        """Set an interrupt point for human intervention."""
        self.interrupts[interrupt_id] = {
            "data": data,
            "timestamp": datetime.now(),
            "resolved": False,
        }
        self.update_timestamp()

    def resolve_interrupt(self, interrupt_id: str, resolution_data: Any = None) -> None:
        """Resolve an interrupt point."""
        if interrupt_id in self.interrupts:
            self.interrupts[interrupt_id]["resolved"] = True
            self.interrupts[interrupt_id]["resolution_data"] = resolution_data
            self.interrupts[interrupt_id]["resolved_at"] = datetime.now()
            self.update_timestamp()

    def has_unresolved_interrupts(self) -> bool:
        """Check if there are any unresolved interrupts."""
        return any(
            not interrupt.get("resolved", False)
            for interrupt in self.interrupts.values()
        )

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get overall workflow status."""
        total_nodes = len(self.node_states)
        completed_nodes = len(self.get_completed_nodes())
        failed_nodes = len(self.get_failed_nodes())
        pending_nodes = total_nodes - completed_nodes - failed_nodes

        return {
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "pending_nodes": pending_nodes,
            "progress_percentage": (completed_nodes / total_nodes * 100)
            if total_nodes > 0
            else 0,
            "has_failures": failed_nodes > 0,
            "has_interrupts": self.has_unresolved_interrupts(),
            "last_updated": self.updated_at,
        }
