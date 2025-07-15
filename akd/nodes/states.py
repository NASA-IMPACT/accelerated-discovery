from typing import TYPE_CHECKING, Annotated, Any, Dict, List

from pydantic import BaseModel, Field

from akd.utils import LANGCHAIN_CORE_INSTALLED

if TYPE_CHECKING or LANGCHAIN_CORE_INSTALLED:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = BaseModel


class NodeState(BaseModel):
    """Unified state class for all node operations.

    This class combines fields from the previous NodeState, SupervisorState, and NodeTemplateState
    to provide a single, unified state structure. Supervisor fields are optional and only used
    when a supervisor is present in the node template.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core node fields
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
    )
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Guardrail results
    input_guardrails: Dict[str, Any] = Field(default_factory=dict)
    output_guardrails: Dict[str, Any] = Field(default_factory=dict)

    # Optional supervisor fields (only used when supervisor is present)
    steps: Dict[str, Any] = Field(default_factory=dict)

    # TODO: fix serialization issues
    # tool_calls: List[ToolSearchResult] = Field(default_factory=list)


def merge_node_states(
    existing: Dict[str, NodeState],
    update: Dict[str, NodeState],
) -> Dict[str, NodeState]:
    """Custom reducer to merge node_states updates."""
    if existing is None:
        return update

    # Create a copy and update with new values
    merged = existing.copy()
    merged.update(update)
    return merged


class GlobalState(NodeState):
    """
    Global state of the system.
    """

    # node_states: Dict[str, NodeState] = Field(default_factory=dict)
    node_states: Annotated[Dict[str, NodeState], merge_node_states] = Field(
        default_factory=dict,
    )


# Backward compatibility aliases
# These will be removed in a future version
SupervisorState = NodeState
NodeTemplateState = NodeState
