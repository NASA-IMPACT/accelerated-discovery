from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import BaseModel, Field

from akd.structures import ToolSearchResult
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

    # Core node fields
    messages: List[Union[BaseMessage, Dict[str, Any]]] = Field(
        default_factory=list,
    )
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Guardrail results
    input_guardrails: Dict[str, Any] = Field(default_factory=dict)
    output_guardrails: Dict[str, Any] = Field(default_factory=dict)

    # Optional supervisor fields (only used when supervisor is present)
    tool_calls: List[ToolSearchResult] = Field(default_factory=list)
    steps: Dict[str, Any] = Field(default_factory=dict)


class GlobalState(NodeState):
    """
    Global state of the system.
    """

    planner_state: Any = Field(default=None)  # temporary placeholder for planner state
    node_states: Dict[str, NodeState] = Field(default_factory=dict)


# Backward compatibility aliases
# These will be removed in a future version
SupervisorState = NodeState
NodeTemplateState = NodeState
