from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import BaseModel, Field

from akd.structures import ToolSearchResult
from akd.utils import LANGCHAIN_CORE_INSTALLED

if TYPE_CHECKING or LANGCHAIN_CORE_INSTALLED:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = BaseModel


class NodeState(BaseModel):
    """Fields common to all node-like states."""

    messages: List[Union[BaseMessage, Dict[str, Any]]] = Field(
        default_factory=list,
    )
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


class SupervisorState(NodeState):
    """Extra fields that only the Supervisor cares about."""

    tool_calls: List[ToolSearchResult] = Field(default_factory=list)
    steps: Dict[str, Any] = Field(default_factory=dict)


class NodeTemplateState(NodeState):
    supervisor_state: SupervisorState = Field(default_factory=SupervisorState)
    input_guardrails: Dict[str, Any] = Field(default_factory=dict)
    output_guardrails: Dict[str, Any] = Field(default_factory=dict)


class GlobalState(NodeState):
    """
    Global state of the system.
    """

    node_states: Dict[str, NodeTemplateState] = Field(default_factory=dict)
