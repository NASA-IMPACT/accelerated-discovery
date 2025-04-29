from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from akd.structures import Tool
from akd.utils import LANGCHAIN_CORE_INSTALLED

if LANGCHAIN_CORE_INSTALLED:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = BaseModel


class ToolSearchResult(BaseModel):
    """
    Hold the tool search result
    """

    tool: Optional[Tool] = Field(
        ...,
        description="Tool found during search",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="When tool is found, what arguments extracted?",
    )
    result: Optional[Any] = Field(
        default=None,
        description="When tool is executed, set this field",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        return getattr(self.tool, "name", self.tool.__class__.__name__)


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
