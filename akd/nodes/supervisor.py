import uuid
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from akd._base import AbstractBase, BaseConfig
from akd.common_types import ToolType as Tool
from akd.structures import ToolSearchResult

from .states import NodeState


class BaseSupervisorConfig(BaseConfig):
    """Configuration for BaseSupervisor."""

    name: str | None = None
    tools: list[Any] = Field(default_factory=list)
    mutation: bool = False
    state: NodeState | None = None


class BaseSupervisor(AbstractBase[NodeState, NodeState]):
    """Base class for node supervisors."""

    input_schema = NodeState
    output_schema = NodeState
    config_schema = BaseSupervisorConfig

    def _post_init(self) -> None:
        super()._post_init()
        # Set default name if not provided
        if not hasattr(self, "name") or self.name is None:
            self.name = f"{self.__classname__}({str(uuid.uuid4().hex)[:5]})"
        # Ensure tools is a list
        if not hasattr(self, "tools"):
            self.tools = []
        elif not isinstance(self.tools, list):
            self.tools = [self.tools] if self.tools else []
        # Initialize state if provided
        if hasattr(self, "state") and self.state:
            self._initial_state = self.state
        else:
            self._initial_state = NodeState()

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__

    @property
    def tool_map(self) -> dict[str, Tool]:
        return {getattr(tool, "name", tool.__class__.__name__): tool for tool in self.tools}

    async def arun_tool(
        self,
        tool: Tool,
        params: BaseModel | dict,
    ) -> dict[str, Any]:
        res = None
        if isinstance(params, BaseModel):
            params = params.model_dump()

        if hasattr(tool, "ainvoke"):
            res = await tool.ainvoke(input=params)
        elif hasattr(tool, "arun"):
            inp = tool.input_schema(**params)
            res = await tool.arun(inp)
            res = res.model_dump()
        return res

    @staticmethod
    def get_tool_by_name(tools: list[Tool], name: str) -> ToolSearchResult:
        name = name.lower().strip()
        search_result = ToolSearchResult(tool=None, args=None)
        for tool in tools:
            tool_name = getattr(tool, "name", tool.__class__.__name__).lower()
            if name in tool_name:
                search_result.tool = tool
                break
        return search_result

    def get_tool(self, tools: list[Tool], query: str) -> ToolSearchResult:
        return self.get_tool_by_name(tools, query)

    def _merge_state(self, base_state: NodeState, updates: NodeState) -> NodeState:
        """
        Create a new NodeState by merging updates into base state.

        Args:
            base_state: The base state to merge into
            updates: A NodeState object containing the updates to apply

        Returns:
            A new NodeState with merged values
        """
        # Create new state from base
        new_state = base_state.model_copy(deep=True)

        # Update messages
        if updates.messages:
            new_state.messages = updates.messages.copy()

        # Update inputs
        if updates.inputs:
            new_state.inputs = updates.inputs.copy()

        # Update outputs
        if updates.outputs:
            new_state.outputs = updates.outputs.copy()

        # # Update tool_calls
        # if updates.tool_calls:
        #     new_state.tool_calls = updates.tool_calls.copy()

        # Update steps - merge new steps with existing
        if updates.steps:
            for key, value in updates.steps.items():
                new_state.steps[key] = value

        return new_state

    @abstractmethod
    async def _arun(  # type: ignore
        self,
        state: NodeState,  # type: ignore
        **kwargs,
    ) -> NodeState:
        raise NotImplementedError("Subclass should implement this.")
