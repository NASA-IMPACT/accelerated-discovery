import json
import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field

from akd._base import AbstractBase, BaseConfig
from akd.common_types import ToolType as Tool

from .states import NodeState, ToolSearchResult


class BaseSupervisorConfig(BaseConfig):
    """Configuration for BaseSupervisor."""

    name: Optional[str] = None
    tools: List[Any] = Field(default_factory=list)
    mutation: bool = False
    state: Optional[NodeState] = None


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
    def tool_map(self) -> Dict[str, Tool]:
        return {
            getattr(tool, "name", tool.__class__.__name__): tool for tool in self.tools
        }

    async def arun_tool(
        self,
        tool: Tool,
        params: Union[BaseModel, Dict],
    ) -> Dict[str, Any]:
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
    def get_tool_by_name(tools: List[Tool], name: str) -> ToolSearchResult:
        name = name.lower().strip()
        search_result = ToolSearchResult(tool=None, args=None)
        for tool in tools:
            tool_name = getattr(tool, "name", tool.__class__.__name__).lower()
            if name in tool_name:
                search_result.tool = tool
                break
        return search_result

    def get_tool(self, tools: List[Tool], query: str) -> ToolSearchResult:
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

        # Update tool_calls
        if updates.tool_calls:
            new_state.tool_calls = updates.tool_calls.copy()

        # Update steps - merge new steps with existing
        if updates.steps:
            for key, value in updates.steps.items():
                new_state.steps[key] = value

        return new_state

    @abstractmethod
    async def _arun(
        self,
        state: NodeState,
        **kwargs,
    ) -> NodeState:
        raise NotImplementedError("Subclass should implement this.")


class LLMSupervisorConfig(BaseSupervisorConfig):
    """Configuration for LLMSupervisor."""

    llm_client: Any


class LLMSupervisor(BaseSupervisor):
    """
    A supervisor that uses an LLM to handle tool calls.
    """

    config_schema = LLMSupervisorConfig

    def _post_init(self) -> None:
        super()._post_init()
        if hasattr(self, "llm_client"):
            self.tools = self._convert_tools_for_binding(self.tools)
            self.llm_client = self.llm_client.bind_tools(self.tools)

    @staticmethod
    def _convert_tools_for_binding(tools: List[Tool]) -> List[StructuredTool]:
        res = []
        for tool in tools:
            if isinstance(tool, StructuredTool):
                res.append(tool)
            elif hasattr(tool, "to_langchain_structured_tool"):
                res.append(tool.to_langchain_structured_tool())
            else:
                res.append(tool)
        return res

    def get_tool(self, tools: List[Tool], query: str) -> ToolSearchResult:
        result = self.llm_client.invoke(query)
        tool_calls = getattr(result, "tool_calls", [])
        if self.debug:
            logger.debug(f"Tool calls => {tool_calls}")

        name, args = query, {}
        if tool_calls:
            name = tool_calls[0].get("name", query)
            args = tool_calls[0].get("args", {})

        tool_res = self.get_tool_by_name(tools, name)
        tool_res.args = args
        return tool_res

    def _convert_tool_messages(self, messages: List) -> List[ToolSearchResult]:
        """
        Extract tool call information from a list of messages.

        Args:
            messages: List of message objects (HumanMessage, AIMessage, ToolMessage)

        Returns:
            List[ToolSearchResult]: List of tool search results with extracted information
        """
        results = []
        # Track tool calls by their ID for matching with tool message responses
        pending_tool_calls = {}

        for msg in messages:
            # Handle AIMessage with tool calls
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tool_call in msg.tool_calls:
                    # Get tool using the base class method
                    tool_result = self.get_tool_by_name(self.tools, tool_call["name"])
                    # Set arguments from the tool call
                    tool_result.args = tool_call["args"]
                    # Store by ID to match with corresponding ToolMessage response
                    pending_tool_calls[tool_call["id"]] = tool_result
                    results.append(tool_result)

            # Handle ToolMessage (the response from executing a tool)
            elif (
                isinstance(msg, ToolMessage)
                and hasattr(msg, "tool_call_id")
                and msg.tool_call_id
            ):
                tool_call_id = msg.tool_call_id
                if tool_call_id in pending_tool_calls:
                    # Add the result to the corresponding ToolSearchResult
                    try:
                        # If content is JSON string, parse it
                        if isinstance(
                            msg.content,
                            str,
                        ) and msg.content.strip().startswith("{"):
                            result = json.loads(msg.content)
                        else:
                            result = msg.content

                        pending_tool_calls[tool_call_id].result = result
                    except Exception:
                        # Handle parsing errors
                        pending_tool_calls[tool_call_id].result = msg.content

        return results


class ReActLLMSupervisor(LLMSupervisor):
    """
    ReAct pattern based supervisor
    """

    def _post_init(self) -> None:
        super()._post_init()
        if hasattr(self, "llm_client"):
            self.graph = create_react_agent(self.llm_client, tools=self.tools)


class ManualSupervisor(BaseSupervisor):
    """
    Manual type with custom rules
    """
