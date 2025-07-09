import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel

from akd.common_types import ToolType as Tool
from akd.utils import AsyncRunMixin

from .states import GlobalState, SupervisorState, ToolSearchResult


class BaseSupervisor(ABC, AsyncRunMixin):
    """Base class for node supervisors."""

    def __init__(
        self,
        name: Optional[str] = None,
        tools: Tool | None = None,
        state: Optional[SupervisorState] = None,
        debug: bool = False,
    ):
        self.tools = tools or []
        self.debug = bool(debug)
        self.name = name or f"{self.__classname__}({str(uuid.uuid4().hex)[:5]})"
        self.state = state or SupervisorState()

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

    def update_state(self, updates: SupervisorState) -> SupervisorState:
        """
        Update the supervisor's state with values from another SupervisorState.

        Args:
            updates: A SupervisorState object containing the updates to apply

        Returns:
            The updated SupervisorState
        """
        # Update messages
        self.state.messages = (
            updates.messages.copy() if updates.messages else self.state.messages
        )

        # Update inputs
        if updates.inputs:
            self.state.inputs = updates.inputs.copy()

        # Update output
        if updates.output:
            self.state.output = updates.output.copy()

        # Update tool_calls - append new ones or replace entirely
        if updates.tool_calls:
            self.state.tool_calls = updates.tool_calls.copy()

        # Update steps - merge new steps with existing
        if updates.steps:
            for key, value in updates.steps.items():
                self.state.steps[key] = value

        return self.state

    @abstractmethod
    async def arun(
        self,
        state: SupervisorState,
        global_state: Optional[GlobalState] = None,
        **kwargs,
    ) -> SupervisorState:
        raise NotImplementedError("Subclass should implement this.")


class LLMSupervisor(BaseSupervisor):
    """
    A supervisor that uses an LLM to handle tool calls.
    """

    def __init__(self, llm_client: Any, tools: Tool | None = None, debug: bool = False):
        super().__init__(tools=tools, debug=debug)
        self.tools = self._convert_tools_for_binding(self.tools)
        self.llm_client = llm_client.bind_tools(self.tools)

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

    def __init__(self, llm_client: Any, tools: Tool | None = None, debug: bool = False):
        super().__init__(llm_client=llm_client, tools=tools, debug=debug)
        self.graph = create_react_agent(self.llm_client, tools=self.tools)


class ManualSupervisor(BaseSupervisor):
    """
    Manual type with custom rules
    """

    pass


class DummyLLMSupervisor(ReActLLMSupervisor):
    """
    Dummy LLM based supervisor using ReAct
    """

    async def arun(
        self,
        state: SupervisorState,
        global_state: Optional[GlobalState] = None,
        **kwargs,
    ) -> SupervisorState:
        query = (
            state.inputs.get("query", "")
            or state.inputs.get("question", "")
            or state.inputs.get("input", "")
        )
        if not query:
            raise ValueError(
                "State must provide an 'inputs' field with a 'query' or 'question'.",
            )
        execution_state = state.model_copy()

        # message update
        messages = state.messages.copy()
        messages.append(HumanMessage(content=query))

        # Update the state object with new messages
        updated_state = SupervisorState(
            messages=messages,
            inputs=state.inputs,
            output=state.output,
            tool_calls=state.tool_calls,
            steps=state.steps,
        )

        # Invoke the graph with the updated state
        result = await self.graph.ainvoke(updated_state)

        # Get updated messages from result (assuming result is also a SupervisorState or dict)
        result_messages = (
            getattr(result, "messages", messages)
            if isinstance(result, BaseModel)
            else result.get("messages", messages)
        )

        # Update execution state
        execution_state.messages = result_messages
        execution_state.tool_calls = self._convert_tool_messages(result_messages)

        if execution_state.tool_calls and len(execution_state.tool_calls) > 0:
            execution_state.output = {"result": execution_state.tool_calls[-1].result}

        for tc in execution_state.tool_calls:
            execution_state.steps.update(tc.result)

        self.update_state(execution_state)
        return execution_state
