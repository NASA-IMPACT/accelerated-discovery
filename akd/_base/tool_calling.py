"""Tool calling support for akd agents."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from akd.observability import span_tool_execution

if TYPE_CHECKING:
    from akd.tools._base import BaseTool


class ToolCall(BaseModel):
    """Tool call request (normalized from any provider).

    This is akd's canonical format for tool calls. Different providers
    (LiteLLM, Pydantic AI, LangChain) convert to this format.

    Aligned with Pydantic AI's ToolCallPart structure.
    """

    tool_call_id: str = Field(description="Unique identifier for this call")
    tool_name: str = Field(description="Name of tool to call")
    arguments: dict[str, Any] = Field(description="Input arguments as dict")


class ToolResult(BaseModel):
    """Tool execution result.

    Aligned with Pydantic AI's ToolReturnPart structure.
    """

    tool_call_id: str = Field(description="Links to original ToolCall")
    tool_name: str = Field(description="Name of tool that was called")
    content: Any = Field(description="Return value from tool")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = Field(default=None, description="Error message if failed")


class ToolCallingMixin:
    """Mixin providing reusable tool execution helpers.

    Requires the inheriting class to have:
        - self.tools: list[BaseTool]

    Example:
        class MyAgent(ToolCallingMixin, BaseAgent):
            async def _astream(self, params, context, **kwargs):
                # Parse provider format → ToolCall
                tool_call = ToolCall(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                result = await self._execute_tool(tool_call)
    """

    def _find_tool(
        self,
        name: str,
        tools: list[BaseTool] | None = None,
    ) -> BaseTool | None:
        """Find a tool by name.

        Args:
            name: The name of the tool to find (checks both tool.name and class name)
            tools: Optional list of tools to search. Defaults to self.tools.

        Returns:
            The matching tool instance, or None if not found
        """
        tools = tools or self.tools
        return next(
            (t for t in tools if t.name == name or t.__class__.__name__ == name),
            None,
        )

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        tools: list[BaseTool] | None = None,
    ) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: Normalized tool call request
            tools: Optional list of tools to search. Defaults to self.tools.

        Returns:
            ToolResult with content or error
        """
        tool = self._find_tool(tool_call.tool_name, tools=tools)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                content=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )

        with span_tool_execution(
            tool_call.tool_name,
            tool_call.tool_call_id,
        ):
            try:
                input_obj = tool.input_schema(**tool_call.arguments)
                result = await tool.arun(input_obj)
                # Use mode='json' to ensure JSON-serializable types (HttpUrl → str, datetime → ISO string)
                content = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
                return ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    content=content,
                )
            except Exception as e:
                logger.exception(f"Tool '{tool_call.tool_name}' failed with args {tool_call.arguments}")
                return ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    content=None,
                    error=str(e),
                )

    async def _execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        tools: list[BaseTool] | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel.

        When an LLM returns multiple tool calls in a single response,
        they are independent by definition and can be executed concurrently.

        Args:
            tool_calls: List of normalized tool calls
            tools: Optional list of tools to search. Defaults to self.tools.

        Returns:
            List of ToolResults in same order as input
        """
        return list(
            await asyncio.gather(*[self._execute_tool(tc, tools=tools) for tc in tool_calls]),
        )


__all__ = ["ToolCall", "ToolResult", "ToolCallingMixin"]
