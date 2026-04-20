"""Tool call / result data models for akd agents.

Provider-specific adapters convert their native tool call formats
into these canonical types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


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


__all__ = ["ToolCall", "ToolResult"]
