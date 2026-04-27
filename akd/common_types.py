"""Shared type aliases."""

from typing import Any, Callable, TypeAlias

from .agents._base import BaseAgent
from .tools._base import BaseTool

# A tool, agent, or raw callable — optionally paired with an input-key mapping.
# Used by ToolRunner to bind a state dict to a tool's input_schema.
CallableSpec: TypeAlias = (
    BaseTool | BaseAgent | Callable[..., Any] | tuple[BaseTool | BaseAgent | Callable[..., Any], dict[str, str]]
)

__all__ = ["CallableSpec"]
