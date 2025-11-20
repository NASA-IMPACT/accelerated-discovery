# Type Definitions for tools, guardrails, and callables
from typing import Any, Callable, Coroutine

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from .agents._base import BaseAgent
from .tools._base import BaseTool

ToolType: TypeAlias = BaseTool | BaseAgent

# Guardrail types
GuardrailType: TypeAlias = BaseTool | Callable | Coroutine

# Callable specifications
AnyCallable: TypeAlias = BaseTool | BaseAgent | Callable[..., Any]
CallableSpec: TypeAlias = AnyCallable | tuple[AnyCallable, dict[str, str]]

__all__ = [
    "ToolType",
    "GuardrailType",
    "AnyCallable",
    "CallableSpec",
]
