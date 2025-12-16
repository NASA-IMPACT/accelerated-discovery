# Type Definitions for tools, guardrails, and callables
from typing import Any, Callable

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from .agents._base import BaseAgent
from .guardrails._base import GuardrailProtocol
from .tools._base import BaseTool

ToolType: TypeAlias = BaseTool | BaseAgent

# Guardrail types - use GuardrailProtocol for unified guardrails API
GuardrailType: TypeAlias = GuardrailProtocol

# Callable specifications (used for ToolRunner)
AnyCallable: TypeAlias = BaseTool | BaseAgent | Callable[..., Any]
CallableSpec: TypeAlias = AnyCallable | tuple[AnyCallable, dict[str, str]]

__all__ = [
    "ToolType",
    "GuardrailType",
    "GuardrailProtocol",
    "AnyCallable",
    "CallableSpec",
]
