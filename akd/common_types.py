# Type Definitions for tools, guardrails, and callables
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Tuple, Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from .agents._base import BaseAgent
from .tools._base import BaseTool

# Tool types - conditional based on langchain availability
try:
    import langchain_core  # noqa: F401

    LANGCHAIN_CORE_INSTALLED = True
except ImportError:
    LANGCHAIN_CORE_INSTALLED = False

if TYPE_CHECKING:
    from langchain_core.tools.structured import StructuredTool as _StructuredTool

    ToolType: TypeAlias = Union[BaseTool, BaseAgent, _StructuredTool]
else:
    try:
        from langchain_core.tools.structured import StructuredTool
    except ImportError:
        StructuredTool = None
    if LANGCHAIN_CORE_INSTALLED and StructuredTool is not None:
        ToolType: TypeAlias = Union[BaseTool, BaseAgent, "StructuredTool"]  # type: ignore
    else:
        ToolType: TypeAlias = Union[BaseTool, BaseAgent]

# Guardrail types
GuardrailType: TypeAlias = Union[BaseTool, Callable, Coroutine]

# Callable specifications
AnyCallable: TypeAlias = Union[BaseTool, BaseAgent, Callable[..., Any]]
CallableSpec: TypeAlias = Union[AnyCallable, Tuple[AnyCallable, Dict[str, str]]]

__all__ = [
    "ToolType",
    "GuardrailType",
    "AnyCallable",
    "CallableSpec",
]
