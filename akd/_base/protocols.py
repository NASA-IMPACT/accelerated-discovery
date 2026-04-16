"""Type protocols for the AKD framework.

Structural supertypes defining the contracts for AKD executables
(agents and tools). Concrete classes like ``AbstractBase``, ``BaseAgent``,
and ``BaseTool`` satisfy these protocols structurally — no explicit
Protocol inheritance required.

Framework adapters can satisfy these protocols either by inheriting
the concrete implementation kit (``BaseAgent``, ``BaseTool``) or by
implementing the methods directly on a framework-native class
(e.g. ``class PydanticAIAgent(pydantic_ai.Agent)``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .streaming import StreamEvent
    from .structures import RunContext


@runtime_checkable
class TokenCounts(Protocol):
    """Structural shape for usage/token-counter objects.

    Both ``akd.RunUsage`` (BaseModel) and ``pydantic_ai.RunUsage``
    (dataclass) satisfy this structurally.
    """

    input_tokens: int
    output_tokens: int
    requests: int


@runtime_checkable
class RunContextProtocol(Protocol):
    """Structural supertype for run-context-like objects.

    Both ``akd.RunContext`` and ``pydantic_ai.RunContext`` satisfy this
    structurally for the overlapping fields below. For akd-specific
    fields (``human_response``, ``extra``) or framework-specific fields
    (``deps``, ``tool_call_id``, etc.), narrow to the concrete type.
    """

    messages: list[Any] | None
    usage: TokenCounts
    run_id: str | None


@runtime_checkable
class AKDExecutable(Protocol):
    """Contract shared by all AKD runnables — agents and tools alike.

    Captures: typed I/O schemas, typed config, name/description metadata,
    and the async ``arun``/``astream`` entry points.
    """

    input_schema: type
    output_schema: type
    config_schema: type
    name: str | None
    description: str | None

    async def arun(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class AKDTool(AKDExecutable, Protocol):
    """Tool-specific extension — adds function export."""

    def as_function(
        self,
        mode: Literal["python", "json"] | None = None,
    ) -> Callable[..., Awaitable[Any]]: ...


# Disambiguating alias — use when pydantic_ai.RunContext is also in scope.
from .structures import RunContext as AKDRunContext  # noqa: E402

__all__ = [
    "AKDExecutable",
    "AKDRunContext",
    "AKDTool",
    "RunContextProtocol",
    "TokenCounts",
]
