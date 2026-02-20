"""Run session abstractions for agents and tools."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from litellm.utils import trim_messages

from .structures import RunContext

type Message = dict[str, Any]


@dataclass(slots=True)
class BaseSession:
    """Base async session with run-context lifecycle hooks."""

    run_context: RunContext

    async def __aenter__(self) -> BaseSession:
        self.run_context.run_id = self.run_context.run_id or uuid.uuid4().hex[:8]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        self.finalize_run_state(outcome=None, error=exc)

    def finalize_run_state(
        self,
        *,
        outcome: Any | None,
        error: BaseException | None,
    ) -> None:
        """Finalize run state on session exit."""
        return None


@dataclass(slots=True)
class AgentSession(BaseSession):
    """Agent run session that manages message lifecycle."""

    store: list[Message] = field(default_factory=list)
    stateless: bool = False
    enable_trimming: bool = False
    model_name: str | None = None
    max_tokens: int | None = None
    trim_ratio: float = 0.75

    @property
    def messages(self) -> list[Message]:
        return self.store

    def append(self, message: Message) -> None:
        self.store.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.store.extend(messages)

    def clear(self) -> None:
        self.store.clear()

    async def __aenter__(self) -> AgentSession:
        await super(AgentSession, self).__aenter__()

        if self.stateless:
            self.clear()

        if self.run_context.messages:
            self.store[:] = list(self.run_context.messages)

        self.run_context.messages = self.store

        if self.enable_trimming and self.model_name and self.max_tokens:
            trimmed = trim_messages(
                self.store,
                model=self.model_name,
                max_tokens=self.max_tokens,
                trim_ratio=self.trim_ratio,
            )
            self.store[:] = trimmed
            self.run_context.messages = self.store

        return self

    def finalize_run_state(
        self,
        *,
        outcome: Any | None,
        error: BaseException | None,
    ) -> None:
        self.run_context.messages = list(self.store)
        if self.stateless:
            self.clear()


@dataclass(slots=True)
class ToolSession(BaseSession):
    """Tool run session metadata."""

    tool_name: str
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    tool_error: str | None = None

    def finalize_run_state(
        self,
        *,
        outcome: Any | None,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            self.tool_error = str(error)
        else:
            self.tool_output = outcome


__all__ = ["BaseSession", "AgentSession", "ToolSession", "Message"]
