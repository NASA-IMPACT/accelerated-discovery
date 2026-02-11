"""Memory storage for agents."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from litellm.utils import trim_messages
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .structures import RunContext


class Memory[T](BaseModel):
    """Message storage with session-based lifecycle management.

    Provides context managers for memory operations:
    - session() for sync code
    - asession() for async code

    Session handles:
    - Stateless clearing (start and end)
    - Context syncing (message_history)
    - Message trimming
    """

    messages: list[T] = Field(default_factory=list)

    def append(self, msg: T) -> None:
        """Add a message."""
        self.messages.append(msg)

    def extend(self, items: list[T]) -> None:
        """Add multiple messages."""
        self.messages.extend(items)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def all(self) -> list[T]:
        """Get copy of all messages."""
        return list(self.messages)

    def sync(self, items: list[T]) -> None:
        """Replace all messages."""
        self.messages.clear()
        self.messages.extend(items)

    def last(self) -> T | None:
        """Get last message."""
        return self.messages[-1] if self.messages else None

    def __len__(self) -> int:
        return len(self.messages)

    def __bool__(self) -> bool:
        return len(self.messages) > 0

    def __iter__(self):
        """Iterate over messages."""
        return iter(self.messages)

    def __getitem__(self, index: int) -> T:
        """Get message by index."""
        return self.messages[index]

    @contextmanager
    def session(
        self,
        stateless: bool = False,
        run_context: RunContext | None = None,
        enable_trimming: bool = False,
        model_name: str | None = None,
        max_tokens: int | None = None,
        trim_ratio: float = 0.75,
    ):
        """Sync context manager for memory operations.

        Args:
            stateless: Clear memory at start and end
            run_context: RunContext with messages for resumption
            enable_trimming: Enable message trimming
            model_name: Model name for trimming
            max_tokens: Max tokens for trimming
            trim_ratio: Trim ratio (default 0.75)

        Yields:
            Reference to self.messages
        """
        # Setup
        if stateless:
            self.clear()

        if run_context and run_context.messages:
            self.sync(run_context.messages)

        if enable_trimming and model_name and max_tokens:
            trimmed = trim_messages(
                self.messages,
                model=model_name,
                max_tokens=max_tokens,
                trim_ratio=trim_ratio,
            )
            self.sync(trimmed)

        try:
            yield self.messages
        finally:
            if stateless:
                self.clear()

    @asynccontextmanager
    async def asession(
        self,
        stateless: bool = False,
        run_context: RunContext | None = None,
        enable_trimming: bool = False,
        model_name: str | None = None,
        max_tokens: int | None = None,
        trim_ratio: float = 0.75,
    ):
        """Async context manager for memory operations.

        Args:
            stateless: Clear memory at start and end
            run_context: RunContext with messages for resumption
            enable_trimming: Enable message trimming
            model_name: Model name for trimming
            max_tokens: Max tokens for trimming
            trim_ratio: Trim ratio (default 0.75)

        Yields:
            Reference to self.messages
        """
        with self.session(
            stateless=stateless,
            run_context=run_context,
            enable_trimming=enable_trimming,
            model_name=model_name,
            max_tokens=max_tokens,
            trim_ratio=trim_ratio,
        ) as messages:
            yield messages


__all__ = ["Memory"]
