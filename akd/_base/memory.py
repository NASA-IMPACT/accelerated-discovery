"""Memory storage for agents."""

from typing import Any

from pydantic import BaseModel, Field


class Memory[T](BaseModel):
    """Message storage with context-aware preparation.

    Handles:
    - Message storage (append, extend, clear)
    - Context syncing (prepare syncs from context["message_history"])
    - Returns reference for auto-sync during runs
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

    def prepare(self, context: dict[str, Any] | None = None) -> list[T]:
        """Prepare messages for a run.

        Syncs from context["message_history"] if provided (human-in-the-loop resumption).
        Returns REFERENCE to self.messages - changes during run auto-sync.
        """
        if context and "message_history" in context:
            self.sync(context["message_history"])
        return self.messages

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


__all__ = ["Memory"]
