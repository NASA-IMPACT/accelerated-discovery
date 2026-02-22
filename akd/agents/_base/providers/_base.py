"""Provider adapter protocol used by BaseAgent run engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from akd._base.structures import RunContext

from .contracts import ProviderEvent, ProviderRequest, ProviderResponse


class ProviderAdapter(Protocol):
    """Protocol for provider transport/event adapters.

    Adapters parse backend-specific streams and emit normalized ProviderEvent values.
    They do not own AKD session or StreamEvent orchestration.
    """

    async def request_once(
        self,
        *,
        run_context: RunContext,
        request: ProviderRequest,
    ) -> ProviderResponse:
        """Return one normalized provider response for a single run."""
        ...

    def request_stream(
        self,
        *,
        run_context: RunContext,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        """Yield normalized provider events for a single run."""
        ...


__all__ = ["ProviderAdapter"]
