"""LiteLLM provider adapter scaffold."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from akd._base.structures import RunContext

from ._base import ProviderAdapter
from .contracts import ProviderEvent, ProviderEventType, ProviderRequest


class LiteLLMAdapter(ProviderAdapter):
    """Adapter that will normalize LiteLLM streaming payloads into ProviderEvent."""

    def __init__(
        self,
        *,
        client: Any,
    ) -> None:
        self.client = client

    async def request_stream(
        self,
        *,
        run_context: RunContext,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        """Yield normalized provider events for a single LiteLLM run."""
        raise NotImplementedError("LiteLLMAdapter.request_stream is not wired yet")
        if False:  # pragma: no cover - keeps AsyncIterator shape explicit for type-checkers
            yield ProviderEvent(kind=ProviderEventType.ERROR, data={})
