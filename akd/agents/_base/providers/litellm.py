"""LiteLLM provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litellm import acompletion

from akd._base.structures import RunContext, RunUsage

from ._base import ProviderAdapter
from .contracts import ProviderEvent, ProviderEventType, ProviderRequest


class LiteLLMAdapter(ProviderAdapter):
    """Adapter that will normalize LiteLLM streaming payloads into ProviderEvent."""

    def __init__(
        self,
        *,
        client: Any,
        completion_callable: Any = acompletion,
    ) -> None:
        self.client = client
        self._completion_callable = completion_callable

    @staticmethod
    def _extract_usage(chunk: Any) -> RunUsage:
        """Extract token usage from LiteLLM stream chunk."""
        run_usage = RunUsage()
        usage = getattr(chunk, "usage", None)
        if usage:
            run_usage.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            run_usage.output_tokens = getattr(usage, "completion_tokens", 0) or 0
            run_usage.requests = 1

            for details_attr in ("completion_tokens_details", "prompt_tokens_details"):
                details_obj = getattr(usage, details_attr, None)
                if details_obj is None:
                    continue
                items = (
                    details_obj.items()
                    if isinstance(details_obj, dict)
                    else ((k, v) for k, v in vars(details_obj).items() if isinstance(v, int) and v > 0)
                )
                for k, v in items:
                    if isinstance(v, int) and v > 0:
                        run_usage.details[f"{details_attr}.{k}"] = v
        return run_usage

    async def request_stream(
        self,
        *,
        run_context: RunContext,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        """Yield normalized provider events for a single LiteLLM run."""
        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before provider request")

        completion_kwargs: dict[str, Any] = {
            "model": request.model_name,
            "messages": messages,
            "stream": True,
        }
        if request.temperature is not None:
            completion_kwargs["temperature"] = request.temperature
        if request.tools:
            completion_kwargs["tools"] = request.tools
        if request.provider_kwargs:
            completion_kwargs.update(request.provider_kwargs)
        completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

        accumulated = ""
        response = await self._completion_callable(**completion_kwargs)
        async for chunk in response:
            usage = self._extract_usage(chunk)
            if usage.requests or usage.input_tokens or usage.output_tokens or usage.details:
                yield ProviderEvent(
                    kind=ProviderEventType.USAGE,
                    data={"usage": usage},
                )

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None)
            if reasoning:
                yield ProviderEvent(
                    kind=ProviderEventType.REASONING_DELTA,
                    data={"text": reasoning},
                )

            content = getattr(delta, "content", None)
            if content:
                accumulated += content
                yield ProviderEvent(
                    kind=ProviderEventType.TEXT_DELTA,
                    data={"text": content},
                )

        yield ProviderEvent(
            kind=ProviderEventType.FINAL_OUTPUT,
            data={"content": accumulated},
        )
