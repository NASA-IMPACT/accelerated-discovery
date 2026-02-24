"""LiteLLM provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from litellm import acompletion
from pydantic import BaseModel, create_model

from akd._base.errors import UnexpectedModelBehavior
from akd._base.structures import RunContext, RunUsage

from ._base import ProviderAdapter
from .contracts import (
    ProviderEvent,
    ProviderEventType,
    ProviderRequest,
    ProviderResponse,
)


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

    @staticmethod
    def _normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
        """Normalize provider tool calls to list[dict[str, Any]]."""
        tool_calls = getattr(message, "tool_calls", None) or []
        normalized: list[dict[str, Any]] = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                normalized.append(call.model_dump())
            elif isinstance(call, dict):
                normalized.append(call)
            else:
                function = getattr(call, "function", None)
                normalized.append(
                    {
                        "id": getattr(call, "id", None),
                        "type": getattr(call, "type", "function"),
                        "function": {
                            "name": getattr(function, "name", None) if function is not None else None,
                            "arguments": (getattr(function, "arguments", None) if function is not None else None),
                        },
                    },
                )
        return normalized

    @staticmethod
    def _create_instructor_compatible_model(response_model: type[Any]) -> type[BaseModel]:
        """Create a BaseModel-only schema for instructor response_model."""
        fields: dict[str, tuple[Any, Any]] = {}
        for field_name, field_info in response_model.model_fields.items():
            fields[field_name] = (field_info.annotation, field_info)
        instructor_model = create_model(
            response_model.__name__,
            __base__=BaseModel,
            **cast(dict[str, Any], fields),
        )
        instructor_model.__doc__ = response_model.__doc__
        return instructor_model

    async def request_once(
        self,
        *,
        run_context: RunContext,
        request: ProviderRequest,
    ) -> ProviderResponse:
        """Return one normalized non-stream response for a LiteLLM request."""
        messages = run_context.messages
        if messages is None:
            raise ValueError("run_context.messages must be initialized before provider request")

        completion_kwargs: dict[str, Any] = {
            "model": request.model_name,
            "messages": messages,
            "stream": False,
        }
        if request.temperature is not None:
            completion_kwargs["temperature"] = request.temperature
        if request.tools:
            completion_kwargs["tools"] = request.tools
        if request.provider_kwargs:
            completion_kwargs.update(request.provider_kwargs)
        completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

        content: str | None = None
        tool_calls: list[dict[str, Any]] = []
        completion: Any
        usage = RunUsage()

        if request.output_schema is not None:
            instructor_model = self._create_instructor_compatible_model(request.output_schema)
            try:
                response, completion = await self.client.chat.completions.create_with_completion(
                    **completion_kwargs,
                    response_model=instructor_model,
                )
                usage = self._extract_usage(completion)
                if hasattr(response, "model_dump"):
                    content = json.dumps(response.model_dump())
                elif hasattr(response, "model_dump_json"):
                    dumped = response.model_dump_json()
                    content = dumped if isinstance(dumped, str) else str(dumped)
                else:
                    content = str(response)
            except Exception:
                # Instructor TOOLS mode can fail when the model returns structured
                # JSON in content instead of via tool calls.  Fall back to raw
                # completion and let the caller parse/validate the content.
                completion = await self._completion_callable(**completion_kwargs)
                usage = self._extract_usage(completion)
                choices = getattr(completion, "choices", None) or []
                if choices:
                    message = getattr(choices[0], "message", None)
                    if message is not None:
                        content = getattr(message, "content", None)
                        tool_calls = self._normalize_tool_calls(message)
        else:
            completion = await self._completion_callable(**completion_kwargs)
            usage = self._extract_usage(completion)

            choices = getattr(completion, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    tool_calls = self._normalize_tool_calls(message)

        if content is None and not tool_calls:
            raise UnexpectedModelBehavior("Provider returned neither content nor tool_calls")

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw=completion,
        )

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

            streamed_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in streamed_tool_calls:
                function = getattr(tc, "function", None)
                yield ProviderEvent(
                    kind=ProviderEventType.TOOL_CALL,
                    data={
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "name": getattr(function, "name", None) if function is not None else None,
                        "arguments_delta": (getattr(function, "arguments", "") if function is not None else ""),
                    },
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
