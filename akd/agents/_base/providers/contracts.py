"""Internal provider adapter contracts for BaseAgent runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProviderEventType(StrEnum):
    """Normalized internal event types emitted by provider adapters."""

    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HUMAN_INPUT_REQUIRED = "human_input_required"
    USAGE = "usage"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"


class ProviderEvent(BaseModel):
    """Normalized internal provider event consumed by BaseAgent run engine."""

    kind: ProviderEventType
    data: dict[str, Any] = Field(default_factory=dict)


class ProviderRequest(BaseModel):
    """Provider-agnostic request payload built by BaseAgent for adapters."""

    model_name: str | None = None
    temperature: float | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    token_batch_size: int = 10
    output_schema: type[Any] | None = None
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ProviderEvent", "ProviderEventType", "ProviderRequest"]
