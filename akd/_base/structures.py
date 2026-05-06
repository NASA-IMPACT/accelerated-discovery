"""Common data structures for the AKD framework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .tool_calling import ToolResult


class HumanResponse(ToolResult):
    """Human's response to a HUMAN_INPUT_REQUIRED event.

    Inherits from ToolResult since a human response IS a tool result
    for the ask_human tool call. The content field is Any (inherited from
    ToolResult) and is serialized via json.dumps() when injected back into
    the conversation. Plain text strings are the simplest and recommended format.

    Note: HumanToolOutput (in akd/tools/human.py) exists only to satisfy
    BaseTool's generic type signature. It is never instantiated at runtime.
    HumanResponse is the actual type callers construct for resumption.

    Example:
        run_context = RunContext(
            messages=event.run_context.messages,
            human_response=HumanResponse(
                tool_call_id=event.data.tool_call_id,
                content="I mean ML transformers like BERT and GPT",
            ),
        )

        async for event in agent.astream(input_data, run_context=run_context):
            ...
    """

    tool_name: str = Field(default="ask_human", description="Tool name (defaults to ask_human)")


class RunUsage(BaseModel):
    """Token usage statistics for an agent run.

    Accumulates across tool loop iterations via ``incr()``.
    Inspired by pydantic-ai's RunUsage pattern.

    Example:
        usage = RunUsage()
        usage.incr(RunUsage(input_tokens=100, output_tokens=50, requests=1))
        usage.incr(RunUsage(input_tokens=200, output_tokens=80, requests=1))
        print(usage)  # RunUsage(input_tokens=300, output_tokens=130, requests=2, total_tokens=430)
    """

    input_tokens: int = Field(default=0, description="Total input/prompt tokens")
    output_tokens: int = Field(default=0, description="Total output/completion tokens")
    requests: int = Field(default=0, description="Number of LLM API requests")
    details: dict[str, int] = Field(
        default_factory=dict,
        description="Provider-specific extra usage details",
    )

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Sum of input_tokens + output_tokens."""
        return self.input_tokens + self.output_tokens

    def incr(self, other: RunUsage) -> None:
        """Increment usage in place.

        Args:
            other: The usage to add.
        """
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests
        for k, v in other.details.items():
            self.details[k] = self.details.get(k, 0) + v

    def __iadd__(self, other: RunUsage) -> RunUsage:
        """Support ``usage += other_usage`` syntax."""
        self.incr(other)
        return self


class RunContext(BaseModel):
    """Execution context for streaming and tool calling.

    A typed context dict that allows arbitrary extra keys while providing
    type safety for known fields.

    Example:
        context = RunContext(
            human_response=HumanResponse(...),
            messages=[...],
            run_id="abc123",
            custom_field="allowed",  # extra keys work
        )
    """

    model_config = ConfigDict(extra="allow")

    human_response: HumanResponse | None = Field(
        default=None,
        description="Human response for resumption after HUMAN_INPUT_REQUIRED",
    )
    messages: list[Any] | None = Field(
        default=None,
        description=(
            "Conversation history for resumption. AKD's own BaseAgent populates "
            "these as OpenAI-style {role, content} dicts; adapters (pydantic-ai, "
            "langchain, etc.) may carry their native message types instead."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description="Unique identifier for this execution run",
    )
    workflow_id: str | None = Field(
        default=None,
        description="Cross-system workflow identifier (defaults to run_id when omitted)",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for multi-turn workflow grouping",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional user/API request identifier",
    )
    parent_run_id: str | None = Field(
        default=None,
        description="Optional parent run identifier for nested executions",
    )
    control_layer: str | None = Field(
        default=None,
        description="Control-plane executor label (for example litellm)",
    )
    provider_runtime: str | None = Field(
        default=None,
        description="Runtime provider label (for example litellm, openai-agents)",
    )
    repo: str | None = Field(
        default=None,
        description="Repository source label for cross-repo traces",
    )
    usage: RunUsage = Field(
        default_factory=RunUsage,
        description="Accumulated token usage for this run",
    )


__all__ = ["HumanResponse", "RunContext", "RunUsage"]
