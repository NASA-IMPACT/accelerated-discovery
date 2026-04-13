"""Human-in-the-loop tool for agent interaction."""

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd._base.errors import HumanInputRequired
from akd.tools._base import BaseTool, BaseToolConfig


class HumanToolInput(InputSchema):
    """Input schema for requesting human input."""

    question: str = Field(description="The question or request to present to the human")
    context: str | None = Field(
        default=None,
        description="Additional context to help the human understand the request",
    )
    options: list[str] | None = Field(
        default=None,
        description="Optional list of choices for the human to select from",
    )


class HumanToolOutput(OutputSchema):
    """Output schema for human input response."""

    response: str = Field(description="The human's response")


class HumanToolConfig(BaseToolConfig):
    """Configuration for HumanTool."""

    name: str = Field(default="ask_human", description="Tool name exposed to the LLM")
    description: str = Field(
        default=(
            "Ask a human for input, clarification, or approval. "
            "Use this when you need information only a human can provide, "
            "need clarification on ambiguous instructions, or require approval "
            "before proceeding with an action."
        ),
        description="Tool description for the LLM",
    )


class HumanTool(BaseTool[HumanToolInput, HumanToolOutput]):
    """Tool for requesting human input during agent execution.

    When the LLM calls this tool, execution pauses and a HUMAN_INPUT_REQUIRED
    event is emitted with the full message history. The stream ends gracefully.
    Caller provides human response and resumes with message_history.

    This tool is intercepted by _run_tool_loop - _arun is never called directly.

    Example:
        agent = MyAgent(tools=[HumanTool()])

        async for event in agent.astream(input_data):
            if event.event_type == StreamEventType.HUMAN_INPUT_REQUIRED:
                saved_history = event.message_history
                prompt = event.human_prompt
                tool_call_id = event.data["tool_call_id"]
            elif event.event_type == StreamEventType.COMPLETED:
                result = event.output

        # Resume with human response
        async for event in agent.astream(
            input_data,
            context={
                "message_history": saved_history,
                "human_response": {
                    "tool_call_id": tool_call_id,
                    "content": {"response": user_input}
                }
            }
        ):
            ...
    """

    config_schema = HumanToolConfig
    input_schema = HumanToolInput
    output_schema = HumanToolOutput

    async def _arun(self, params: HumanToolInput, **kwargs) -> HumanToolOutput:
        """Raise HumanInputRequired - tool should be intercepted by agent loop."""
        raise HumanInputRequired(
            prompt=params.question,
            context={"description": params.context} if params.context else None,
            options=params.options,
        )
