"""Behavioral tests for human interrupt flow in agents.

Tests using real LLM calls (no mocking). Skipped if OPENAI_API_KEY not set.
"""

import os

import pytest

from akd._base.streaming import (
    CompletedEvent,
    HumanInputRequiredEvent,
    HumanResponseEvent,
)
from akd._base.tool_calling import HumanResponse, RunContext
from akd.tools.human import HumanTool

from .conftest import LiteLLMTestInputSchema, TestLiteLLMAgent

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@requires_openai
@pytest.mark.asyncio
async def test_human_tool_pause_and_resume(litellm_config):
    """Test full HumanTool flow: pause on ask_human, resume with answer, complete.

    Note: HumanTool expects HumanToolOutput schema with {"response": str} format,
    so content must be a dict matching that schema (injected as tool message).
    """
    config = litellm_config.model_copy()
    config.tools = [HumanTool()]
    config.system_prompt = "You must ask the human for their name using ask_human tool before responding."

    agent = TestLiteLLMAgent(config=config)
    input_data = LiteLLMTestInputSchema(query="Hello, who am I?")

    # First call - should pause for human input
    human_input_event = None
    async for event in agent.astream(input_data):
        if isinstance(event, HumanInputRequiredEvent):
            human_input_event = event
            break

    assert human_input_event is not None, "Should yield HumanInputRequiredEvent"
    assert human_input_event.run_context.messages is not None

    # Resume with human response (dict format matches HumanToolOutput schema)
    resume_context = RunContext(
        run_id=human_input_event.run_context.run_id,
        messages=human_input_event.run_context.messages,
        human_response=HumanResponse(
            tool_call_id=human_input_event.data.tool_call_id,
            content={"response": "My name is Paradox"},  # Dict for tool result
        ),
    )

    got_human_response = False
    got_completed = False
    async for event in agent.astream(input_data, run_context=resume_context):
        if isinstance(event, HumanResponseEvent):
            got_human_response = True
        if isinstance(event, CompletedEvent):
            got_completed = True

    assert got_human_response, "Should emit HumanResponseEvent on resume"
    assert got_completed, "Should complete after resume"


@requires_openai
@pytest.mark.asyncio
async def test_non_tool_agent_with_human_response(litellm_config):
    """Test non-tool agent injects human_response as user message.

    Note: For non-tool agents, human_response content is injected as a "user" message.
    Plain string format is preferred for freeform human input.
    """
    agent = TestLiteLLMAgent(config=litellm_config)  # No tools

    # Simulate resuming with prior conversation + human response
    run_context = RunContext(
        run_id="test",
        messages=[
            {"role": "system", "content": "You are helpful. Just say OK."},
            {"role": "assistant", "content": "What do you need?"},
        ],
        human_response=HumanResponse(
            tool_call_id="test_123",
            content="Just confirm you received this",  # Plain string for non-tool
        ),
    )

    got_human_response = False
    got_completed = False
    async for event in agent.astream(
        LiteLLMTestInputSchema(query="test"),
        run_context=run_context,
    ):
        if isinstance(event, HumanResponseEvent):
            got_human_response = True
        if isinstance(event, CompletedEvent):
            got_completed = True

    assert got_human_response, "Should emit HumanResponseEvent"
    assert got_completed, "Should complete"
