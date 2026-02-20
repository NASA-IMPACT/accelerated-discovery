"""Functional tests for tool calling with MathAgent."""

import re

import pytest

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.tools._base import BaseTool

# === Tools ===


class AddToolInput(InputSchema):
    """Add input."""

    a: float
    b: float


class AddToolOutput(OutputSchema):
    """Add output."""

    result: float


class AddTool(BaseTool):
    """Add two numbers."""

    input_schema = AddToolInput
    output_schema = AddToolOutput

    async def _arun(self, params: AddToolInput) -> AddToolOutput:
        return AddToolOutput(result=params.a + params.b)


class MultiplyToolInput(InputSchema):
    """Multiply input."""

    a: float
    b: float


class MultiplyToolOutput(OutputSchema):
    """Multiply output."""

    result: float


class MultiplyTool(BaseTool):
    """Multiply two numbers."""

    input_schema = MultiplyToolInput
    output_schema = MultiplyToolOutput

    async def _arun(self, params: MultiplyToolInput) -> MultiplyToolOutput:
        return MultiplyToolOutput(result=params.a * params.b)


# === Agent ===


class MathInput(InputSchema):
    """Math query."""

    query: str


class MathOutput(OutputSchema):
    """Math result."""

    result: str


class MathAgent(LiteLLMInstructorBaseAgent):
    """Math agent."""

    input_schema = MathInput
    output_schema = MathOutput


MathAgent.__test__ = False


def _tools_were_called(messages: list[dict]) -> bool:
    """Check if any tools were called by inspecting run-context messages."""
    for msg in messages:
        if msg.get("role") == "tool":
            return True
    return False


# === Fixtures ===


@pytest.fixture
def math_agent():
    config = BaseAgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        system_prompt=(
            "You are a math assistant. Always use the provided tools for calculations. "
            "Never compute in your head - use the Add and Multiply tools."
        ),
        tools=[AddTool(), MultiplyTool()],
        max_tool_iterations=10,
        stateless=False,  # Keep history in run_context to verify tool calls
    )
    return MathAgent(config)


# === Tests ===


@pytest.mark.integration
class TestMathAgent:
    """Functional IO tests: query → expected result."""

    @pytest.mark.asyncio
    async def test_1_plus_1_equals_2(self, math_agent):
        result = await math_agent.arun(MathInput(query="What is 1 + 1?"))
        assert "2" in result.result
        assert result.run_context is not None and result.run_context.messages is not None
        assert _tools_were_called(result.run_context.messages)

    @pytest.mark.asyncio
    async def test_5_plus_3_equals_8(self, math_agent):
        result = await math_agent.arun(MathInput(query="What is 5 + 3?"))
        assert "8" in result.result
        assert result.run_context is not None and result.run_context.messages is not None
        assert _tools_were_called(result.run_context.messages)

    @pytest.mark.asyncio
    async def test_3_times_4_equals_12(self, math_agent):
        result = await math_agent.arun(MathInput(query="What is 3 * 4?"))
        assert "12" in result.result
        assert result.run_context is not None and result.run_context.messages is not None
        assert _tools_were_called(result.run_context.messages)

    @pytest.mark.asyncio
    async def test_5_plus_3_times_2_equals_16(self, math_agent):
        """(5 + 3) * 2 = 16"""
        result = await math_agent.arun(
            MathInput(query="Add 5 and 3, then multiply by 2"),
        )
        assert "16" in result.result
        assert result.run_context is not None and result.run_context.messages is not None
        assert _tools_were_called(result.run_context.messages)

    @pytest.mark.asyncio
    async def test_complex_add_multiply(self, math_agent):
        """Complex add/multiply calculation with large decimals.

        (12345.67 + 89012.34) * 56.789 = 101358.01 * 56.789 ≈ 5755855.74
        """
        result = await math_agent.arun(
            MathInput(
                query="Add 12345.67 and 89012.34, then multiply by 56.789",
            ),
        )
        # Find all numbers and check if any is within 1% of expected
        numbers = re.findall(r"[\d]+\.?\d*", result.result)
        assert numbers, f"No number found in: {result.result}"
        expected = 5755855.74
        assert any(float(n) == pytest.approx(expected, rel=0.01) for n in numbers), (
            f"Expected ~{expected}, got: {result.result}"
        )
        assert result.run_context is not None and result.run_context.messages is not None
        assert _tools_were_called(result.run_context.messages)
