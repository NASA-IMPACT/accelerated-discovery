"""E2E tests for guardrails integration.

Tests apply_guardrails() and @guardrail decorator with real providers.
- GraniteGuardianTool: Ollama calls mocked
- RiskAgent: Real LLM calls (CI has OPENAI_API_KEY)
"""

import pytest

from akd.errors import InputGuardrailTriggered
from akd.guardrails import apply_guardrails, guardrail
from akd.guardrails.providers import GraniteGuardianTool

from .conftest import TestAgent


class TestApplyGuardrails:
    """Tests for apply_guardrails() function."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_guardrail",
        [
            pytest.param(
                GraniteGuardianTool(),
                id="granite",
            ),
            # RiskAgent uses real LLM calls - skip if too slow/expensive for quick iteration
            # pytest.param(RiskAgent(), id="risk_agent"),
        ],
    )
    async def test_input_guardrail_detects_jailbreak(
        self,
        input_guardrail,
        simple_agent,
        jailbreak_input,
        mock_ollama,
    ):
        """Input guardrail should detect jailbreak attempts."""
        guarded = apply_guardrails(simple_agent, input_guardrail=input_guardrail)
        result = await guarded.arun(jailbreak_input)

        # Verify guardrail result is attached
        assert result.input_guardrail_result is not None
        assert len(result.input_guardrail_result.detected_risks) > 0
        assert result.guardrails_passed is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_guardrail",
        [
            pytest.param(GraniteGuardianTool(), id="granite"),
        ],
    )
    async def test_input_guardrail_passes_safe_content(
        self,
        input_guardrail,
        simple_agent,
        safe_input,
        mock_ollama,
    ):
        """Input guardrail should pass safe content."""
        guarded = apply_guardrails(simple_agent, input_guardrail=input_guardrail)
        result = await guarded.arun(safe_input)

        # Verify guardrail passed
        assert result.input_guardrail_result is not None
        assert result.input_guardrail_result.passed is True
        assert result.guardrails_passed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_guardrail",
        [
            pytest.param(GraniteGuardianTool(), id="granite"),
        ],
    )
    async def test_fail_on_input_risk_raises_exception(
        self,
        input_guardrail,
        simple_agent,
        jailbreak_input,
        mock_ollama,
    ):
        """fail_on_input_risk=True should raise InputGuardrailTriggered."""
        guarded = apply_guardrails(
            simple_agent,
            input_guardrail=input_guardrail,
            fail_on_input_risk=True,
        )

        with pytest.raises(InputGuardrailTriggered) as exc_info:
            await guarded.arun(jailbreak_input)

        # Verify exception contains guardrail output
        assert exc_info.value.output is not None
        assert len(exc_info.value.output.detected_risks) > 0

    @pytest.mark.asyncio
    async def test_no_guardrail_passes_through(self, simple_agent, jailbreak_input):
        """Without guardrails, agent should work normally."""
        guarded = apply_guardrails(simple_agent)  # No guardrails
        result = await guarded.arun(jailbreak_input)

        # No guardrail results since none configured
        assert result.input_guardrail_result is None
        assert result.output_guardrail_result is None
        assert result.guardrails_passed is True

    @pytest.mark.asyncio
    async def test_both_input_and_output_guardrails(
        self,
        simple_agent,
        safe_input,
        mock_ollama,
    ):
        """Both input and output guardrails can be applied."""
        granite = GraniteGuardianTool()
        guarded = apply_guardrails(
            simple_agent,
            input_guardrail=granite,
            output_guardrail=granite,
        )
        result = await guarded.arun(safe_input)

        # Both results should be present
        assert result.input_guardrail_result is not None
        assert result.output_guardrail_result is not None
        assert result.guardrails_passed is True


class TestGuardrailDecorator:
    """Tests for @guardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_applies_input_guardrail(
        self,
        jailbreak_input,
        mock_ollama,
    ):
        """@guardrail decorator should apply input guardrail."""

        @guardrail(input_guardrail=GraniteGuardianTool())
        class DecoratedAgent(TestAgent):
            pass

        DecoratedAgent.__test__ = False

        agent = DecoratedAgent()
        result = await agent.arun(jailbreak_input)

        assert result.input_guardrail_result is not None
        assert len(result.input_guardrail_result.detected_risks) > 0
        assert result.guardrails_passed is False

    @pytest.mark.asyncio
    async def test_decorator_with_fail_on_risk(
        self,
        jailbreak_input,
        mock_ollama,
    ):
        """@guardrail with fail_on_input_risk should raise exception."""

        @guardrail(
            input_guardrail=GraniteGuardianTool(),
            fail_on_input_risk=True,
        )
        class StrictAgent(TestAgent):
            pass

        StrictAgent.__test__ = False

        agent = StrictAgent()
        with pytest.raises(InputGuardrailTriggered):
            await agent.arun(jailbreak_input)

    @pytest.mark.asyncio
    async def test_decorator_passes_safe_content(
        self,
        safe_input,
        mock_ollama,
    ):
        """@guardrail should pass safe content."""

        @guardrail(input_guardrail=GraniteGuardianTool())
        class SafeAgent(TestAgent):
            pass

        SafeAgent.__test__ = False

        agent = SafeAgent()
        result = await agent.arun(safe_input)

        assert result.input_guardrail_result is not None
        assert result.input_guardrail_result.passed is True
        assert result.guardrails_passed is True
