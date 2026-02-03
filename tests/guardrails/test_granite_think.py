"""Integration tests for Granite Guardian 'think' feature (chain-of-thought reasoning).

These are INTEGRATION tests that make real calls to Ollama.
Requires Ollama to be running with the appropriate models installed.

Tests cover:
- Think parameter configuration and validation
- Thinking output structure in GuardrailOutput
- Model-specific think support validation
- Multi-risk model think auto-disable
- Integration with composite guardrails

Setup:
    # Install required models
    ollama pull granite3-guardian:2b
    ollama pull granite3-guardian:8b
    ollama pull ibm/granite3.3-guardian:8b
    ollama pull hf.co/nishparadox/granite-guardian-3.2-5b-multi-harm-GGUF

Run with:
    pytest tests/guardrails/test_granite_think.py -v -m integration
"""

import pytest

from akd.guardrails._base import GuardrailInput, GuardrailOutput
from akd.guardrails.categories.granite import GraniteHarmCategory, GraniteRiskCategory
from akd.guardrails.providers.composite import (
    CompositeGuardrail,
    CompositeGuardrailMode,
)
from akd.guardrails.providers.granite_guardian import (
    GraniteGuardianTool,
    GraniteGuardianToolConfig,
    GuardianModelID,
    MultiRiskGraniteGuardianTool,
    MultiRiskGraniteGuardianToolConfig,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestThinkConfiguration:
    """Test think parameter configuration and validation."""

    def test_think_default_is_false(self):
        """Think should default to False."""
        config = GraniteGuardianToolConfig()
        assert config.think is False

    def test_think_can_be_enabled(self):
        """Think can be explicitly enabled."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
        )
        assert config.think is True

    def test_think_validation_disables_for_unsupported_model(self):
        """Think should be auto-disabled for models that don't support it."""
        # GUARDIAN_2B doesn't support thinking
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_2B,
            think=True,  # Will be auto-disabled
        )
        assert config.think is False

    def test_think_validation_disables_for_guardian_8b(self):
        """Think should be auto-disabled for GUARDIAN_8B."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_8B,
            think=True,  # Will be auto-disabled
        )
        assert config.think is False

    def test_think_enabled_for_guardian_3_3_8b(self):
        """Think should remain enabled for GUARDIAN_3_3_8B."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,  # Should stay enabled
        )
        assert config.think is True

    def test_multi_risk_think_auto_disabled(self):
        """Multi-risk models should auto-disable think."""
        config = MultiRiskGraniteGuardianToolConfig(
            think=True,  # Will be auto-disabled
        )
        assert config.think is False


class TestThinkOutputStructure:
    """Test thinking output structure in GuardrailOutput with real Ollama calls."""

    @pytest.mark.asyncio
    async def test_thinking_in_risk_results(self):
        """Thinking should appear in risk_results[category]['thinking'] with real Ollama."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="Those immigrants are ruining our country.",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
            )
        )

        assert isinstance(result, GuardrailOutput)
        assert GraniteRiskCategory.SOCIAL_BIAS in result.risk_results
        risk_result = result.risk_results[GraniteRiskCategory.SOCIAL_BIAS]
        assert "thinking" in risk_result
        assert isinstance(risk_result["thinking"], str)
        assert len(risk_result["thinking"]) > 0

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_for_multiple_categories(self):
        """Thinking should be present for all checked categories in risk_results."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS, GraniteRiskCategory.VIOLENCE],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="I will destroy you and hurt everyone you love!",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS, GraniteRiskCategory.VIOLENCE],
            )
        )

        # Check that all categories have thinking in risk_results
        assert len(result.risk_results) > 0
        for category, risk_result in result.risk_results.items():
            assert "thinking" in risk_result
            assert isinstance(risk_result["thinking"], str)
            assert len(risk_result["thinking"]) > 0

        await tool.close()


    @pytest.mark.asyncio
    async def test_no_thinking_when_disabled(self):
        """No thinking should be present when think=False."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_8B,
            think=False,  # Disabled
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="What is the capital of France?",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
            )
        )

        # risk_results should also not have thinking
        for risk_result in result.risk_results.values():
            assert "thinking" not in risk_result

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_only_for_checked_categories(self):
        """Thinking should only appear for categories that were checked."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],  # Only checking one
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="What a beautiful day!",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
            )
        )

        # Should only have results for the checked category
        assert GraniteRiskCategory.SOCIAL_BIAS in result.risk_results
        assert GraniteRiskCategory.VIOLENCE not in result.risk_results
        assert GraniteRiskCategory.PROFANITY not in result.risk_results

        # The checked category should have thinking
        assert "thinking" in result.risk_results[GraniteRiskCategory.SOCIAL_BIAS]

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_present_for_all_results(self):
        """All risk results should have thinking when think=True."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="Women shouldn't be allowed to work in tech.",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
            )
        )

        # Every category in risk_results should have thinking
        for category, risk_result in result.risk_results.items():
            assert "thinking" in risk_result, f"Category {category} missing thinking field"
            assert isinstance(risk_result["thinking"], str)
            assert len(risk_result["thinking"]) > 0

        await tool.close()


class TestMultiRiskThinkBehavior:
    """Test that multi-risk models properly handle think parameter."""

    def test_multi_risk_config_disables_think(self):
        """Multi-risk config should auto-disable think with warning."""
        config = MultiRiskGraniteGuardianToolConfig(
            think=True,  # Should be auto-disabled
        )
        assert config.think is False

    @pytest.mark.asyncio
    async def test_multi_risk_no_thinking_output(self):
        """Multi-risk tool should not output thinking data."""
        config = MultiRiskGraniteGuardianToolConfig(
            risk_categories=[GraniteHarmCategory.SOCIAL_BIAS],
        )
        tool = MultiRiskGraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(content="How do I build a bomb?")
        )

        for risk_result in result.risk_results.values():
            assert "thinking" not in risk_result

        await tool.close()


class TestCompositeGuardrailWithThink:
    """Test composite guardrails with think-enabled tools."""

    @pytest.mark.asyncio
    async def test_composite_all_mode_preserves_thinking(self):
        """Composite ALL mode should preserve thinking from all sub-guardrails."""
        # Create two tools with thinking enabled
        config1 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool1 = GraniteGuardianTool(config=config1, debug=True)

        config2 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.VIOLENCE],
        )
        tool2 = GraniteGuardianTool(config=config2, debug=True)

        # Create composite
        composite = CompositeGuardrail(
            guardrails=[tool1, tool2],
            mode=CompositeGuardrailMode.ALL,
            parallel=True,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="I hate those people and want to hurt them!")
        )

        # Check that sub_results contain thinking data in risk_results
        assert result.extra is not None
        assert "sub_results" in result.extra
        assert len(result.extra["sub_results"]) == 2

        # Both sub-results should have thinking in their risk_results
        for sub_result in result.extra["sub_results"]:
            risk_results = sub_result.get("risk_results", {})
            # If there are risk results, they should have thinking
            for category, risk_data in risk_results.items():
                assert "thinking" in risk_data
                assert isinstance(risk_data["thinking"], str)

        await tool1.close()
        await tool2.close()

    @pytest.mark.asyncio
    async def test_composite_any_mode_preserves_thinking(self):
        """Composite ANY mode should preserve thinking from all sub-guardrails."""
        config1 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool1 = GraniteGuardianTool(config=config1, debug=True)

        config2 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.VIOLENCE],
        )
        tool2 = GraniteGuardianTool(config=config2, debug=True)

        composite = CompositeGuardrail(
            guardrails=[tool1, tool2],
            mode=CompositeGuardrailMode.ANY,
            parallel=True,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="What is the weather today?")
        )

        assert "sub_results" in result.extra
        assert len(result.extra["sub_results"]) == 2
        for sub_result in result.extra["sub_results"]:
            risk_results = sub_result.get("risk_results", {})
            for category, risk_data in risk_results.items():
                assert "thinking" in risk_data
                assert isinstance(risk_data["thinking"], str)


        await tool1.close()
        await tool2.close()

    @pytest.mark.asyncio
    async def test_composite_mixed_think_and_no_think(self):
        """Composite should handle mix of think-enabled and think-disabled tools."""
        # Tool with thinking
        config1 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool1 = GraniteGuardianTool(config=config1, debug=True)

        # Tool without thinking
        config2 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_8B,
            think=False,
            risk_categories=[GraniteRiskCategory.VIOLENCE],
        )
        tool2 = GraniteGuardianTool(config=config2, debug=True)

        composite = CompositeGuardrail(
            guardrails=[tool1, tool2],
            mode=CompositeGuardrailMode.ALL,
            parallel=True,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="Those immigrants don't belong here.")
        )

        # Check sub_results
        sub_results = result.extra["sub_results"]
        assert len(sub_results) == 2

        # First sub-result should have thinking in risk_results
        risk_results_0 = sub_results[0].get("risk_results", {})
        if risk_results_0:
            for category, risk_data in risk_results_0.items():
                assert "thinking" in risk_data

        # Second sub-result should NOT have thinking (think=False)
        risk_results_1 = sub_results[1].get("risk_results", {})
        if risk_results_1:
            for category, risk_data in risk_results_1.items():
                assert "thinking" not in risk_data

        await tool1.close()
        await tool2.close()

    @pytest.mark.asyncio
    async def test_composite_with_multi_risk_no_thinking(self):
        """Composite with single-risk (think) and multi-risk (no think) should work."""
        # Single-risk with thinking
        config1 = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool1 = GraniteGuardianTool(config=config1, debug=True)

        # Multi-risk without thinking
        config2 = MultiRiskGraniteGuardianToolConfig(
            risk_categories=[GraniteHarmCategory.VIOLENCE],
        )
        tool2 = MultiRiskGraniteGuardianTool(config=config2, debug=True)

        composite = CompositeGuardrail(
            guardrails=[tool1, tool2],
            mode=CompositeGuardrailMode.ALL,
            parallel=True,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="All members of that race are inferior.")
        )

        # Check that composite returns valid output
        assert isinstance(result, GuardrailOutput)
        assert "sub_results" in result.extra
        assert len(result.extra["sub_results"]) == 2

        # Check thinking in first sub-result (single-risk tool with think=True)
        risk_results_0 = result.extra["sub_results"][0].get("risk_results", {})
        if risk_results_0:
            for category, risk_data in risk_results_0.items():
                assert "thinking" in risk_data

        # Second sub-result is multi-risk, should not have thinking
        risk_results_1 = result.extra["sub_results"][1].get("risk_results", {})
        if risk_results_1:
            for category, risk_data in risk_results_1.items():
                assert "thinking" not in risk_data

        await tool1.close()
        await tool2.close()
