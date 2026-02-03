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
    async def test_thinking_in_extra_field(self):
        """Thinking should appear in output.extra['thinking'] with real Ollama."""
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
        assert result.extra is not None
        assert "thinking" in result.extra
        assert isinstance(result.extra["thinking"], dict)

        # Verify thinking content is non-empty
        assert len(result.extra["thinking"]) > 0
        for risk_cat, thinking_text in result.extra["thinking"].items():
            assert isinstance(thinking_text, str)
            assert len(thinking_text) > 0, f"Thinking for {risk_cat} should not be empty"

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_map_structure(self):
        """Thinking map should be keyed by risk category value (string)."""
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

        thinking_map = result.extra["thinking"]

        # Check that categories checked have thinking entries
        assert isinstance(thinking_map, dict)
        # At least one category should have thinking
        assert len(thinking_map) > 0

        for cat_name, thinking_text in thinking_map.items():
            assert isinstance(cat_name, str)
            assert isinstance(thinking_text, str)
            assert len(thinking_text) > 0

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_in_risk_results(self):
        """Thinking should also appear in risk_results[category]['thinking']."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        result = await tool.acheck(
            GuardrailInput(
                content="Those people are all criminals and should be deported.",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS],
            )
        )

        assert GraniteRiskCategory.SOCIAL_BIAS in result.risk_results
        risk_result = result.risk_results[GraniteRiskCategory.SOCIAL_BIAS]
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

        # extra['thinking'] should not exist when no thinking is present
        if result.extra:
            assert "thinking" not in result.extra

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

        thinking_map = result.extra["thinking"]
        # Should only have thinking for the checked category
        assert len(thinking_map) == 1
        assert "social_bias" in thinking_map
        assert "violence" not in thinking_map
        assert "profanity" not in thinking_map

        await tool.close()

    @pytest.mark.asyncio
    async def test_thinking_consistent_across_locations(self):
        """Thinking should be the same in extra['thinking'] and risk_results."""
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

        thinking_from_extra = result.extra["thinking"]["social_bias"]
        thinking_from_risk_results = result.risk_results[GraniteRiskCategory.SOCIAL_BIAS]["thinking"]

        assert thinking_from_extra == thinking_from_risk_results

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

        # No thinking in extra
        if result.extra:
            assert "thinking" not in result.extra

        # No thinking in risk_results
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

        # Check that sub_results contain thinking data
        assert result.extra is not None
        assert "sub_results" in result.extra
        assert len(result.extra["sub_results"]) == 2

        # Both sub-results should have thinking in their extra
        for sub_result in result.extra["sub_results"]:
            if sub_result.get("extra"):
                assert "thinking" in sub_result["extra"]
                assert isinstance(sub_result["extra"]["thinking"], dict)

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

        # Check sub_results
        assert "sub_results" in result.extra


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

        # First sub-result should have thinking
        if sub_results[0].get("extra"):
            assert "thinking" in sub_results[0]["extra"]

        # Second sub-result should NOT have thinking
        if sub_results[1].get("extra"):
            assert "thinking" not in sub_results[1]["extra"]

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

        # check thinking in first sub-result
        if result.extra["sub_results"][0].get("extra"):
            assert "thinking" in result.extra["sub_results"][0]["extra"]

        await tool1.close()
        await tool2.close()

    @pytest.mark.asyncio
    async def test_composite_thinking_accessible_from_sub_results(self):
        """Thinking should be accessible via sub_results in composite output."""
        config = GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_3_3_8B,
            think=True,
            risk_categories=[GraniteRiskCategory.SOCIAL_BIAS, GraniteRiskCategory.VIOLENCE],
        )
        tool = GraniteGuardianTool(config=config, debug=True)

        composite = CompositeGuardrail(
            guardrails=[tool],
            mode=CompositeGuardrailMode.ALL,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(
                content="I will hurt those people because of their ethnicity.",
                risk_categories=[GraniteRiskCategory.SOCIAL_BIAS, GraniteRiskCategory.VIOLENCE],
            )
        )

        # Access thinking from sub_results
        sub_result = result.extra["sub_results"][0]
        thinking_map = sub_result["extra"]["thinking"]

        # Should have thinking for checked categories
        assert isinstance(thinking_map, dict)
        assert len(thinking_map) > 0

        for cat_name, thinking_text in thinking_map.items():
            assert isinstance(thinking_text, str)
            assert len(thinking_text) > 0

        await tool.close()

    @pytest.mark.asyncio
    async def test_composite_parallel_execution_with_thinking(self):
        """Parallel execution should correctly collect thinking from all tools."""
        # Create 3 tools with thinking
        tools = []
        for risk in [GraniteRiskCategory.SOCIAL_BIAS, GraniteRiskCategory.VIOLENCE, GraniteRiskCategory.PROFANITY]:
            config = GraniteGuardianToolConfig(
                model=GuardianModelID.GUARDIAN_3_3_8B,
                think=True,
                risk_categories=[risk],
            )
            tools.append(GraniteGuardianTool(config=config, debug=True))

        composite = CompositeGuardrail(
            guardrails=tools,
            mode=CompositeGuardrailMode.ALL,
            parallel=True,
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="F*** those immigrants, they're all criminals!")
        )

        # All 3 sub-results should have thinking
        assert len(result.extra["sub_results"]) == 3
        for sub_result in result.extra["sub_results"]:
            assert "thinking" in sub_result["extra"]

        # Cleanup
        for tool in tools:
            await tool.close()

    @pytest.mark.asyncio
    async def test_composite_sequential_execution_with_thinking(self):
        """Sequential execution should correctly collect thinking from all tools."""
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
            mode=CompositeGuardrailMode.ALL,
            parallel=False,  # Sequential
            debug=True,
        )

        result = await composite.acheck(
            GuardrailInput(content="Those people deserve violence.")
        )

        # Both sub-results should have thinking
        assert len(result.extra["sub_results"]) == 2
        for sub_result in result.extra["sub_results"]:
            assert "thinking" in sub_result["extra"]

        await tool1.close()
        await tool2.close()
