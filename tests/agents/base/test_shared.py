# noqa: F841
"""Shared integration tests for base agents."""

from unittest.mock import MagicMock

import pytest

from akd._base import RunContext
from akd.agents._base import BaseAgentConfig

from .conftest import (
    AgentTestInputSchema,
    AgentTestOutputSchema,
    TestInstructorBaseAgent,
    TestLiteLLMAgent,
    setup_async_mock_response,
)


class TestBaseAgentSharedFunctionality:
    """Test shared functionality between base agents."""

    @pytest.mark.asyncio
    async def test_arun_with_instructor_base_agent(self, mock_instructor_client, test_input):
        """Test full arun execution with InstructorBaseAgent."""
        response_data = {
            "response": "Test response",
            "metadata": {"test": True},
        }
        await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        config = BaseAgentConfig(stateless=False)
        agent = TestInstructorBaseAgent(config=config)

        result = await agent.arun(test_input)

        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test response"
        assert result.metadata == {"test": True}
        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3

    @pytest.mark.asyncio
    async def test_stateless_behavior_default(self, mock_instructor_client):
        """Test that agents are stateless by default."""
        instructor_response_data = {
            "response": "Instructor response",
            "metadata": {},
        }
        await setup_async_mock_response(
            mock_instructor_client,
            instructor_response_data,
            "instructor",
        )

        instructor_agent = TestInstructorBaseAgent()
        assert instructor_agent.stateless is True

        test_input = AgentTestInputSchema(query="test query")
        result = await instructor_agent.arun(test_input)

        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3
        assert isinstance(result, AgentTestOutputSchema)

    def test_schema_validation_integration(self, mock_instructor_client):
        """Test that agents properly validate input/output schemas."""
        agent = TestInstructorBaseAgent()

        assert agent.input_schema == AgentTestInputSchema
        assert agent.output_schema == AgentTestOutputSchema

        valid_input = AgentTestInputSchema(query="test")
        validated = agent._validate_input(valid_input)
        assert isinstance(validated, AgentTestInputSchema)
        assert validated.query == "test"

        input_dict = {"query": "test dict"}
        validated = agent._validate_input(input_dict)
        assert isinstance(validated, AgentTestInputSchema)
        assert validated.query == "test dict"

    def test_configuration_attribute_mapping_across_agents(self, mock_instructor_client):
        """Test that configuration attributes are properly mapped across all agent types."""
        custom_config = BaseAgentConfig(
            model_name="gpt-4o-mini",
            temperature=0.8,
            api_key="custom_key",
            stateless=False,
        )

        instructor_agent = TestInstructorBaseAgent(config=custom_config)
        litellm_agent = TestLiteLLMAgent(
            config=BaseAgentConfig(
                model_name="gpt-4o-mini",
                temperature=0.8,
                api_key="custom_key",
                stateless=False,
                max_tokens=50000,
                trim_ratio=0.75,
                enable_trimming=True,
            ),
        )

        for agent in [instructor_agent, litellm_agent]:
            assert agent.model_name == "gpt-4o-mini"
            assert agent.temperature == 0.8
            assert agent.api_key == "custom_key"
            assert agent.stateless is False


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_none_config_handling(self, mock_instructor_client):
        """Test handling of None config."""
        agent = TestInstructorBaseAgent(config=None)
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_response_type_validation(self, mock_instructor_client):
        """Test that response types are properly validated."""
        invalid_response = MagicMock()
        invalid_response.model_dump.return_value = {"invalid": "response"}
        mock_completion = MagicMock()
        mock_completion.usage = None
        mock_instructor_client.chat.completions.create_with_completion.return_value = (
            invalid_response,
            mock_completion,
        )

        agent = TestInstructorBaseAgent()

        try:
            result = await agent.get_response_async(
                run_context=RunContext(messages=[{"role": "user", "content": "test"}]),
            )
            assert isinstance(result, AgentTestOutputSchema)
        except Exception:
            pass

    def test_cross_agent_compatibility(self, mock_instructor_client):
        """Test that all agents implement the same base interface."""
        instructor_agent = TestInstructorBaseAgent()
        litellm_agent = TestLiteLLMAgent()

        required_methods = [
            "arun",
            "get_response_async",
        ]
        required_properties = [
            "input_schema",
            "output_schema",
        ]

        for agent in [instructor_agent, litellm_agent]:
            for method in required_methods:
                assert hasattr(agent, method), f"{type(agent).__name__} missing {method}"
                assert callable(getattr(agent, method)), f"{type(agent).__name__}.{method} not callable"

            for prop in required_properties:
                assert hasattr(agent, prop), f"{type(agent).__name__} missing {prop}"

    def test_agent_inheritance_hierarchy(self):
        """Test that inheritance hierarchy is correct."""
        from akd.agents._base import (
            AKDAgent,
            BaseAgent,
            InstructorBaseAgent,
            LiteLLMInstructorBaseAgent,
        )

        assert issubclass(TestInstructorBaseAgent, BaseAgent)
        assert issubclass(TestInstructorBaseAgent, AKDAgent)
        assert issubclass(TestLiteLLMAgent, AKDAgent)

        # All aliases point to AKDAgent
        assert InstructorBaseAgent is AKDAgent
        assert LiteLLMInstructorBaseAgent is AKDAgent

    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, mock_instructor_client):
        """Test that agents handle errors gracefully."""
        mock_instructor_client.chat.completions.create_with_completion.side_effect = Exception(
            "Instructor error",
        )

        instructor_agent = TestInstructorBaseAgent()

        test_input = AgentTestInputSchema(query="test query")

        with pytest.raises(Exception) as exc_info:
            await instructor_agent.arun(test_input)
        assert "Instructor error" in str(exc_info.value)
