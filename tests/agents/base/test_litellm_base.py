"""Test cases for AKDAgent (formerly LiteLLMInstructorBaseAgent)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd.agents._base import AKDAgent, InstructorBaseAgent, LiteLLMInstructorBaseAgent

from .conftest import (
    LiteLLMTestInputSchema,
    LiteLLMTestOutputSchema,
    TestLiteLLMAgent,
    create_config_with_overrides,
)


class TestLiteLLMInstructorBaseAgent:
    """Test suite for LiteLLMInstructorBaseAgent."""

    def test_initialization(self, litellm_config):
        """Test that the agent initializes correctly."""
        agent = TestLiteLLMAgent(config=litellm_config, debug=True)

        assert agent.model_name == "gpt-4o-mini"
        assert agent.temperature == 0.1
        assert agent.max_tokens == 50000
        assert agent.trim_ratio == 0.75
        assert agent.enable_trimming is True
        assert agent.stateless is True

        # Check that the client is properly initialized
        assert agent.client is not None

    def test_config_access(self, litellm_config):
        """Test that config fields are accessible via self.attribute."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # These should work because they're dynamically set from config
        assert hasattr(agent, "max_tokens")
        assert hasattr(agent, "trim_ratio")
        assert hasattr(agent, "enable_trimming")

        assert agent.max_tokens == 50000
        assert agent.trim_ratio == 0.75
        assert agent.enable_trimming is True

    @pytest.mark.asyncio
    @patch("akd._base.session.trim_messages")
    @patch("instructor.from_litellm")
    async def test_full_arun_workflow(
        self,
        mock_instructor: MagicMock,
        mock_trim_messages: MagicMock,
        litellm_config,
    ):
        """Test the complete _arun workflow."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # Setup mocks
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "response": "Processed query: test query",
            "confidence": 0.95,
        }
        # create_with_completion returns (response, completion) tuple
        mock_completion = MagicMock()
        mock_completion.usage = None
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(mock_response, mock_completion),
        )

        agent.client = mock_client

        # Mock trim_messages to pass through
        mock_trim_messages.side_effect = lambda messages, **kwargs: messages

        # Create input
        test_input = LiteLLMTestInputSchema(
            query="test query",
            context="test context",
        )

        # Run the agent
        result = await agent.arun(test_input)

        # Verify result
        assert isinstance(result, LiteLLMTestOutputSchema)
        assert result.response == "Processed query: test query"
        assert result.confidence == 0.95

    def test_backward_compatibility_aliases(self, litellm_config):
        """Test that AKDAgent backward compatibility aliases work."""
        # All aliases point to the same class
        assert InstructorBaseAgent is AKDAgent
        assert LiteLLMInstructorBaseAgent is AKDAgent

        # Create agents via alias
        class TestInstructorAgent(
            InstructorBaseAgent[LiteLLMTestInputSchema, LiteLLMTestOutputSchema],
        ):
            input_schema = LiteLLMTestInputSchema
            output_schema = LiteLLMTestOutputSchema

        litellm_agent = TestLiteLLMAgent(config=litellm_config)
        instructor_agent = TestInstructorAgent(config=litellm_config)

        # Verify they have the same interface
        assert hasattr(litellm_agent, "arun")
        assert hasattr(litellm_agent, "get_response_async")
        # Verify same attributes exist
        for attr in ["model_name", "temperature", "stateless", "api_key"]:
            assert hasattr(litellm_agent, attr)
            assert hasattr(instructor_agent, attr)
            assert getattr(litellm_agent, attr) == getattr(instructor_agent, attr)

    def test_custom_token_limits(self, litellm_config):
        """Test that custom token limits are properly configured."""
        custom_config = create_config_with_overrides(
            litellm_config,
            max_tokens=25000,
            trim_ratio=0.6,
            enable_trimming=False,
        )

        agent = TestLiteLLMAgent(config=custom_config)

        assert agent.max_tokens == 25000
        assert agent.trim_ratio == 0.6
        assert agent.enable_trimming is False

    def test_litellm_client_initialization(self, litellm_config):
        """Test that LiteLLM client is properly initialized."""
        with patch("instructor.from_litellm") as mock_from_litellm:
            with patch("akd.agents._base._base.acompletion") as mock_acompletion:
                mock_client = MagicMock()
                mock_from_litellm.return_value = mock_client

                agent = TestLiteLLMAgent(config=litellm_config)

                # Verify LiteLLM instructor client was created
                mock_from_litellm.assert_called_once_with(mock_acompletion)
                assert agent.client == mock_client

    def test_token_management_configuration(self, litellm_config):
        """Test token management specific configuration."""
        # Test various token configurations
        test_configs = [
            {"max_tokens": 10000, "trim_ratio": 0.5, "enable_trimming": True},
            {"max_tokens": 100000, "trim_ratio": 0.9, "enable_trimming": False},
            {"max_tokens": 1000, "trim_ratio": 0.1, "enable_trimming": True},
        ]

        for config_overrides in test_configs:
            config = create_config_with_overrides(litellm_config, **config_overrides)
            agent = TestLiteLLMAgent(config=config)

            assert agent.max_tokens == config_overrides["max_tokens"]
            assert agent.trim_ratio == config_overrides["trim_ratio"]
            assert agent.enable_trimming == config_overrides["enable_trimming"]
