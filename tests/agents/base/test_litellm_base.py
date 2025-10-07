"""Test cases for LiteLLMInstructorBaseAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd.agents._base import InstructorBaseAgent

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

        # Check memory initialization
        assert isinstance(agent.memory, list)
        assert len(agent.memory) == 0

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

    @patch("akd.agents._base.trim_messages")
    @patch("instructor.from_litellm")
    async def test_message_trimming_enabled(
        self,
        mock_instructor: MagicMock,
        mock_trim_messages: MagicMock,
        litellm_config,
    ):
        """Test that message trimming is called when enabled."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # Setup mocks
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "response": "Test response",
            "confidence": 0.9,
        }
        mock_client.chat.completions.create.return_value = mock_response

        # Mock trim_messages to return the same messages (simulate no trimming needed)
        test_messages = [{"role": "user", "content": "test"}]
        mock_trim_messages.return_value = test_messages

        # Re-initialize agent to use mocked instructor
        agent.client = mock_instructor.return_value

        # Call the method
        result = await agent.get_response_async(test_messages)

        # Verify trim_messages was called with correct parameters
        mock_trim_messages.assert_called_once_with(
            test_messages,
            model=agent.model_name,
            max_tokens=agent.max_tokens,
            trim_ratio=agent.trim_ratio,
        )

        # Verify the response
        assert result.response == "Test response"
        assert result.confidence == 0.9

    @patch("instructor.from_litellm")
    async def test_message_trimming_disabled(
        self,
        mock_instructor: MagicMock,
        litellm_config,
    ):
        """Test that message trimming is skipped when disabled."""
        # Create config with trimming disabled
        config = create_config_with_overrides(litellm_config, enable_trimming=False)
        agent = TestLiteLLMAgent(config=config)

        # Setup mocks
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "response": "Test response",
            "confidence": 0.9,
        }
        mock_client.chat.completions.create.return_value = mock_response

        agent.client = mock_instructor.return_value

        test_messages = [{"role": "user", "content": "test"}]

        with patch("akd.agents._base.trim_messages") as mock_trim_messages:
            result = await agent.get_response_async(test_messages)

            # Verify trim_messages was NOT called
            mock_trim_messages.assert_not_called()

            # Verify the response
            assert result.response == "Test response"
            assert result.confidence == 0.9

    @patch("instructor.from_litellm")
    async def test_full_arun_workflow(
        self,
        mock_instructor: MagicMock,
        litellm_config,
    ):
        """Test the complete _arun workflow with message trimming."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # Setup mocks
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "response": "Processed query: test query",
            "confidence": 0.95,
        }
        mock_client.chat.completions.create.return_value = mock_response

        agent.client = mock_instructor.return_value

        # Create input
        test_input = LiteLLMTestInputSchema(
            query="test query",
            context="test context",
        )

        with patch("akd.agents._base.trim_messages") as mock_trim_messages:
            # Mock trim_messages to return shortened messages
            mock_trim_messages.side_effect = lambda messages, **kwargs: messages[
                :1
            ]  # Keep only first message

            # Run the agent
            result = await agent.arun(test_input)

            # Verify result
            assert isinstance(result, LiteLLMTestOutputSchema)
            assert result.response == "Processed query: test query"
            assert result.confidence == 0.95

            # Verify trim_messages was called
            mock_trim_messages.assert_called()

    def test_backward_compatibility_with_instructor_base_agent(self, litellm_config):
        """Test that LiteLLMInstructorBaseAgent maintains compatibility with InstructorBaseAgent."""

        # Create both agents
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
        assert hasattr(litellm_agent, "memory")
        assert hasattr(litellm_agent, "reset_memory")

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

    @patch("instructor.from_litellm")
    async def test_large_context_handling(
        self,
        mock_instructor: MagicMock,
        litellm_config,
    ):
        """Test that large contexts are properly handled with trimming."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # Setup mocks
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "response": "Handled large context",
            "confidence": 0.8,
        }
        mock_client.chat.completions.create.return_value = mock_response

        agent.client = mock_instructor.return_value

        # Create large message history (simulate the original token overflow scenario)
        large_messages = []
        for i in range(100):
            large_messages.append(
                {
                    "role": "user",
                    "content": f"This is a very long message {i} " + "x" * 1000,
                },
            )
            large_messages.append(
                {
                    "role": "assistant",
                    "content": f"Response to message {i} " + "y" * 1000,
                },
            )

        with patch("akd.agents._base.trim_messages") as mock_trim_messages:
            # Simulate trimming to a reasonable size
            mock_trim_messages.return_value = large_messages[
                :10
            ]  # Keep only first 10 messages

            result = await agent.get_response_async(large_messages)

            # Verify trimming was called
            mock_trim_messages.assert_called_once_with(
                large_messages,
                model=agent.model_name,
                max_tokens=agent.max_tokens,
                trim_ratio=agent.trim_ratio,
            )

            # Verify client was called with trimmed messages
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args.kwargs["messages"]) == 10

            assert result.response == "Handled large context"

    def test_memory_management_stateful(self, litellm_config):
        """Test memory management in stateful mode."""
        config = create_config_with_overrides(litellm_config, stateless=False)
        agent = TestLiteLLMAgent(config=config)

        # Initially empty
        assert len(agent.memory) == 0

        # Should maintain memory when stateless=False
        assert agent.stateless is False

    def test_memory_management_stateless(self, litellm_config):
        """Test memory management in stateless mode."""
        agent = TestLiteLLMAgent(config=litellm_config)

        # Should be stateless by default
        assert agent.stateless is True

        # Memory should be empty
        assert len(agent.memory) == 0

    def test_litellm_client_initialization(self, litellm_config):
        """Test that LiteLLM client is properly initialized."""
        with patch("instructor.from_litellm") as mock_from_litellm:
            with patch("akd.agents._base.acompletion") as mock_acompletion:
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

    async def test_large_input_exception_when_trimming_disabled(self, litellm_config):
        """Test that large input throws exception when trimming is disabled."""
        # Create config with trimming disabled and very low token limit
        config = create_config_with_overrides(
            litellm_config,
            enable_trimming=False,
            max_tokens=100,
        )
        agent = TestLiteLLMAgent(config=config)

        # Create very large input that will exceed token limits
        large_input = LiteLLMTestInputSchema(
            query="love " * 1000000,
            context="test context",
        )

        # Should throw exception due to token limit exceeded
        with pytest.raises(Exception):
            await agent.arun(large_input)

    async def test_large_input_success_when_trimming_enabled(self, litellm_config):
        """Test that large input succeeds when trimming is enabled."""
        # Create config with trimming enabled and same low token limit
        config = create_config_with_overrides(
            litellm_config,
            enable_trimming=True,
            max_tokens=100,
        )
        agent = TestLiteLLMAgent(config=config)

        # Create same very large input
        large_input = LiteLLMTestInputSchema(
            query="love " * 1000000,
            context="test context",
        )

        # Should succeed without throwing exception due to automatic trimming
        result = await agent.arun(large_input)

        # Verify we get a valid response
        assert isinstance(result, LiteLLMTestOutputSchema)
        assert hasattr(result, "response")
        assert hasattr(result, "confidence")
