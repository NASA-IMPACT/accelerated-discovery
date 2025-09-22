"""Test cases for LiteLLMInstructorBaseAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent


# Test schemas
class LiteLLMTestInputSchema(InputSchema):
    """Test input schema for LiteLLM agent."""

    query: str = Field(..., description="Test query input")
    context: str = Field(default="", description="Optional context")


class LiteLLMTestOutputSchema(OutputSchema):
    """Test output schema for LiteLLM agent."""

    response: str = Field(..., description="Test response output")
    confidence: float = Field(default=0.8, description="Response confidence")


# Test agent implementation
class TestLiteLLMAgent(
    LiteLLMInstructorBaseAgent[LiteLLMTestInputSchema, LiteLLMTestOutputSchema],
):
    """Test implementation of LiteLLMInstructorBaseAgent."""

    input_schema = LiteLLMTestInputSchema
    output_schema = LiteLLMTestOutputSchema


class TestLiteLLMInstructorBaseAgent:
    """Test suite for LiteLLMInstructorBaseAgent."""

    @pytest.fixture
    def default_config(self) -> BaseAgentConfig:
        """Create a default configuration for testing."""
        return BaseAgentConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=50000,
            trim_ratio=0.75,
            enable_trimming=True,
            stateless=True,
        )

    @pytest.fixture
    def agent(self, default_config: BaseAgentConfig) -> TestLiteLLMAgent:
        """Create a test agent instance."""
        return TestLiteLLMAgent(config=default_config, debug=True)

    def test_initialization(
        self,
        agent: TestLiteLLMAgent,
        default_config: BaseAgentConfig,
    ):
        """Test that the agent initializes correctly."""
        assert agent.model_name == "gpt-3.5-turbo"
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

    def test_config_access(self, agent: TestLiteLLMAgent):
        """Test that config fields are accessible via self.attribute."""
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
        agent: TestLiteLLMAgent,
    ):
        """Test that message trimming is called when enabled."""
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
        default_config: BaseAgentConfig,
    ):
        """Test that message trimming is skipped when disabled."""
        # Create config with trimming disabled
        config_dict = default_config.model_dump()
        config_dict["enable_trimming"] = False
        config = BaseAgentConfig(**config_dict)
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
        agent: TestLiteLLMAgent,
    ):
        """Test the complete _arun workflow with message trimming."""
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

    def test_backward_compatibility_with_instructor_base_agent(
        self,
        default_config: BaseAgentConfig,
    ):
        """Test that LiteLLMInstructorBaseAgent maintains compatibility with InstructorBaseAgent."""
        from akd.agents._base import InstructorBaseAgent

        # Create both agents
        class TestInstructorAgent(
            InstructorBaseAgent[LiteLLMTestInputSchema, LiteLLMTestOutputSchema],
        ):
            input_schema = LiteLLMTestInputSchema
            output_schema = LiteLLMTestOutputSchema

        litellm_agent = TestLiteLLMAgent(config=default_config)
        instructor_agent = TestInstructorAgent(config=default_config)

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

    def test_custom_token_limits(self, default_config: BaseAgentConfig):
        """Test that custom token limits are properly configured."""
        config_dict = default_config.model_dump()
        config_dict.update(
            {
                "max_tokens": 25000,
                "trim_ratio": 0.6,
                "enable_trimming": False,
            },
        )
        custom_config = BaseAgentConfig(**config_dict)

        agent = TestLiteLLMAgent(config=custom_config)

        assert agent.max_tokens == 25000
        assert agent.trim_ratio == 0.6
        assert agent.enable_trimming is False

    @patch("instructor.from_litellm")
    async def test_large_context_handling(
        self,
        mock_instructor: MagicMock,
        agent: TestLiteLLMAgent,
    ):
        """Test that large contexts are properly handled with trimming."""
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

    def test_memory_management_stateful(self, default_config: BaseAgentConfig):
        """Test memory management in stateful mode."""
        config_dict = default_config.model_dump()
        config_dict["stateless"] = False
        config = BaseAgentConfig(**config_dict)
        agent = TestLiteLLMAgent(config=config)

        # Initially empty
        assert len(agent.memory) == 0

        # Should maintain memory when stateless=False
        assert agent.stateless is False

    def test_memory_management_stateless(self, agent: TestLiteLLMAgent):
        """Test memory management in stateless mode."""
        # Should be stateless by default
        assert agent.stateless is True

        # Memory should be empty
        assert len(agent.memory) == 0
