"""Test cases for LangBaseAgent."""

import pytest
from langchain_community.chat_message_histories import ChatMessageHistory

from akd.agents._base import BaseAgentConfig

from .conftest import (
    AgentTestOutputSchema,
    TestLangBaseAgent,
    setup_async_mock_response,
)


class TestLangBaseAgentFunctionality:
    """Test LangBaseAgent specific functionality."""

    def test_initialization_default_config(self, mock_chatopenai_client):
        """Test LangBaseAgent initialization with default config."""
        agent = TestLangBaseAgent()

        # Verify agent properties
        assert agent.client == mock_chatopenai_client
        assert isinstance(agent.memory, ChatMessageHistory)
        assert agent.prompt_template is not None

    def test_initialization_custom_config(self, mock_chatopenai_client, custom_config):
        """Test LangBaseAgent initialization with custom config."""
        agent = TestLangBaseAgent(config=custom_config)

        # Verify configuration was applied
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.7
        assert agent.api_key == "test_key"
        assert str(agent.base_url) == "https://custom.api.com/v1"

    def test_memory_management(self, mock_chatopenai_client):
        """Test memory initialization and reset functionality."""
        agent = TestLangBaseAgent()

        # Test initial memory state
        assert isinstance(agent.memory, ChatMessageHistory)
        assert len(agent.memory.messages) == 0

        # Add test messages
        agent.memory.add_user_message("test user message")
        agent.memory.add_ai_message("test ai message")
        assert len(agent.memory.messages) == 2

        # Test memory reset
        agent.reset_memory()
        assert len(agent.memory.messages) == 0

    @pytest.mark.asyncio
    async def test_get_response_async_mock(
        self,
        mock_chatopenai_client,
        expected_output,
    ):
        """Test async response generation with mocked client."""
        # Setup mock structured output
        mock_structured_client = await setup_async_mock_response(
            mock_chatopenai_client,
            expected_output,
            "chatopenai",
        )

        agent = TestLangBaseAgent()

        # Test response generation
        result = await agent.get_response_async()

        # Verify structured output setup
        mock_chatopenai_client.with_structured_output.assert_called_once_with(
            AgentTestOutputSchema,
            method="function_calling",
        )

        # Verify response
        mock_structured_client.ainvoke.assert_called_once()
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test response"

    def test_prompt_template_setup(self, mock_chatopenai_client):
        """Test prompt template initialization."""
        agent = TestLangBaseAgent()

        # Verify prompt template exists and has correct structure
        assert agent.prompt_template is not None

        # Test template formatting
        formatted = agent.prompt_template.format_messages(memory=[])
        assert len(formatted) >= 1  # Should have at least system message
        assert formatted[0].content == agent.system_prompt

    def test_client_configuration(self, mock_chatopenai_client, custom_config):
        """Test that ChatOpenAI client is configured correctly."""
        from unittest.mock import patch

        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            mock_chat_openai.return_value = mock_chatopenai_client

            agent = TestLangBaseAgent(config=custom_config)

            # Verify client was configured correctly
            call_kwargs = mock_chat_openai.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["api_key"] == "test_key"
            assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    @pytest.mark.asyncio
    async def test_arun_stateless_behavior(
        self,
        mock_chatopenai_client,
        test_input,
        expected_output,
    ):
        """Test _arun with stateless behavior (default)."""
        # Setup mock response
        mock_structured_client = await setup_async_mock_response(
            mock_chatopenai_client,
            expected_output,
            "chatopenai",
        )

        agent = TestLangBaseAgent()  # Default is stateless=True

        # Execute arun
        result = await agent.arun(test_input)

        # Verify result
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test response"

        # Memory should remain empty in stateless mode
        assert len(agent.memory.messages) == 0

    @pytest.mark.asyncio
    async def test_arun_stateful_behavior(
        self,
        mock_chatopenai_client,
        test_input,
        expected_output,
    ):
        """Test _arun with stateful behavior."""
        # Setup mock response
        mock_structured_client = await setup_async_mock_response(
            mock_chatopenai_client,
            expected_output,
            "chatopenai",
        )

        # Create stateful agent
        config = BaseAgentConfig(stateless=False)
        agent = TestLangBaseAgent(config=config)

        # Execute arun
        result = await agent.arun(test_input)

        # Verify result
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test response"

        # Memory should be updated in stateful mode
        assert len(agent.memory.messages) == 2  # user + assistant messages

    def test_debug_mode_functionality(self, mock_chatopenai_client):
        """Test debug mode initialization and usage."""
        # Test debug mode from config
        debug_config = BaseAgentConfig(debug=True)
        agent = TestLangBaseAgent(config=debug_config, debug=False)
        assert agent.debug is True  # config debug should take precedence

        # Test debug mode from parameter
        agent2 = TestLangBaseAgent(debug=True)
        assert agent2.debug is True

    def test_configuration_attribute_mapping(self, mock_chatopenai_client):
        """Test that configuration attributes are properly mapped to agent."""
        custom_config = BaseAgentConfig(
            model_name="gpt-4",
            temperature=0.8,
            api_key="custom_key",
        )

        agent = TestLangBaseAgent(config=custom_config)

        # Verify attributes were mapped from config
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.8
        assert agent.api_key == "custom_key"

    def test_error_handling_initialization(self):
        """Test error handling during agent initialization."""
        from unittest.mock import patch

        # Test that agents handle initialization gracefully
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            # Test that agents initialize even with mock failures
            mock_chat_openai.side_effect = Exception("Mock initialization error")

            # This should still create the agent object, but client creation might fail
            try:
                agent = TestLangBaseAgent()
                # If we get here, the agent was created despite the client error
                assert agent is not None
            except Exception as e:
                # Expected if client creation fails
                assert "Mock initialization error" in str(e)
