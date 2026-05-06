# noqa: F841
"""Test cases for InstructorBaseAgent (now an alias for LiteLLMInstructorBaseAgent)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd._base import RunContext
from akd.agents._base import BaseAgentConfig

from .conftest import (
    AgentTestOutputSchema,
    TestInstructorBaseAgent,
    setup_async_mock_response,
)


class TestInstructorBaseAgentFunctionality:
    """Test InstructorBaseAgent specific functionality."""

    def test_initialization_default_config(self, mock_instructor_client):
        """Test InstructorBaseAgent initialization with default config."""
        agent = TestInstructorBaseAgent()
        assert agent.client == mock_instructor_client

    def test_initialization_custom_config(self, mock_instructor_client, custom_config):
        """Test InstructorBaseAgent initialization with custom config."""
        agent = TestInstructorBaseAgent(config=custom_config)
        assert agent.api_key == "test_key"
        assert str(agent.base_url) == "https://custom.api.com/v1"

    @pytest.mark.asyncio
    async def test_get_response_async_mock(self, mock_instructor_client):
        """Test async response generation with mocked client."""
        response_data = {
            "response": "test response",
            "metadata": {},
        }
        mock_create = await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()

        test_messages = [{"role": "user", "content": "test message"}]
        result = await agent.get_response_async(
            run_context=RunContext(messages=test_messages),
        )

        mock_create.assert_called_once()
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "test response"

    def test_default_system_message(self, mock_instructor_client):
        """Test default system message creation."""
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        assert isinstance(system_message, dict)
        assert system_message["role"] == "system"
        assert agent.system_prompt in system_message["content"]
        assert "AGENT DESCRIPTION:" in system_message["content"]

    @pytest.mark.asyncio
    async def test_arun_stateless_behavior(self, mock_instructor_client, test_input):
        """Test _arun with stateless behavior (default)."""
        response_data = {
            "response": "Test instructor response",
            "metadata": {"test": True},
        }
        await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()

        result = await agent.arun(test_input)

        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test instructor response"
        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3

    @pytest.mark.asyncio
    async def test_arun_stateful_behavior(self, mock_instructor_client, test_input):
        """Test _arun with stateful behavior."""
        response_data = {
            "response": "Test instructor response",
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
        assert result.response == "Test instructor response"
        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3

    def test_client_configuration(self, custom_config):
        """Test that LiteLLM instructor client is configured correctly."""
        with patch("akd.agents._base._base.instructor.from_litellm") as mock_from_litellm:
            mock_client = MagicMock()
            mock_from_litellm.return_value = mock_client

            agent = TestInstructorBaseAgent(config=custom_config)

            mock_from_litellm.assert_called_once()
            assert agent.client == mock_client

    def test_deepcopy_functionality(self, mock_instructor_client):
        """Test custom deepcopy implementation."""
        import copy

        agent = TestInstructorBaseAgent()

        copied_agent = copy.deepcopy(agent)

        assert copied_agent.api_key == agent.api_key
        assert copied_agent.model_name == agent.model_name
        assert copied_agent.temperature == agent.temperature
        assert copied_agent.client is not None
        assert agent.client is not None

    @pytest.mark.asyncio
    async def test_get_response_async_with_custom_model(self, mock_instructor_client):
        """Test get_response_async with custom response model."""
        response_data = {
            "response": "custom model response",
            "metadata": {"custom": True},
        }
        await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()

        test_messages = [{"role": "user", "content": "test message"}]
        result = await agent.get_response_async(
            run_context=RunContext(messages=test_messages),
            response_model=AgentTestOutputSchema,
        )

        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "custom model response"
        assert result.metadata == {"custom": True}

    @pytest.mark.asyncio
    async def test_instructor_usage_emits_normalized_token_fields(self, mock_instructor_client, monkeypatch):
        """Token usage fields should be normalized on both span and usage event."""
        monkeypatch.setenv("LOGFIRE_INSTRUMENT_HTTPX", "false")

        class FakeSpan:
            def __init__(self):
                self.attrs = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set_attributes(self, attrs):
                self.attrs.update(attrs)

        fake_span = FakeSpan()
        with patch("akd.agents._base._base.logfire.span", return_value=fake_span):
            with patch("akd.agents._base._base.logfire.info") as mock_logfire_info:
                mock_response = MagicMock()
                mock_response.model_dump.return_value = {
                    "response": "usage response",
                    "metadata": {},
                }
                mock_completion = MagicMock()
                mock_completion.usage = SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                )
                mock_instructor_client.chat.completions.create_with_completion = AsyncMock(
                    return_value=(mock_response, mock_completion),
                )

                agent = TestInstructorBaseAgent()
                test_messages = [{"role": "user", "content": "test message"}]
                await agent.get_response_async(run_context=RunContext(messages=test_messages))

                assert fake_span.attrs.get("event_name") == "akd.llm.instructor_call"
                assert fake_span.attrs.get("input_tokens") == 11
                assert fake_span.attrs.get("output_tokens") == 7
                assert fake_span.attrs.get("total_tokens") == 18
                assert fake_span.attrs.get("prompt_tokens") == 11
                assert fake_span.attrs.get("completion_tokens") == 7
                assert fake_span.attrs.get("usage_missing") is False

                usage_event = next(
                    call.kwargs["attributes"]
                    for call in mock_logfire_info.call_args_list
                    if call.args and call.args[0] == "akd.llm.instructor_usage"
                )
                assert usage_event["event_name"] == "akd.llm.instructor_usage"
                assert usage_event["input_tokens"] == 11
                assert usage_event["output_tokens"] == 7
                assert usage_event["total_tokens"] == 18
                assert usage_event["prompt_tokens"] == 11
                assert usage_event["completion_tokens"] == 7
                assert usage_event["usage_missing"] is False

    @pytest.mark.asyncio
    async def test_instructor_usage_missing_is_logged_safely(self, mock_instructor_client, monkeypatch):
        """Missing usage should not crash and should set usage_missing=true."""
        monkeypatch.setenv("LOGFIRE_INSTRUMENT_HTTPX", "false")

        class FakeSpan:
            def __init__(self):
                self.attrs = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set_attributes(self, attrs):
                self.attrs.update(attrs)

        fake_span = FakeSpan()
        with patch("akd.agents._base._base.logfire.span", return_value=fake_span):
            with patch("akd.agents._base._base.logfire.info") as mock_logfire_info:
                mock_response = MagicMock()
                mock_response.model_dump.return_value = {
                    "response": "missing usage response",
                    "metadata": {},
                }
                mock_completion = MagicMock()
                mock_completion.usage = None
                mock_instructor_client.chat.completions.create_with_completion = AsyncMock(
                    return_value=(mock_response, mock_completion),
                )

                agent = TestInstructorBaseAgent()
                test_messages = [{"role": "user", "content": "test message"}]
                await agent.get_response_async(run_context=RunContext(messages=test_messages))

                assert fake_span.attrs.get("usage_missing") is True
                assert fake_span.attrs.get("input_tokens") == 0
                assert fake_span.attrs.get("output_tokens") == 0
                assert fake_span.attrs.get("total_tokens") == 0

                usage_event = next(
                    call.kwargs["attributes"]
                    for call in mock_logfire_info.call_args_list
                    if call.args and call.args[0] == "akd.llm.instructor_usage"
                )
                assert usage_event["usage_missing"] is True
                assert usage_event["input_tokens"] == 0
                assert usage_event["output_tokens"] == 0
                assert usage_event["total_tokens"] == 0

    def test_io_hints_enabled_by_default(self, mock_instructor_client):
        """Test that IO hints are enabled by default (io_hints=True in BaseConfig)."""
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        assert "INPUT FIELD DESCRIPTIONS:" in system_message["content"]
        assert "OUTPUT FIELD DESCRIPTIONS:" in system_message["content"]
        assert agent.system_prompt in system_message["content"]

    def test_io_hints_disabled(self, mock_instructor_client):
        """Test IO hints functionality when disabled."""
        config = BaseAgentConfig(io_hints=False)
        agent = TestInstructorBaseAgent(config=config)

        system_message = agent._default_system_message()

        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]
        assert "OUTPUT FIELD DESCRIPTIONS:" not in system_message["content"]
        assert agent.system_prompt in system_message["content"]

    def test_agent_description_enabled_by_default(self, mock_instructor_client):
        """Test that agent description is included by default (io_hints=True)."""
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert agent.system_prompt in system_message["content"]

    def test_agent_description_when_cleared(self, mock_instructor_client):
        """Test behavior when description is cleared."""
        agent = TestInstructorBaseAgent()

        agent.description = ""

        system_message = agent._default_system_message()

        assert "AGENT DESCRIPTION:" not in system_message["content"]
        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]

    def test_agent_description_with_custom_description(self, mock_instructor_client):
        """Test agent description when manually set."""
        agent = TestInstructorBaseAgent()

        test_description = "This is a test instructor agent for testing purposes"
        agent.description = test_description

        system_message = agent._default_system_message()

        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert test_description in system_message["content"]
        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]

    def test_agent_description_from_config(self, mock_instructor_client):
        """Test that agent description can be set via config."""
        test_description = "Instructor agent description from config"
        config = BaseAgentConfig(description=test_description)
        agent = TestInstructorBaseAgent(config=config)

        system_message = agent._default_system_message()

        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert test_description in system_message["content"]
        assert test_description in agent.description
        assert "INPUT FIELD DESCRIPTIONS:" in agent.description
        assert "OUTPUT FIELD DESCRIPTIONS:" in agent.description
