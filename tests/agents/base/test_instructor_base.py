# noqa: F841
"""Test cases for InstructorBaseAgent."""

from unittest.mock import MagicMock

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

    def test_initialization_default_config(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test InstructorBaseAgent initialization with default config."""
        agent = TestInstructorBaseAgent()

        # Verify agent properties
        assert agent.client == mock_instructor_client

    def test_initialization_custom_config(
        self,
        mock_openai_client,
        mock_instructor_client,
        custom_config,
    ):
        """Test InstructorBaseAgent initialization with custom config."""
        agent = TestInstructorBaseAgent(config=custom_config)

        # Verify configuration was applied
        assert agent.api_key == "test_key"
        assert str(agent.base_url) == "https://custom.api.com/v1"

    def test_instructor_compatible_model_creation(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test instructor compatible model creation."""
        agent = TestInstructorBaseAgent()

        # Test model creation
        instructor_model = agent._create_instructor_compatible_model(
            AgentTestOutputSchema,
        )

        # Verify model properties
        assert instructor_model.__name__ == AgentTestOutputSchema.__name__
        assert instructor_model.__doc__ == AgentTestOutputSchema.__doc__

        # Verify fields are preserved
        original_fields = AgentTestOutputSchema.model_fields.keys()
        instructor_fields = instructor_model.model_fields.keys()
        assert original_fields == instructor_fields

    @pytest.mark.asyncio
    async def test_get_response_async_mock(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test async response generation with mocked client."""
        # Setup mock response
        response_data = {
            "response": "test response",
            "metadata": {},
        }
        mock_chat_completions = await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()

        # Test response generation with run_context messages
        test_messages = [{"role": "user", "content": "test message"}]
        result = await agent.get_response_async(
            run_context=RunContext(messages=test_messages),
        )

        # Verify instructor call
        mock_chat_completions.assert_called_once()
        call_kwargs = mock_chat_completions.call_args.kwargs
        assert "messages" in call_kwargs
        assert "model" in call_kwargs
        assert "temperature" in call_kwargs
        assert "response_model" in call_kwargs

        # Verify response
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "test response"

    def test_default_system_message(self, mock_openai_client, mock_instructor_client):
        """Test default system message creation.

        Note: io_hints=True by default, so description (with IO hints) is included.
        """
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        assert isinstance(system_message, dict)
        assert system_message["role"] == "system"
        # System message includes base prompt + description (with IO hints)
        assert agent.system_prompt in system_message["content"]
        assert "AGENT DESCRIPTION:" in system_message["content"]

    @pytest.mark.asyncio
    async def test_arun_stateless_behavior(
        self,
        mock_openai_client,
        mock_instructor_client,
        test_input,
    ):
        """Test _arun with stateless behavior (default)."""
        # Setup mock response
        response_data = {
            "response": "Test instructor response",
            "metadata": {"test": True},
        }
        mock_chat_completions = await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()  # Default is stateless=True

        # Execute arun
        result = await agent.arun(test_input)

        # Verify result
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test instructor response"

        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3

    @pytest.mark.asyncio
    async def test_arun_stateful_behavior(
        self,
        mock_openai_client,
        mock_instructor_client,
        test_input,
    ):
        """Test _arun with stateful behavior."""
        # Setup mock response
        response_data = {
            "response": "Test instructor response",
            "metadata": {"test": True},
        }
        mock_chat_completions = await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        # Create stateful agent
        config = BaseAgentConfig(stateless=False)
        agent = TestInstructorBaseAgent(config=config)

        # Execute arun
        result = await agent.arun(test_input)

        # Verify result
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test instructor response"

        assert result.run_context is not None
        assert result.run_context.messages is not None
        assert len(result.run_context.messages) == 3

    def test_client_configuration(self, custom_config):
        """Test that OpenAI and instructor clients are configured correctly."""
        from unittest.mock import patch

        with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
            with patch("akd.agents._base.openai.AsyncOpenAI") as mock_openai:
                mock_instructor_client = MagicMock()
                mock_openai_client = MagicMock()
                mock_instructor.return_value = mock_instructor_client
                mock_openai.return_value = mock_openai_client

                agent = TestInstructorBaseAgent(config=custom_config)

                # Verify OpenAI client configuration
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "test_key"
                assert call_kwargs["base_url"] == "https://custom.api.com/v1"

                # Verify instructor wrapper
                mock_instructor.assert_called_once_with(mock_openai_client)

    def test_deepcopy_functionality(self, mock_openai_client, mock_instructor_client):
        """Test custom deepcopy implementation."""
        import copy

        agent = TestInstructorBaseAgent()

        # Test deepcopy
        copied_agent = copy.deepcopy(agent)

        # Verify copied agent has the same configuration
        assert copied_agent.api_key == agent.api_key
        assert copied_agent.model_name == agent.model_name
        assert copied_agent.temperature == agent.temperature

        # Verify client exists (deepcopy should work)
        assert copied_agent.client is not None
        assert agent.client is not None

    def test_response_model_creation_edge_cases(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test edge cases in instructor model creation."""
        agent = TestInstructorBaseAgent()

        # Test with complex schema
        complex_model = agent._create_instructor_compatible_model(AgentTestOutputSchema)

        # Should handle all field types properly
        assert hasattr(complex_model, "model_fields")
        assert "response" in complex_model.model_fields
        assert "metadata" in complex_model.model_fields

        # Test instantiation
        instance = complex_model(response="test", metadata={"key": "value"})
        assert instance.response == "test"
        assert instance.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_response_async_with_custom_model(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test get_response_async with custom response model."""
        # Setup mock response
        response_data = {
            "response": "custom model response",
            "metadata": {"custom": True},
        }
        mock_chat_completions = await setup_async_mock_response(
            mock_instructor_client,
            response_data,
            "instructor",
        )

        agent = TestInstructorBaseAgent()

        # Test with custom response model
        test_messages = [{"role": "user", "content": "test message"}]
        result = await agent.get_response_async(
            run_context=RunContext(messages=test_messages),
            response_model=AgentTestOutputSchema,
        )

        # Verify response
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "custom model response"
        assert result.metadata == {"custom": True}

    def test_io_hints_enabled_by_default(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that IO hints are enabled by default (io_hints=True in BaseConfig)."""
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        # Verify IO hints are present by default
        assert "INPUT FIELD DESCRIPTIONS:" in system_message["content"]
        assert "OUTPUT FIELD DESCRIPTIONS:" in system_message["content"]
        assert agent.system_prompt in system_message["content"]

    def test_io_hints_disabled(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test IO hints functionality when disabled."""
        config = BaseAgentConfig(io_hints=False)
        agent = TestInstructorBaseAgent(config=config)

        system_message = agent._default_system_message()

        # Verify IO hints are NOT present when disabled
        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]
        assert "OUTPUT FIELD DESCRIPTIONS:" not in system_message["content"]

        # Description is still added (just without IO field info)
        # The docstring from the agent class is still there
        assert agent.system_prompt in system_message["content"]

    def test_input_schema_info_property(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test _input_schema_info property."""
        agent = TestInstructorBaseAgent()

        schema_info = agent._input_schema_info

        # Verify schema info extraction
        assert "**query**:" in schema_info
        assert "**optional_param**:" in schema_info
        assert "Test query input" in schema_info
        assert "Optional parameter" in schema_info
        assert schema_info.startswith("- **query**:")

    def test_input_schema_info_no_schema(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test _input_schema_info property when no input schema."""
        agent = TestInstructorBaseAgent()

        # Temporarily remove input schema
        original_schema = agent.input_schema
        agent.input_schema = None

        schema_info = agent._input_schema_info

        # Verify empty string returned
        assert schema_info == ""

        # Restore original schema
        agent.input_schema = original_schema

    def test_agent_description_enabled_by_default(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that agent description is included by default (io_hints=True)."""
        agent = TestInstructorBaseAgent()

        system_message = agent._default_system_message()

        # Should include description by default
        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert agent.system_prompt in system_message["content"]

    def test_agent_description_when_cleared(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test behavior when description is cleared.

        Note: Since AbstractBase adds INPUT/OUTPUT FIELD DESCRIPTIONS to description,
        clearing the description also clears the field descriptions.
        """
        agent = TestInstructorBaseAgent()

        # Clear the description to simulate no description
        agent.description = ""

        system_message = agent._default_system_message()

        # When description is cleared, both agent description and field descriptions are gone
        assert "AGENT DESCRIPTION:" not in system_message["content"]
        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]

    def test_agent_description_with_custom_description(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test agent description when manually set.

        Note: When manually setting description, you override the INPUT/OUTPUT FIELD DESCRIPTIONS
        that were added by AbstractBase. To keep them, append to existing description instead.
        """
        agent = TestInstructorBaseAgent()

        # Set a test description (this replaces the auto-generated one with field descriptions)
        test_description = "This is a test instructor agent for testing purposes"
        agent.description = test_description

        system_message = agent._default_system_message()

        # Should include the custom agent description
        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert test_description in system_message["content"]

        # Note: INPUT FIELD DESCRIPTIONS are NOT present because we replaced the description
        # If we want both, we should append instead of replace
        assert "INPUT FIELD DESCRIPTIONS:" not in system_message["content"]

    def test_agent_description_from_config(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that agent description can be set via config.

        Note: AbstractBase appends INPUT/OUTPUT FIELD DESCRIPTIONS to the description
        during _post_init(), so the final description will be longer than the input.
        """
        test_description = "Instructor agent description from config"
        config = BaseAgentConfig(description=test_description)
        agent = TestInstructorBaseAgent(config=config)

        system_message = agent._default_system_message()

        # Should include the description from config plus field descriptions
        assert "AGENT DESCRIPTION:" in system_message["content"]
        assert test_description in system_message["content"]
        # Description now includes INPUT/OUTPUT FIELD DESCRIPTIONS appended by AbstractBase
        assert test_description in agent.description
        assert "INPUT FIELD DESCRIPTIONS:" in agent.description
        assert "OUTPUT FIELD DESCRIPTIONS:" in agent.description
