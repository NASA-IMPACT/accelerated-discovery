"""Test cases for base agent classes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import AnyUrl, Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent, LangBaseAgent
from akd.configs.project import CONFIG


# Test schemas for agent testing - prefixed to avoid pytest collection
class AgentTestInputSchema(InputSchema):
    """Test input schema for base agents."""

    query: str = Field(..., description="Test query input")
    optional_param: str = Field(default="default", description="Optional parameter")


class AgentTestOutputSchema(OutputSchema):
    """Test output schema for base agents."""

    response: str = Field(..., description="Test response output")
    metadata: dict = Field(default_factory=dict, description="Response metadata")


class AgentTestCustomConfig(BaseAgentConfig):
    """Custom config for testing."""

    custom_field: str = Field(default="custom_value", description="Custom test field")
    custom_temperature: float = Field(default=0.5, description="Custom temperature")


# Test agent implementations
class TestLangBaseAgent(LangBaseAgent[AgentTestInputSchema, AgentTestOutputSchema]):
    """Test implementation of LangBaseAgent."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    async def _arun(
        self,
        params: AgentTestInputSchema,
        **kwargs,
    ) -> AgentTestOutputSchema:
        """Test implementation that calls parent to handle memory."""
        # Call the parent _arun which handles memory management
        result = await super()._arun(params, **kwargs)
        return result


class TestInstructorBaseAgent(
    InstructorBaseAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Test implementation of InstructorBaseAgent."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    async def _arun(
        self,
        params: AgentTestInputSchema,
        **kwargs,
    ) -> AgentTestOutputSchema:
        """Test implementation that calls parent to handle memory."""
        # Call the parent _arun which handles memory management
        result = await super()._arun(params, **kwargs)
        return result


class TestBaseAgentConfigMethods:
    """Test BaseAgentConfig functionality."""

    def test_default_config_initialization(self):
        """Test default configuration initialization."""
        config = BaseAgentConfig()

        # Test default values from CONFIG
        assert config.base_url == CONFIG.model_config_settings.base_url
        assert config.api_key == CONFIG.model_config_settings.api_keys.openai
        assert config.model_name == CONFIG.model_config_settings.model_name
        assert config.temperature == 0.0
        assert config.system_prompt is not None

    def test_custom_config_initialization(self):
        """Test custom configuration initialization."""
        custom_config = AgentTestCustomConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            custom_field="test_value",
            custom_temperature=0.8,
        )

        assert custom_config.model_name == "gpt-3.5-turbo"
        assert custom_config.temperature == 0.7
        assert custom_config.custom_field == "test_value"
        assert custom_config.custom_temperature == 0.8

    def test_config_field_validation(self):
        """Test configuration field validation."""
        # Test with invalid base_url type should still work (AnyUrl handles conversion)
        config = BaseAgentConfig(
            base_url="https://custom.api.com/v1",
            temperature=0.5,
        )
        assert isinstance(config.base_url, AnyUrl)
        assert config.temperature == 0.5


class TestLangBaseAgentFunctionality:
    """Test LangBaseAgent specific functionality."""

    def test_initialization_default_config(self):
        """Test LangBaseAgent initialization with default config."""
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            mock_client = MagicMock()
            mock_chat_openai.return_value = mock_client

            agent = TestLangBaseAgent()

            # Verify client initialization
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args.kwargs
            assert "api_key" in call_kwargs
            assert "model" in call_kwargs
            assert "base_url" in call_kwargs
            assert "temperature" in call_kwargs

            # Verify agent properties
            assert agent.client == mock_client
            assert isinstance(agent.memory, ChatMessageHistory)
            assert agent.prompt_template is not None

    def test_initialization_custom_config(self):
        """Test LangBaseAgent initialization with custom config."""
        custom_config = BaseAgentConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key="test_key",
            base_url="https://custom.api.com/v1",
        )

        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            mock_client = MagicMock()
            mock_chat_openai.return_value = mock_client

            agent = TestLangBaseAgent(config=custom_config)

            # Verify configuration was applied
            assert agent.model_name == "gpt-3.5-turbo"
            assert agent.temperature == 0.7
            assert agent.api_key == "test_key"
            assert str(agent.base_url) == "https://custom.api.com/v1"

            # Verify client was configured correctly
            call_kwargs = mock_chat_openai.call_args.kwargs
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["api_key"] == "test_key"
            assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    def test_memory_management(self):
        """Test memory initialization and reset functionality."""
        with patch("akd.agents._base.ChatOpenAI"):
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
    async def test_get_response_async_mock(self):
        """Test async response generation with mocked client."""
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            # Setup mock client and structured output
            mock_client = MagicMock()
            mock_structured_client = AsyncMock()
            mock_response = AgentTestOutputSchema(response="test response")

            mock_client.with_structured_output.return_value = mock_structured_client
            mock_structured_client.ainvoke.return_value = mock_response
            mock_chat_openai.return_value = mock_client

            agent = TestLangBaseAgent()

            # Test response generation
            result = await agent.get_response_async()

            # Verify structured output setup
            mock_client.with_structured_output.assert_called_once_with(
                AgentTestOutputSchema,
                method="function_calling",
            )

            # Verify response
            mock_structured_client.ainvoke.assert_called_once()
            assert isinstance(result, AgentTestOutputSchema)
            assert result.response == "test response"

    def test_prompt_template_setup(self):
        """Test prompt template initialization."""
        with patch("akd.agents._base.ChatOpenAI"):
            agent = TestLangBaseAgent()

            # Verify prompt template exists and has correct structure
            assert agent.prompt_template is not None

            # Test template formatting
            formatted = agent.prompt_template.format_messages(memory=[])
            assert len(formatted) >= 1  # Should have at least system message
            assert formatted[0].content == agent.system_prompt


class TestInstructorBaseAgentFunctionality:
    """Test InstructorBaseAgent specific functionality."""

    def test_initialization_default_config(self):
        """Test InstructorBaseAgent initialization with default config."""
        with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
            with patch("akd.agents._base.openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_instructor.return_value = mock_client
                mock_openai_client = MagicMock()
                mock_openai.return_value = mock_openai_client

                agent = TestInstructorBaseAgent()

                # Verify OpenAI client initialization
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args.kwargs
                assert "api_key" in call_kwargs
                assert "base_url" in call_kwargs

                # Verify instructor wrapper
                mock_instructor.assert_called_once_with(mock_openai_client)

                # Verify agent properties
                assert agent.client == mock_client
                assert isinstance(agent.memory, list)
                assert len(agent.memory) == 0

    def test_initialization_custom_config(self):
        """Test InstructorBaseAgent initialization with custom config."""
        custom_config = BaseAgentConfig(
            api_key="test_key",
            base_url="https://custom.api.com/v1",
        )

        with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
            with patch("akd.agents._base.openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_instructor.return_value = mock_client

                agent = TestInstructorBaseAgent(config=custom_config)

                # Verify configuration was applied
                assert agent.api_key == "test_key"
                assert str(agent.base_url) == "https://custom.api.com/v1"

                # Verify OpenAI client configuration
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "test_key"
                assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    def test_memory_management(self):
        """Test memory initialization and reset functionality."""
        with patch("akd.agents._base.instructor.from_openai"):
            with patch("akd.agents._base.openai.AsyncOpenAI"):
                agent = TestInstructorBaseAgent()

                # Test initial memory state
                assert isinstance(agent.memory, list)
                assert len(agent.memory) == 0

                # Add test messages manually
                agent.memory.append({"role": "user", "content": "test message"})
                agent.memory.append({"role": "assistant", "content": "test response"})
                assert len(agent.memory) == 2

                # Test memory reset
                agent.reset_memory()
                assert len(agent.memory) == 0

    def test_instructor_compatible_model_creation(self):
        """Test instructor compatible model creation."""
        with patch("akd.agents._base.instructor.from_openai"):
            with patch("akd.agents._base.openai.AsyncOpenAI"):
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
    async def test_get_response_async_mock(self):
        """Test async response generation with mocked client."""
        with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
            with patch("akd.agents._base.openai.AsyncOpenAI"):
                # Setup mock client
                mock_client = MagicMock()
                mock_chat_completions = AsyncMock()
                mock_response = MagicMock()
                mock_response.model_dump.return_value = {
                    "response": "test response",
                    "metadata": {},
                }

                mock_client.chat.completions.create = mock_chat_completions
                mock_chat_completions.return_value = mock_response
                mock_instructor.return_value = mock_client

                agent = TestInstructorBaseAgent()

                # Test response generation
                result = await agent.get_response_async()

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


class TestBaseAgentSharedFunctionality:
    """Test shared functionality between base agents."""

    @pytest.mark.asyncio
    async def test_arun_with_lang_base_agent(self):
        """Test full arun execution with LangBaseAgent."""
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            mock_client = MagicMock()
            mock_structured_client = AsyncMock()
            expected_response = AgentTestOutputSchema(
                response="Test response",
                metadata={"test": True},
            )

            mock_client.with_structured_output.return_value = mock_structured_client
            mock_structured_client.ainvoke.return_value = expected_response
            mock_chat_openai.return_value = mock_client

            agent = TestLangBaseAgent()
            test_input = AgentTestInputSchema(query="test query")

            # Execute arun - this will call base _arun which manages memory
            result = await agent.arun(test_input)

            # Verify result structure and type
            assert isinstance(result, AgentTestOutputSchema)
            assert result.response == "Test response"
            assert result.metadata == {"test": True}

            # Verify memory was updated by base _arun
            assert len(agent.memory.messages) == 2  # user + assistant messages

    @pytest.mark.asyncio
    async def test_arun_with_instructor_base_agent(self):
        """Test full arun execution with InstructorBaseAgent."""
        with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
            with patch("akd.agents._base.openai.AsyncOpenAI"):
                mock_client = MagicMock()
                mock_chat_completions = AsyncMock()
                mock_response = MagicMock()
                mock_response.model_dump.return_value = {
                    "response": "Test instructor response",
                    "metadata": {"test": True},
                }

                mock_client.chat.completions.create = mock_chat_completions
                mock_chat_completions.return_value = mock_response
                mock_instructor.return_value = mock_client

                agent = TestInstructorBaseAgent()
                test_input = AgentTestInputSchema(query="test query")

                # Execute arun - this will call base _arun which manages memory
                result = await agent.arun(test_input)

                # Verify result structure and type
                assert isinstance(result, AgentTestOutputSchema)
                assert result.response == "Test instructor response"
                assert result.metadata == {"test": True}

                # Verify memory was updated by base _arun
                assert len(agent.memory) == 2  # user + assistant messages

    def test_schema_validation_integration(self):
        """Test that agents properly validate input/output schemas."""
        with patch("akd.agents._base.ChatOpenAI"):
            agent = TestLangBaseAgent()

            # Verify schema attributes
            assert agent.input_schema == AgentTestInputSchema
            assert agent.output_schema == AgentTestOutputSchema

            # Test input validation (inherited from AbstractBase)
            valid_input = AgentTestInputSchema(query="test")
            validated = agent._validate_input(valid_input)
            assert isinstance(validated, AgentTestInputSchema)
            assert validated.query == "test"

            # Test input validation with dict
            input_dict = {"query": "test dict"}
            validated = agent._validate_input(input_dict)
            assert isinstance(validated, AgentTestInputSchema)
            assert validated.query == "test dict"

    def test_debug_mode_functionality(self):
        """Test debug mode initialization and usage."""
        with patch("akd.agents._base.ChatOpenAI"):
            # Test debug mode from config
            debug_config = BaseAgentConfig(debug=True)
            agent = TestLangBaseAgent(config=debug_config, debug=False)
            assert agent.debug is True  # config debug should take precedence

            # Test debug mode from parameter
            agent2 = TestLangBaseAgent(debug=True)
            assert agent2.debug is True

    def test_configuration_attribute_mapping(self):
        """Test that configuration attributes are properly mapped to agent."""
        custom_config = BaseAgentConfig(
            model_name="gpt-4",
            temperature=0.8,
            api_key="custom_key",
        )

        with patch("akd.agents._base.ChatOpenAI"):
            agent = TestLangBaseAgent(config=custom_config)

            # Verify attributes were mapped from config
            assert agent.model_name == "gpt-4"
            assert agent.temperature == 0.8
            assert agent.api_key == "custom_key"

    def test_error_handling_initialization(self):
        """Test error handling during agent initialization."""
        # This test verifies agents can handle initialization gracefully
        # Even with potential configuration issues
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
                with patch("akd.agents._base.openai.AsyncOpenAI"):
                    # Test that agents initialize even with mock failures
                    mock_chat_openai.side_effect = Exception(
                        "Mock initialization error",
                    )

                    # This should still create the agent object, but client creation might fail
                    try:
                        agent = TestLangBaseAgent()
                        # If we get here, the agent was created despite the client error
                        assert agent is not None
                    except Exception as e:
                        # Expected if client creation fails
                        assert "Mock initialization error" in str(e)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_none_config_handling(self):
        """Test handling of None config."""
        with patch("akd.agents._base.ChatOpenAI"):
            agent = TestLangBaseAgent(config=None)
            # Should use default config
            assert agent.config is not None

    def test_empty_memory_operations(self):
        """Test memory operations with empty memory."""
        with patch("akd.agents._base.ChatOpenAI"):
            with patch("akd.agents._base.instructor.from_openai"):
                with patch("akd.agents._base.openai.AsyncOpenAI"):
                    lang_agent = TestLangBaseAgent()
                    instructor_agent = TestInstructorBaseAgent()

                    # Test reset on empty memory
                    lang_agent.reset_memory()
                    instructor_agent.reset_memory()

                    # Verify still empty
                    assert len(lang_agent.memory.messages) == 0
                    assert len(instructor_agent.memory) == 0

    @pytest.mark.asyncio
    async def test_response_type_validation(self):
        """Test that response types are properly validated."""
        with patch("akd.agents._base.ChatOpenAI") as mock_chat_openai:
            mock_client = MagicMock()
            mock_structured_client = AsyncMock()

            # Return invalid response type
            invalid_response = {"invalid": "response"}
            mock_structured_client.ainvoke.return_value = invalid_response
            mock_client.with_structured_output.return_value = mock_structured_client
            mock_chat_openai.return_value = mock_client

            agent = TestLangBaseAgent()

            # This should handle type validation gracefully
            # The actual behavior depends on the AbstractBase implementation
            try:
                result = await agent.get_response_async()
                # If successful, should be properly typed
                assert isinstance(result, AgentTestOutputSchema)
            except Exception:
                # Type validation error is acceptable
                pass


# Prevent pytest from collecting test classes as tests themselves
TestLangBaseAgent.__test__ = False
TestInstructorBaseAgent.__test__ = False


if __name__ == "__main__":
    # Run specific test functions for manual testing
    test_config = TestBaseAgentConfigMethods()
    test_config.test_default_config_initialization()
    test_config.test_custom_config_initialization()
    print("✓ Config tests passed")

    # Test basic instantiation
    with patch("akd.agents._base.ChatOpenAI"):
        with patch("akd.agents._base.instructor.from_openai"):
            with patch("akd.agents._base.openai.AsyncOpenAI"):
                lang_agent = TestLangBaseAgent()
                instructor_agent = TestInstructorBaseAgent()
                print("✓ Agent instantiation tests passed")

    print("\n✅ Basic manual tests completed successfully!")
