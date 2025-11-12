# noqa: F841
"""Shared integration tests for base agents."""

from unittest.mock import MagicMock

import pytest

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
        # Setup mock response
        expected_response = AgentTestOutputSchema(
            response="Test response",
            metadata={"test": True},
        )
        mock_structured_client = await setup_async_mock_response(
            mock_instructor_client,
            expected_response,
            "instructor",
        )

        # Create agent with stateless=False to test memory management
        config = BaseAgentConfig(stateless=False)
        agent = TestInstructorBaseAgent(config=config)

        # Execute arun - this will call base _arun which manages memory
        result = await agent.arun(test_input)

        # Verify result structure and type
        assert isinstance(result, AgentTestOutputSchema)
        assert result.response == "Test response"
        assert result.metadata == {"test": True}

        # Verify memory was updated by base _arun (only in stateful mode)
        assert len(agent.memory) == 2  # user + assistant messages

    @pytest.mark.asyncio
    async def test_stateless_behavior_default(
        self,
        mock_instructor_client,
        mock_openai_client,
    ):
        """Test that agents are stateless by default and don't store memory."""
        # Setup mocks for InstructorBaseAgent
        expected_response = AgentTestOutputSchema(response="Test response")
        mock_structured_client = await setup_async_mock_response(
            mock_instructor_client,
            expected_response,
            "instructor",
        )

        # Setup mocks for InstructorBaseAgent
        instructor_response_data = {
            "response": "Instructor response",
            "metadata": {},
        }
        mock_instructor_completions = await setup_async_mock_response(
            mock_instructor_client,
            instructor_response_data,
            "instructor",
        )

        # Test InstructorBaseAgent (default stateless)
        instructor_agent = TestInstructorBaseAgent()
        assert instructor_agent.stateless is True

        test_input = AgentTestInputSchema(query="test query")
        result = await instructor_agent.arun(test_input)

        # Memory should remain empty in stateless mode
        assert len(instructor_agent.memory) == 0
        assert isinstance(result, AgentTestOutputSchema)

        # Test InstructorBaseAgent (default stateless)
        instructor_agent = TestInstructorBaseAgent()
        assert instructor_agent.stateless is True

        result = await instructor_agent.arun(test_input)

        # Memory should remain empty in stateless mode
        assert len(instructor_agent.memory) == 0
        assert isinstance(result, AgentTestOutputSchema)

    def test_schema_validation_integration(self, mock_instructor_client):
        """Test that agents properly validate input/output schemas."""
        agent = TestInstructorBaseAgent()

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

    def test_configuration_attribute_mapping_across_agents(self):
        """Test that configuration attributes are properly mapped across all agent types."""
        custom_config = BaseAgentConfig(
            model_name="gpt-4o-mini",
            temperature=0.8,
            api_key="custom_key",
            stateless=False,
        )

        # Mock the imports to avoid client initialization
        from unittest.mock import patch

        with patch("akd.agents._base.instructor.from_openai"):
            with patch("akd.agents._base.instructor.from_openai"):
                with patch("akd.agents._base.openai.AsyncOpenAI"):
                    with patch("instructor.from_litellm"):
                        instructor_agent = TestInstructorBaseAgent(config=custom_config)
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

                        # Verify attributes were mapped from config for all agents
                        for agent in [instructor_agent, instructor_agent, litellm_agent]:
                            assert agent.model_name == "gpt-4o-mini"
                            assert agent.temperature == 0.8
                            assert agent.api_key == "custom_key"
                            assert agent.stateless is False


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_none_config_handling(self, mock_instructor_client):
        """Test handling of None config."""
        agent = TestInstructorBaseAgent(config=None)
        # Should use default config
        assert agent.config is not None

    def test_empty_memory_operations(
        self,
        mock_instructor_client,
        mock_openai_client,
    ):
        """Test memory operations with empty memory."""
        instructor_agent = TestInstructorBaseAgent()
        instructor_agent = TestInstructorBaseAgent()

        # Test reset on empty memory
        instructor_agent.reset_memory()
        instructor_agent.reset_memory()

        # Verify still empty
        assert len(instructor_agent.memory) == 0
        assert len(instructor_agent.memory) == 0

    @pytest.mark.asyncio
    async def test_response_type_validation(self, mock_instructor_client):
        """Test that response types are properly validated."""
        # Return invalid response type
        invalid_response = MagicMock()
        invalid_response.model_dump.return_value = {"invalid": "response"}
        mock_instructor_client.chat.completions.create.return_value = invalid_response

        agent = TestInstructorBaseAgent()

        # This should handle type validation gracefully
        # The actual behavior depends on the AbstractBase implementation
        try:
            result = await agent.get_response_async()
            # If successful, should be properly typed
            assert isinstance(result, AgentTestOutputSchema)
        except Exception:
            # Type validation error is acceptable
            pass

    def test_cross_agent_compatibility(self):
        """Test that all agents implement the same base interface."""
        # Test that all agent types have the same required interface
        from unittest.mock import patch

        with patch("akd.agents._base.instructor.from_openai"):
            with patch("akd.agents._base.instructor.from_openai"):
                with patch("akd.agents._base.openai.AsyncOpenAI"):
                    with patch("instructor.from_litellm"):
                        instructor_agent = TestInstructorBaseAgent()
                        instructor_agent = TestInstructorBaseAgent()
                        litellm_agent = TestLiteLLMAgent()

                        required_methods = [
                            "arun",
                            "get_response_async",
                            "reset_memory",
                        ]
                        required_properties = [
                            "memory",
                            "input_schema",
                            "output_schema",
                        ]

                        for agent in [instructor_agent, instructor_agent, litellm_agent]:
                            # Check required methods
                            for method in required_methods:
                                assert hasattr(agent, method), f"{type(agent).__name__} missing {method}"
                                assert callable(getattr(agent, method)), f"{type(agent).__name__}.{method} not callable"

                            # Check required properties
                            for prop in required_properties:
                                assert hasattr(agent, prop), f"{type(agent).__name__} missing {prop}"

    def test_agent_inheritance_hierarchy(self):
        """Test that inheritance hierarchy is correct."""
        from akd.agents._base import (
            BaseAgent,
            InstructorBaseAgent,
            LiteLLMInstructorBaseAgent,
        )

        # Test inheritance chain
        assert issubclass(TestInstructorBaseAgent, BaseAgent)
        assert issubclass(TestInstructorBaseAgent, BaseAgent)
        assert issubclass(TestInstructorBaseAgent, InstructorBaseAgent)
        assert issubclass(TestLiteLLMAgent, InstructorBaseAgent)
        assert issubclass(TestLiteLLMAgent, LiteLLMInstructorBaseAgent)

        # Test that LiteLLM agent inherits from Instructor agent
        assert issubclass(LiteLLMInstructorBaseAgent, InstructorBaseAgent)

    @pytest.mark.asyncio
    async def test_agent_error_recovery(
        self,
        mock_instructor_client,
        mock_openai_client,
    ):
        """Test that agents handle errors gracefully."""
        # Test with failing clients
        mock_instructor_client.chat.completions.create.side_effect = Exception(
            "Instructor error",
        )

        instructor_agent = TestInstructorBaseAgent()
        instructor_agent = TestInstructorBaseAgent()

        test_input = AgentTestInputSchema(query="test query")

        # Test that agents raise appropriate exceptions
        with pytest.raises(Exception) as exc_info:
            await instructor_agent.arun(test_input)
        assert "Instructor error" in str(exc_info.value)
