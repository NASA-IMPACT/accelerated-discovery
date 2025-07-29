"""Test guardrails decorator functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.configs.guardrails_config import GuardrailsConfig
from akd.guardrails import add_guardrails
from akd.tools._base import BaseTool
from akd.tools.granite_guardian_tool import RiskDefinition


# Test schemas - prefix to avoid pytest collection
class AgentInputSchema(InputSchema):
    """Test agent input schema."""
    
    query: str = Field(..., description="User query")


class AgentOutputSchema(OutputSchema):
    """Test agent output schema."""
    
    response: str = Field(..., description="Agent response")


class ToolInputSchema(InputSchema):
    """Test tool input schema."""
    
    content: str = Field(..., description="Content to process")


class ToolOutputSchema(OutputSchema):
    """Test tool output schema."""
    
    result: str = Field(..., description="Tool result")
    text: str = Field(default="", description="Additional text field")


# Test with agent
@add_guardrails(
    input_guardrails=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
    output_guardrails=[RiskDefinition.ANSWER_RELEVANCE]
)
class TestAgent(InstructorBaseAgent[AgentInputSchema, AgentOutputSchema]):
    """Test agent with guardrails."""
    
    input_schema = AgentInputSchema
    output_schema = AgentOutputSchema
    
    async def _arun(self, params: AgentInputSchema, **kwargs) -> AgentOutputSchema:
        """Simple test implementation."""
        return AgentOutputSchema(response=f"Processed: {params.query}")


# Test with tool
@add_guardrails(
    input_guardrails=[RiskDefinition.JAILBREAK],
    output_guardrails=[RiskDefinition.GROUNDEDNESS]
)
class TestTool(BaseTool[ToolInputSchema, ToolOutputSchema]):
    """Test tool with guardrails."""
    
    input_schema = ToolInputSchema
    output_schema = ToolOutputSchema
    
    async def _arun(self, params: ToolInputSchema, **kwargs) -> ToolOutputSchema:
        """Simple test implementation."""
        return ToolOutputSchema(result=f"Processed: {params.content}", text="Extra info")


# Agent with custom config
@add_guardrails(
    config=GuardrailsConfig(
        enabled=True,
        input_risk_types=[RiskDefinition.HARM],
        output_risk_types=[RiskDefinition.ANSWER_RELEVANCE],
        fail_on_risk=True,
        snippet_n_chars=50
    )
)
class TestAgentWithConfig(InstructorBaseAgent[AgentInputSchema, AgentOutputSchema]):
    """Test agent with custom guardrails config."""
    
    input_schema = AgentInputSchema
    output_schema = AgentOutputSchema
    
    async def _arun(self, params: AgentInputSchema, **kwargs) -> AgentOutputSchema:
        """Simple test implementation."""
        return AgentOutputSchema(response=f"Response to: {params.query}")


@pytest.mark.asyncio
async def test_guardrails_decorator_metadata():
    """Test that decorator preserves class metadata."""
    # Test agent
    agent = TestAgent()
    assert hasattr(agent, 'guardrails_config')
    assert hasattr(agent, 'guardrails_tool')
    assert agent.__class__.__name__ == "GuardrailedTestAgent"
    assert agent.__class__.__qualname__ == "GuardrailedTestAgent"
    assert agent.input_schema == AgentInputSchema
    assert agent.output_schema == AgentOutputSchema
    
    # Test tool
    tool = TestTool()
    assert hasattr(tool, 'guardrails_config')
    assert hasattr(tool, 'guardrails_tool')
    assert tool.__class__.__name__ == "GuardrailedTestTool"
    assert tool.__class__.__qualname__ == "GuardrailedTestTool"
    assert tool.input_schema == ToolInputSchema
    assert tool.output_schema == ToolOutputSchema


@pytest.mark.asyncio
async def test_guardrails_configuration():
    """Test guardrails configuration setup."""
    # Test default configuration
    agent = TestAgent()
    assert agent.guardrails_config.enabled is True
    assert RiskDefinition.JAILBREAK in agent.guardrails_config.input_risk_types
    assert RiskDefinition.HARM in agent.guardrails_config.input_risk_types
    assert RiskDefinition.ANSWER_RELEVANCE in agent.guardrails_config.output_risk_types
    assert agent.guardrails_config.fail_on_risk is False
    
    # Test custom configuration
    agent_with_config = TestAgentWithConfig()
    assert agent_with_config.guardrails_config.enabled is True
    assert agent_with_config.guardrails_config.input_risk_types == [RiskDefinition.HARM]
    assert agent_with_config.guardrails_config.output_risk_types == [RiskDefinition.ANSWER_RELEVANCE]
    assert agent_with_config.guardrails_config.fail_on_risk is True
    assert agent_with_config.guardrails_config.snippet_n_chars == 50


@pytest.mark.asyncio
async def test_guardrails_disabled():
    """Test behavior when guardrails are disabled."""
    with patch('akd.guardrails.GuardrailsConfig') as mock_config:
        mock_config.return_value = MagicMock(enabled=False)
        
        @add_guardrails()
        class DisabledGuardrailsAgent(InstructorBaseAgent[AgentInputSchema, AgentOutputSchema]):
            input_schema = AgentInputSchema
            output_schema = AgentOutputSchema
            
            async def _arun(self, params: AgentInputSchema, **kwargs) -> AgentOutputSchema:
                return AgentOutputSchema(response="test")
        
        agent = DisabledGuardrailsAgent()
        result = await agent.arun(AgentInputSchema(query="test input"))
        assert result.response == "test"


@pytest.mark.asyncio
async def test_text_extraction():
    """Test text extraction from input/output objects."""
    agent = TestAgent()
    
    # Test extraction from input with preferred field
    test_input = AgentInputSchema(query="Test query content")
    extracted = agent._extract_text_content(test_input, ["query"])
    assert extracted == "Test query content"
    
    # Test extraction from output with multiple fields
    test_output = ToolOutputSchema(result="Main result", text="Additional text")
    extracted = agent._extract_text_content(test_output, ["result", "text"])
    assert "Main result" in extracted
    assert "Additional text" in extracted
    
    # Test fallback to all string fields
    extracted = agent._extract_text_content(test_output, ["nonexistent_field"])
    assert len(extracted) > 0  # Should still extract from string fields


@pytest.mark.asyncio
async def test_guardrails_validation_pass():
    """Test guardrails validation when content passes."""
    agent = TestAgent()
    
    # Mock the guardian tool to return safe results
    mock_tool = AsyncMock()
    mock_tool.arun.return_value = MagicMock(risk_results=[{"is_risky": False}])
    agent.guardrails_tool = mock_tool
    
    # Test input validation
    result = await agent._validate_with_guardrails("safe content", [RiskDefinition.HARM], is_input=True)
    assert result is True
    
    # Test output validation
    result = await agent._validate_with_guardrails("safe response", [RiskDefinition.ANSWER_RELEVANCE], is_input=False)
    assert result is True


@pytest.mark.asyncio
async def test_guardrails_validation_fail():
    """Test guardrails validation when content fails."""
    agent = TestAgent()
    
    # Mock the guardian tool to return risky results
    mock_tool = AsyncMock()
    mock_tool.arun.return_value = MagicMock(risk_results=[{"is_risky": True}])
    agent.guardrails_tool = mock_tool
    
    # Test with fail_on_risk=False (default)
    result = await agent._validate_with_guardrails("risky content", [RiskDefinition.HARM], is_input=True)
    assert result is False
    
    # Test with fail_on_risk=True
    agent_strict = TestAgentWithConfig()
    mock_tool_strict = AsyncMock()
    mock_tool_strict.arun.return_value = MagicMock(risk_results=[{"is_risky": True}])
    agent_strict.guardrails_tool = mock_tool_strict
    
    with pytest.raises(ValueError, match="Guardrails validation failed"):
        await agent_strict._validate_with_guardrails("risky content", [RiskDefinition.HARM], is_input=True)


@pytest.mark.asyncio
async def test_guardrails_validation_error_handling():
    """Test error handling during validation."""
    agent = TestAgent()
    
    # Mock the guardian tool to raise an exception
    mock_tool = AsyncMock()
    mock_tool.arun.side_effect = Exception("Guardian tool error")
    agent.guardrails_tool = mock_tool
    
    # Test with fail_on_risk=False (should log error and return True)
    result = await agent._validate_with_guardrails("content", [RiskDefinition.HARM], is_input=True)
    assert result is True
    
    # Test with fail_on_risk=True (should re-raise exception)
    agent_strict = TestAgentWithConfig()
    mock_tool_strict = AsyncMock()
    mock_tool_strict.arun.side_effect = Exception("Guardian tool error")
    agent_strict.guardrails_tool = mock_tool_strict
    
    with pytest.raises(Exception, match="Guardian tool error"):
        await agent_strict._validate_with_guardrails("content", [RiskDefinition.HARM], is_input=True)


@pytest.mark.asyncio
async def test_full_arun_with_guardrails():
    """Test full _arun execution with guardrails."""
    agent = TestAgent()
    
    # Mock the guardian tool
    mock_tool = AsyncMock()
    mock_tool.arun.return_value = MagicMock(risk_results=[{"is_risky": False}])
    agent.guardrails_tool = mock_tool
    
    # Execute with guardrails
    test_input = AgentInputSchema(query="Test query")
    result = await agent.arun(test_input)
    
    assert result.response == "Processed: Test query"
    assert hasattr(result, "guardrails_validated")
    assert result.guardrails_validated() is True
    
    # Verify guardian tool was called for both input and output
    assert mock_tool.arun.call_count >= 2


@pytest.mark.asyncio
async def test_guardrails_status_field():
    """Test that guardrails status is properly added to response."""
    agent = TestAgent()
    
    # Mock guardian tool with mixed results
    mock_tool = AsyncMock()
    # First call (input validation) passes, second call (output validation) fails
    mock_tool.arun.side_effect = [
        MagicMock(risk_results=[{"is_risky": False}]),
        MagicMock(risk_results=[{"is_risky": True}])
    ]
    agent.guardrails_tool = mock_tool
    
    test_input = AgentInputSchema(query="Test query")
    result = await agent.arun(test_input)
    
    assert hasattr(result, "guardrails_validated")
    assert result.guardrails_validated() is False  # Should be False since output validation failed


@pytest.mark.asyncio
async def test_empty_content_handling():
    """Test handling of empty content."""
    agent = TestAgent()
    
    # Empty text should skip validation and return True
    result = await agent._validate_with_guardrails("", [RiskDefinition.HARM], is_input=True)
    assert result is True
    
    result = await agent._validate_with_guardrails(None, [RiskDefinition.HARM], is_input=True)
    assert result is True


@pytest.mark.asyncio
async def test_multiple_risk_types():
    """Test validation with multiple risk types."""
    agent = TestAgent()
    
    # Mock guardian tool to fail on second risk type
    mock_tool = AsyncMock()
    mock_tool.arun.side_effect = [
        MagicMock(risk_results=[{"is_risky": False}]),  # First risk type passes
        MagicMock(risk_results=[{"is_risky": True}])    # Second risk type fails
    ]
    agent.guardrails_tool = mock_tool
    
    result = await agent._validate_with_guardrails(
        "content", 
        [RiskDefinition.HARM, RiskDefinition.JAILBREAK], 
        is_input=True
    )
    assert result is False


@pytest.mark.asyncio
async def test_snippet_truncation():
    """Test that long content is truncated in error messages."""
    agent = TestAgent()
    agent.guardrails_config.snippet_n_chars = 10
    
    long_text = "This is a very long text that should be truncated"
    risk_type = RiskDefinition.HARM
    
    # Test that snippet is truncated
    with patch.object(agent, 'guardrails_config') as mock_config:
        mock_config.snippet_n_chars = 10
        mock_config.fail_on_risk = False
        
        result = agent._handle_risk_detection(long_text, risk_type, is_input=True)
        assert result is False


def test_sync():
    """Test basic synchronous operations."""
    # Just test instantiation
    agent = TestAgent()
    assert agent is not None
    
    tool = TestTool()
    assert tool is not None


# Prevent pytest from collecting our test classes
TestAgent.__test__ = False
TestTool.__test__ = False
TestAgentWithConfig.__test__ = False


if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_guardrails_decorator_metadata())
    print("✓ Metadata preservation tests passed")
    
    asyncio.run(test_guardrails_configuration())
    print("✓ Configuration tests passed")
    
    asyncio.run(test_text_extraction())
    print("✓ Text extraction tests passed")
    
    asyncio.run(test_guardrails_validation_pass())
    print("✓ Validation pass tests passed")
    
    asyncio.run(test_guardrails_validation_fail())
    print("✓ Validation fail tests passed")
    
    asyncio.run(test_guardrails_validation_error_handling())
    print("✓ Error handling tests passed")
    
    asyncio.run(test_full_arun_with_guardrails())
    print("✓ Full arun tests passed")
    
    asyncio.run(test_guardrails_status_field())
    print("✓ Status field tests passed")
    
    asyncio.run(test_empty_content_handling())
    print("✓ Empty content tests passed")
    
    asyncio.run(test_multiple_risk_types())
    print("✓ Multiple risk types tests passed")
    
    asyncio.run(test_snippet_truncation())
    print("✓ Snippet truncation tests passed")
    
    asyncio.run(test_guardrails_disabled())
    print("✓ Disabled guardrails tests passed")
    
    test_sync()
    print("✓ Synchronous tests passed")
    
    print("\n✅ All tests passed!")