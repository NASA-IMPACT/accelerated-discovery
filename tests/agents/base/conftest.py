"""Shared fixtures and utilities for base agent tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import AKDAgent, BaseAgentConfig


# Shared test schemas
class AgentTestInputSchema(InputSchema):
    """Test input schema for base agents."""

    query: str = Field(..., description="Test query input")
    optional_param: str = Field(default="default", description="Optional parameter")


class AgentTestOutputSchema(OutputSchema):
    """Test output schema for base agents."""

    response: str = Field(..., description="Test response output")
    metadata: dict = Field(default_factory=dict, description="Response metadata")


class LiteLLMTestInputSchema(InputSchema):
    """Test input schema for LiteLLM agent."""

    query: str = Field(..., description="Test query input")
    context: str = Field(default="", description="Optional context")


class LiteLLMTestOutputSchema(OutputSchema):
    """Test output schema for LiteLLM agent."""

    response: str = Field(..., description="Test response output")
    confidence: float = Field(default=0.8, description="Response confidence")


class AgentTestCustomConfig(BaseAgentConfig):
    """Custom config for testing."""

    custom_field: str = Field(default="custom_value", description="Custom test field")
    custom_temperature: float = Field(default=0.5, description="Custom temperature")


# Test agent implementations
class TestInstructorBaseAgent(
    AKDAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Test implementation of AKDAgent (InstructorBaseAgent is now an alias for AKDAgent)."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    async def _arun(
        self,
        params: AgentTestInputSchema,
        **kwargs,
    ) -> AgentTestOutputSchema:
        """Test implementation that calls parent to handle memory."""
        result = await super()._arun(params, **kwargs)
        return result


class TestLiteLLMAgent(
    AKDAgent[LiteLLMTestInputSchema, LiteLLMTestOutputSchema],
):
    """Test implementation of AKDAgent."""

    input_schema = LiteLLMTestInputSchema
    output_schema = LiteLLMTestOutputSchema


# Prevent pytest from collecting test classes as tests themselves
TestInstructorBaseAgent.__test__ = False
TestLiteLLMAgent.__test__ = False


# Shared fixtures
@pytest.fixture
def default_config() -> BaseAgentConfig:
    """Create a default configuration for testing."""
    return BaseAgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=50000,
        trim_ratio=0.75,
        enable_trimming=True,
        stateless=True,
    )


@pytest.fixture
def custom_config() -> BaseAgentConfig:
    """Create a custom configuration for testing."""
    return BaseAgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.7,
        api_key="test_key",
        base_url="https://custom.api.com/v1",
        stateless=False,
        max_tokens=25000,
        trim_ratio=0.6,
        enable_trimming=False,
    )


@pytest.fixture
def litellm_config() -> BaseAgentConfig:
    """Create a LiteLLM-specific configuration for testing."""
    return BaseAgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=50000,
        trim_ratio=0.75,
        enable_trimming=True,
        stateless=True,
    )


@pytest.fixture
def test_input() -> AgentTestInputSchema:
    """Create a test input schema instance."""
    return AgentTestInputSchema(query="test query", optional_param="test_param")


@pytest.fixture
def litellm_test_input() -> LiteLLMTestInputSchema:
    """Create a LiteLLM test input schema instance."""
    return LiteLLMTestInputSchema(query="test query", context="test context")


@pytest.fixture
def expected_output() -> AgentTestOutputSchema:
    """Create expected output for testing."""
    return AgentTestOutputSchema(
        response="Test response",
        metadata={"test": True},
    )


@pytest.fixture
def litellm_expected_output() -> LiteLLMTestOutputSchema:
    """Create expected LiteLLM output for testing."""
    return LiteLLMTestOutputSchema(
        response="Test response",
        confidence=0.9,
    )


# Mock fixtures
@pytest.fixture
def mock_instructor_client():
    """Create a mock instructor client for testing.

    Patches instructor.from_litellm in the agent module so that
    AKDAgent.__init__ gets a mock client.
    """
    with patch("instructor.from_litellm") as mock_from_litellm:
        mock_client = MagicMock()
        mock_from_litellm.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_litellm_client():
    """Create a mock LiteLLM instructor client for testing."""
    with patch("instructor.from_litellm") as mock_from_litellm:
        mock_client = AsyncMock()
        mock_from_litellm.return_value = mock_client
        yield mock_client


# Helper functions
def create_config_with_overrides(
    base_config: BaseAgentConfig,
    **overrides,
) -> BaseAgentConfig:
    """Create a new config with specific field overrides."""
    config_dict = base_config.model_dump()
    config_dict.update(overrides)
    return BaseAgentConfig(**config_dict)


def setup_mock_response(
    mock_client: Any,
    response_data: dict[str, Any],
    client_type: str = "instructor",
) -> None:
    """Setup mock response for different client types."""
    mock_response = MagicMock()
    mock_response.model_dump.return_value = response_data
    # Instructor create_with_completion returns (response, completion) tuple
    mock_completion = MagicMock()
    mock_completion.usage = None
    mock_client.chat.completions.create_with_completion.return_value = (
        mock_response,
        mock_completion,
    )


async def setup_async_mock_response(
    mock_client: Any,
    response_data: dict[str, Any],
    client_type: str = "instructor",
) -> Any:
    """Setup async mock response for different client types.

    The instructor client's create_with_completion is called when output_schema is set.
    Returns the AsyncMock so callers can assert on it.
    """
    mock_response = MagicMock()
    mock_response.model_dump.return_value = response_data

    mock_completion = MagicMock()
    mock_completion.usage = None

    mock_create = AsyncMock(return_value=(mock_response, mock_completion))
    mock_client.chat.completions.create_with_completion = mock_create
    return mock_create
