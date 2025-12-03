"""Shared fixtures and utilities for base agent tests."""

from typing import Annotated, Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd._base.exposure import Exposed
from akd.agents._base import (
    BaseAgentConfig,
    InstructorBaseAgent,
    LiteLLMInstructorBaseAgent,
)


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


class TestLiteLLMAgent(
    LiteLLMInstructorBaseAgent[LiteLLMTestInputSchema, LiteLLMTestOutputSchema],
):
    """Test implementation of LiteLLMInstructorBaseAgent."""

    input_schema = LiteLLMTestInputSchema
    output_schema = LiteLLMTestOutputSchema


class DummyExposedConfigSchema(BaseAgentConfig):
    """Dummy config schema with exposed parameters for testing."""

    topic_steer: Annotated[str, Exposed(description="Topic steering parameter")] = "default"
    temperature: Annotated[float, Exposed(description="Temperature setting")] = 0.7


class DummyAgentWithExposedConfig(
    LiteLLMInstructorBaseAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Dummy agent with exposed config schema fields for testing."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema
    config_schema = DummyExposedConfigSchema


# ============================================================================
# Exposure Testing Components
# ============================================================================


class ExposedSimpleComponent:
    """Simple component with exposed field for testing."""

    value: Annotated[str, Exposed(description="Simple value")] = "default"


class ExposedNestedConfig(BaseAgentConfig):
    """Nested configuration with exposed parameters."""

    temperature: Annotated[float, Exposed(description="Temperature parameter")] = 0.7
    max_tokens: Annotated[int, Exposed(description="Maximum tokens")] = 100


class ExposedComplexComponent:
    """Complex component with nested configuration for testing."""

    name: Annotated[str, Exposed(description="Component name")] = "component"

    def __init__(self):
        self.config = ExposedNestedConfig()


class ExposedAgentWithSimpleComponent(
    LiteLLMInstructorBaseAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Agent with simple nested component for exposure testing."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    def __init__(self, config: BaseAgentConfig | None = None):
        super().__init__(config)
        self.component = ExposedSimpleComponent()


class ExposedAgentWithComplexComponent(
    LiteLLMInstructorBaseAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Agent with complex nested component for exposure testing."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    def __init__(self, config: BaseAgentConfig | None = None):
        super().__init__(config)
        self.component = ExposedComplexComponent()


class ExposedAgentWithMultipleComponents(
    LiteLLMInstructorBaseAgent[AgentTestInputSchema, AgentTestOutputSchema],
):
    """Agent with multiple nested components for exposure testing."""

    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema

    def __init__(self, config: BaseAgentConfig | None = None):
        super().__init__(config)
        self.simple = ExposedSimpleComponent()
        self.complex = ExposedComplexComponent()
        self.pipe_mixed = PipeMixedComponent()


# ============================================================================
# Pipe Operator Testing Components (Instance-Level Exposure)
# ============================================================================


class PipeNestedComponent:
    """Nested component using pipe operator for instance-level exposure."""

    def __init__(self):
        self.nested_field = "deeply nested value" | Exposed(
            description="Nested field via pipe",
        )


class PipeMixedComponent:
    """Component mixing class-level annotations and pipe operator."""

    class_field: Annotated[str, Exposed(description="Class-level field")] = "class_value"

    def __init__(self):
        self.instance_field = "instance_value" | Exposed(description="Instance field via pipe")
        self.nested = PipeNestedComponent()


# Prevent pytest from collecting test classes as tests themselves
TestInstructorBaseAgent.__test__ = False
TestLiteLLMAgent.__test__ = False
DummyAgentWithExposedConfig.__test__ = False
ExposedAgentWithSimpleComponent.__test__ = False
ExposedAgentWithComplexComponent.__test__ = False
ExposedAgentWithMultipleComponents.__test__ = False


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


# Exposure testing fixtures
@pytest.fixture
def exposed_simple_agent(default_config: BaseAgentConfig) -> ExposedAgentWithSimpleComponent:
    """Create agent with simple nested component for exposure testing."""
    return ExposedAgentWithSimpleComponent(default_config)


@pytest.fixture
def exposed_complex_agent(default_config: BaseAgentConfig) -> ExposedAgentWithComplexComponent:
    """Create agent with complex nested component for exposure testing."""
    return ExposedAgentWithComplexComponent(default_config)


@pytest.fixture
def exposed_multi_agent(default_config: BaseAgentConfig) -> ExposedAgentWithMultipleComponents:
    """Create agent with multiple nested components for exposure testing."""
    return ExposedAgentWithMultipleComponents(default_config)


# Mock fixtures
@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    with patch("akd.agents._base.openai.AsyncOpenAI") as mock_openai:
        mock_openai_client = MagicMock()
        mock_openai.return_value = mock_openai_client
        yield mock_openai_client


@pytest.fixture
def mock_instructor_client():
    """Create a mock instructor client for testing."""
    with patch("akd.agents._base.instructor.from_openai") as mock_instructor:
        mock_client = MagicMock()
        mock_instructor.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_litellm_client():
    """Create a mock LiteLLM instructor client for testing."""
    with patch("instructor.from_litellm") as mock_instructor:
        mock_client = AsyncMock()
        mock_instructor.return_value = mock_client
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
    response_data: Dict[str, Any],
    client_type: str = "instructor",
) -> None:
    """Setup mock response for different client types."""
    if client_type == "instructor":
        mock_response = MagicMock()
        mock_response.model_dump.return_value = response_data
        mock_client.chat.completions.create.return_value = mock_response
    elif client_type == "litellm":
        mock_response = MagicMock()
        mock_response.model_dump.return_value = response_data
        mock_client.chat.completions.create.return_value = mock_response


async def setup_async_mock_response(
    mock_client: Any,
    response_data: Dict[str, Any],
    client_type: str = "instructor",
) -> Any:
    """Setup async mock response for different client types."""
    if client_type == "instructor":
        mock_chat_completions = AsyncMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = response_data
        mock_client.chat.completions.create = mock_chat_completions
        mock_chat_completions.return_value = mock_response
        return mock_chat_completions
    elif client_type == "litellm":
        mock_response = MagicMock()
        mock_response.model_dump.return_value = response_data
        mock_client.chat.completions.create.return_value = mock_response
        return mock_response
