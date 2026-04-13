"""Test that agent/tool attributes reference config values instead of copying them.

This tests the fix in AbstractBase.__set_attrs_from_config() which creates properties
that reference self.config.field_name instead of copying primitive values.
"""

from pydantic import Field, computed_field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.tools._base import BaseTool, BaseToolConfig

# ============================================================================
# Test Fixtures - Agent
# ============================================================================


class RefTestAgentConfig(BaseAgentConfig):
    """Config for testing reference semantics in agents."""

    temperature: float = Field(default=0.7, description="Temperature setting")
    max_tokens: int = Field(default=100, description="Max tokens")
    system_prompt: str = Field(default="default prompt", description="System prompt")
    enabled: bool = Field(default=True, description="Whether agent is enabled")

    @computed_field
    @property
    def computed_value(self) -> str:
        """Computed field for testing read-only properties."""
        return f"temp={self.temperature}"


class RefTestAgentInput(InputSchema):
    """Test input schema."""

    query: str = Field(default="", description="Query string")


class RefTestAgentOutput(OutputSchema):
    """Test output schema."""

    result: str = Field(default="", description="Result string")


class RefTestAgent(LiteLLMInstructorBaseAgent):
    """Agent for testing config reference semantics."""

    input_schema = RefTestAgentInput
    output_schema = RefTestAgentOutput
    config_schema = RefTestAgentConfig


# ============================================================================
# Test Fixtures - Tool
# ============================================================================


class RefTestToolConfig(BaseToolConfig):
    """Config for testing reference semantics in tools."""

    timeout: int = Field(default=30, description="Timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries")
    api_key: str = Field(default="default_key", description="API key")
    debug: bool = Field(default=False, description="Debug mode")

    @computed_field
    @property
    def computed_timeout(self) -> str:
        """Computed field for testing."""
        return f"timeout={self.timeout}s"


class RefTestToolInput(InputSchema):
    """Test tool input schema."""

    query: str = Field(default="", description="Query")


class RefTestToolOutput(OutputSchema):
    """Test tool output schema."""

    result: str = Field(default="", description="Result")


class RefTestTool(BaseTool):
    """Tool for testing config reference semantics."""

    input_schema = RefTestToolInput
    output_schema = RefTestToolOutput
    config_schema = RefTestToolConfig

    async def _arun(self, params: RefTestToolInput) -> RefTestToolOutput:
        """Dummy async implementation."""
        return RefTestToolOutput(result="test")


# ============================================================================
# Tests - Agent Reference Semantics
# ============================================================================


def test_agent_float_field_references_config():
    """Test that float fields reference config values (not copies)."""
    config = RefTestAgentConfig(temperature=0.5)
    agent = RefTestAgent(config=config)

    # Initial values should match
    assert agent.temperature == 0.5
    assert agent.config.temperature == 0.5

    # Modifying via agent.field should update config.field
    agent.temperature = 0.9
    assert agent.config.temperature == 0.9, "Changing agent.temperature should update config.temperature"

    # Modifying via config.field should update agent.field
    agent.config.temperature = 0.3
    assert agent.temperature == 0.3, "Changing config.temperature should update agent.temperature"


def test_agent_int_field_references_config():
    """Test that int fields reference config values."""
    config = RefTestAgentConfig(max_tokens=200)
    agent = RefTestAgent(config=config)

    assert agent.max_tokens == 200

    # Modify via agent
    agent.max_tokens = 500
    assert agent.config.max_tokens == 500

    # Modify via config
    agent.config.max_tokens = 1000
    assert agent.max_tokens == 1000


def test_agent_string_field_references_config():
    """Test that string fields reference config values."""
    config = RefTestAgentConfig(system_prompt="initial prompt")
    agent = RefTestAgent(config=config)

    assert agent.system_prompt == "initial prompt"

    # Modify via agent
    agent.system_prompt = "modified prompt"
    assert agent.config.system_prompt == "modified prompt"

    # Modify via config
    agent.config.system_prompt = "another prompt"
    assert agent.system_prompt == "another prompt"


def test_agent_bool_field_references_config():
    """Test that boolean fields reference config values."""
    config = RefTestAgentConfig(enabled=True)
    agent = RefTestAgent(config=config)

    assert agent.enabled is True

    # Modify via agent
    agent.enabled = False
    assert agent.config.enabled is False

    # Modify via config
    agent.config.enabled = True
    assert agent.enabled is True


def test_agent_computed_field_is_read_only():
    """Test that computed fields are read-only properties."""
    config = RefTestAgentConfig(temperature=0.7)
    agent = RefTestAgent(config=config)

    # Should be able to read computed field
    assert agent.computed_value == "temp=0.7"

    # Computed field should update when underlying field changes
    agent.config.temperature = 0.9
    assert agent.computed_value == "temp=0.9"


def test_agent_config_update_reflects_in_agent():
    """Test that updating config after agent creation is reflected in agent attributes."""
    config = RefTestAgentConfig(temperature=0.5, max_tokens=100)
    agent = RefTestAgent(config=config)

    # Update config directly
    agent.config.temperature = 0.8
    agent.config.max_tokens = 500

    # Agent attributes should reflect the changes
    assert agent.temperature == 0.8
    assert agent.max_tokens == 500


def test_agent_update_reflects_in_config():
    """Test that updating agent attribute is reflected in config."""
    config = RefTestAgentConfig(temperature=0.5, max_tokens=100)
    agent = RefTestAgent(config=config)

    # Update via agent attributes
    agent.temperature = 0.2
    agent.max_tokens = 300

    # Config should reflect the changes
    assert agent.config.temperature == 0.2
    assert agent.config.max_tokens == 300


# ============================================================================
# Tests - Tool Reference Semantics
# ============================================================================


def test_tool_int_field_references_config():
    """Test that tool int fields reference config values."""
    config = RefTestToolConfig(timeout=30, retry_count=3)
    tool = RefTestTool(config=config)

    # Initial values should match
    assert tool.timeout == 30
    assert tool.retry_count == 3

    # Modify via tool
    tool.timeout = 60
    assert tool.config.timeout == 60

    # Modify via config
    tool.config.retry_count = 5
    assert tool.retry_count == 5


def test_tool_string_field_references_config():
    """Test that tool string fields reference config values."""
    config = RefTestToolConfig(api_key="initial_key")
    tool = RefTestTool(config=config)

    assert tool.api_key == "initial_key"

    # Modify via tool
    tool.api_key = "new_key"
    assert tool.config.api_key == "new_key"

    # Modify via config
    tool.config.api_key = "another_key"
    assert tool.api_key == "another_key"


def test_tool_bool_field_references_config():
    """Test that tool bool fields reference config values."""
    config = RefTestToolConfig(debug=False)
    tool = RefTestTool(config=config)

    assert tool.debug is False

    # Modify via tool
    tool.debug = True
    assert tool.config.debug is True

    # Modify via config
    tool.config.debug = False
    assert tool.debug is False


def test_tool_computed_field_is_read_only():
    """Test that tool computed fields are read-only properties."""
    config = RefTestToolConfig(timeout=30)
    tool = RefTestTool(config=config)

    # Should be able to read computed field
    assert tool.computed_timeout == "timeout=30s"

    # Computed field should update when underlying field changes
    tool.config.timeout = 60
    assert tool.computed_timeout == "timeout=60s"


# ============================================================================
# Tests - Multiple Instances
# ============================================================================


def test_multiple_agent_instances_have_independent_configs():
    """Test that multiple instances don't share config values."""
    config1 = RefTestAgentConfig(temperature=0.5)
    config2 = RefTestAgentConfig(temperature=0.8)

    agent1 = RefTestAgent(config=config1)
    agent2 = RefTestAgent(config=config2)

    # Each agent should have its own config values
    assert agent1.temperature == 0.5
    assert agent2.temperature == 0.8

    # Changing one should not affect the other
    agent1.temperature = 0.9
    assert agent1.temperature == 0.9
    assert agent2.temperature == 0.8  # Should remain unchanged


def test_multiple_tool_instances_have_independent_configs():
    """Test that multiple tool instances don't share config values."""
    config1 = RefTestToolConfig(timeout=30)
    config2 = RefTestToolConfig(timeout=60)

    tool1 = RefTestTool(config=config1)
    tool2 = RefTestTool(config=config2)

    # Each tool should have its own config values
    assert tool1.timeout == 30
    assert tool2.timeout == 60

    # Changing one should not affect the other
    tool1.timeout = 45
    assert tool1.timeout == 45
    assert tool2.timeout == 60  # Should remain unchanged


# ============================================================================
# Tests - Property Behavior
# ============================================================================


def test_config_attributes_are_properties():
    """Test that config attributes are implemented as properties."""
    config = RefTestAgentConfig(temperature=0.5)
    agent = RefTestAgent(config=config)

    # The attribute should be a property on the class (may be on parent class)
    assert isinstance(getattr(type(agent), "temperature", None), property)
    assert isinstance(getattr(type(agent), "max_tokens", None), property)
    assert isinstance(getattr(type(agent), "system_prompt", None), property)
