"""
Unit tests for the AgentRegistry module.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from akd.planner.config import AgentRegistryConfig
from akd.planner.registry import (
    AgentEntry,
    AgentRegistry,
    AgentSchemaDefinition,
    FieldDefinition,
    get_agent_registry,
)


@pytest.fixture
def temp_registry_file():
    """Create a temporary registry file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        yield f.name
    # Clean up
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def sample_registry_data():
    """Sample registry data for testing."""
    return {
        "version": "1.0.0",
        "created_at": "2024-09-10T12:00:00.000Z",
        "updated_at": "2024-09-10T12:00:00.000Z",
        "agents": {
            "test_agent": {
                "agent_id": "test_agent",
                "name": "Test Agent",
                "description": "A test agent",
                "agent_class": "test.module.TestAgent",
                "enabled": True,
                "input_schema": {
                    "fields": [
                        {
                            "name": "input_field",
                            "type": "string",
                            "description": "Test input field",
                            "required": True,
                            "default": None,
                            "items_type": None,
                        },
                    ],
                },
                "output_schema": {
                    "fields": [
                        {
                            "name": "output_field",
                            "type": "string",
                            "description": "Test output field",
                            "required": True,
                            "default": None,
                            "items_type": None,
                        },
                    ],
                },
                "tags": ["test"],
                "use_cases": ["Testing"],
                "dependencies": [],
            },
        },
    }


class TestAgentRegistryConfig:
    """Test AgentRegistryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentRegistryConfig()
        # Path should end with the expected relative path
        assert config.registry_path.endswith("akd/mapping/agent_registry.json")
        assert config.auto_discover is True
        assert config.use_agents is None
        assert config.validate_schemas is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentRegistryConfig(
            registry_path="/custom/path.json",
            auto_discover=False,
            use_agents=["agent1", "agent2"],
            validate_schemas=False,
            debug=True,
        )
        assert config.registry_path == "/custom/path.json"
        assert config.auto_discover is False
        assert config.use_agents == ["agent1", "agent2"]
        assert config.validate_schemas is False
        assert config.debug is True


class TestAgentSchemaDefinition:
    """Test AgentSchemaDefinition class."""

    def test_empty_schema(self):
        """Test empty schema creation."""
        schema = AgentSchemaDefinition()
        assert schema.fields == []

    def test_schema_with_fields(self):
        """Test schema with field definitions."""
        field = FieldDefinition(name="test_field", type="string", description="A test field", required=True)
        schema = AgentSchemaDefinition(fields=[field])
        assert len(schema.fields) == 1
        assert schema.fields[0].name == "test_field"


class TestFieldDefinition:
    """Test FieldDefinition class."""

    def test_field_creation(self):
        """Test creating a field definition."""
        field = FieldDefinition(name="test_field", type="string", description="A test field", required=True)
        assert field.name == "test_field"
        assert field.type == "string"
        assert field.required is True
        assert field.default is None


class TestAgentEntry:
    """Test AgentEntry class."""

    def test_agent_entry_creation(self):
        """Test creating an agent entry."""
        entry = AgentEntry(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            agent_class="test.module.TestAgent",
            input_schema=AgentSchemaDefinition(),
            output_schema=AgentSchemaDefinition(),
        )
        assert entry.agent_id == "test_agent"
        assert entry.name == "Test Agent"
        assert entry.enabled is True  # default value


class TestAgentRegistry:
    """Test AgentRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Auto-reset singleton before and after each test."""
        AgentRegistry._reset_singleton()
        yield
        AgentRegistry._reset_singleton()

    def test_registry_with_existing_file(self, temp_registry_file, sample_registry_data):
        """Test loading registry from existing file."""
        # Write test data to file
        with open(temp_registry_file, "w") as f:
            json.dump(sample_registry_data, f)

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)

        assert len(registry.registry_data.agents) > 0
        assert "test_agent" in registry.registry_data.agents

        agent = registry.get_agent("test_agent")
        assert agent is not None
        assert agent.name == "Test Agent"

    def test_registry_auto_discovery(self, temp_registry_file):
        """Test auto-discovery when file is missing."""
        # Ensure file doesn't exist
        if os.path.exists(temp_registry_file):
            os.unlink(temp_registry_file)

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=True)

        with patch("akd.planner.registry.importlib.import_module") as mock_import:
            # Mock schema class with fields
            mock_schema_class = type(
                "MockSchema",
                (),
                {
                    "model_json_schema": lambda: {
                        "properties": {"test_field": {"type": "string", "description": "Test field"}},
                        "required": ["test_field"],
                    },
                },
            )

            # Mock a simple agent class
            mock_agent_class = type(
                "MockAgent",
                (),
                {
                    "input_schema": mock_schema_class,
                    "output_schema": mock_schema_class,
                    "__doc__": "Mock agent for testing",
                },
            )
            mock_module = type("Module", (), {"QueryAgent": mock_agent_class})
            mock_import.return_value = mock_module

            registry = AgentRegistry(config)

        # Should have discovered some agents (even if mocked)
        assert os.path.exists(temp_registry_file)  # File should be created

    def test_get_enabled_agents(self, temp_registry_file, sample_registry_data):
        """Test getting enabled agents."""
        # Modify sample data to have one disabled agent
        sample_registry_data["agents"]["disabled_agent"] = {
            **sample_registry_data["agents"]["test_agent"],
            "agent_id": "disabled_agent",
            "enabled": False,
        }

        with open(temp_registry_file, "w") as f:
            json.dump(sample_registry_data, f)

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)

        enabled_agents = registry.get_enabled_agents()
        assert len(enabled_agents) == 1
        assert enabled_agents[0].agent_id == "test_agent"

    def test_get_agents_by_tag(self, temp_registry_file, sample_registry_data):
        """Test getting agents by tag."""
        with open(temp_registry_file, "w") as f:
            json.dump(sample_registry_data, f)

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)

        test_agents = registry.get_agents_by_tag("test")
        assert len(test_agents) == 1
        assert test_agents[0].agent_id == "test_agent"

        # Test non-existent tag
        empty_agents = registry.get_agents_by_tag("nonexistent")
        assert len(empty_agents) == 0

    def test_update_agent(self, temp_registry_file, sample_registry_data):
        """Test updating agent enabled status."""
        with open(temp_registry_file, "w") as f:
            json.dump(sample_registry_data, f)

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)

        # Disable the agent
        result = registry.update_agent("test_agent", False)
        assert result is True

        agent = registry.get_agent("test_agent")
        assert agent.enabled is False

        # Try to update non-existent agent
        result = registry.update_agent("nonexistent", True)
        assert result is False

    def test_selective_agent_loading(self, temp_registry_file):
        """Test loading only specific agents using USE_AGENTS."""
        # Test with real agents that exist
        config = AgentRegistryConfig(
            registry_path=temp_registry_file,
            auto_discover=True,
            use_agents=["deep_search", "gap_analysis"],  # Use real agents
        )

        registry = AgentRegistry(config)

        # Should only discover the specified agents
        discovered_ids = list(registry.registry_data.agents.keys())
        assert "deep_search" in discovered_ids
        assert "gap_analysis" in discovered_ids
        assert "code_search" not in discovered_ids  # This one should NOT be loaded
        assert len(discovered_ids) == 2


class TestGlobalRegistry:
    """Test global registry functions."""

    def teardown_method(self):
        """Reset singleton state after each test."""
        AgentRegistry._reset_singleton()

    def test_get_agent_registry_singleton(self):
        """Test that get_agent_registry returns a singleton."""
        # Reset singleton instance
        AgentRegistry._reset_singleton()

        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        assert registry1 is registry2

    def test_get_agent_registry_with_config(self):
        """Test getting registry with custom config."""
        # Reset singleton instance
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=False)
        registry = get_agent_registry(config)

        assert registry.config.auto_discover is False


class TestRealAgents:
    """Test with real agents in the system."""

    def teardown_method(self):
        """Reset singleton state after each test."""
        AgentRegistry._reset_singleton()

    def test_discover_deep_search_agent(self):
        """Test discovering the DeepLitSearchAgent."""
        # Reset singleton
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["deep_search"])
        registry = AgentRegistry(config)

        agent = registry.get_agent("deep_search")
        assert agent is not None
        assert agent.agent_id == "deep_search"
        assert agent.name == "Deep Search"
        assert agent.enabled is True

        # Verify schemas have fields
        assert len(agent.input_schema.fields) > 0
        assert len(agent.output_schema.fields) > 0

        # Verify input schema has expected fields
        input_field_names = [f.name for f in agent.input_schema.fields]
        assert "query" in input_field_names

        # Verify output schema has expected fields
        output_field_names = [f.name for f in agent.output_schema.fields]
        assert "results" in output_field_names

    def test_discover_gap_analysis_agent(self):
        """Test discovering the GapAgent."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["gap_analysis"])
        registry = AgentRegistry(config)

        agent = registry.get_agent("gap_analysis")
        assert agent is not None
        assert agent.agent_id == "gap_analysis"
        assert agent.enabled is True

        # Verify schemas
        assert len(agent.input_schema.fields) > 0
        assert len(agent.output_schema.fields) > 0

        # Verify specific required inputs
        input_field_names = [f.name for f in agent.input_schema.fields if f.required]
        assert "gap" in input_field_names or "search_results" in input_field_names

    def test_discover_code_search_agent(self):
        """Test discovering the CodeSearchAgent."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["code_search"])
        registry = AgentRegistry(config)

        agent = registry.get_agent("code_search")
        assert agent is not None
        assert agent.agent_id == "code_search"
        assert agent.enabled is True

        # Verify schemas
        assert len(agent.input_schema.fields) > 0
        assert len(agent.output_schema.fields) > 0

    def test_discover_all_known_agents(self):
        """Test discovering all known agents."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True)
        registry = AgentRegistry(config)

        # Should discover all agents in KNOWN_AGENTS
        assert len(registry.registry_data.agents) >= 3
        assert "deep_search" in registry.registry_data.agents
        assert "gap_analysis" in registry.registry_data.agents
        assert "code_search" in registry.registry_data.agents

    def test_agent_schema_field_types(self):
        """Test that agent schemas have proper field types."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["deep_search"])
        registry = AgentRegistry(config)

        agent = registry.get_agent("deep_search")

        # Check that each field has required attributes
        for field in agent.input_schema.fields:
            assert field.name is not None
            assert field.type is not None
            assert field.description is not None
            assert isinstance(field.required, bool)

        for field in agent.output_schema.fields:
            assert field.name is not None
            assert field.type is not None
            assert field.description is not None

    def test_agent_class_path(self):
        """Test that agent_class path is correctly set."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["deep_search"])
        registry = AgentRegistry(config)

        agent = registry.get_agent("deep_search")
        assert agent.agent_class == "akd.agents.search.deep_search.DeepLitSearchAgent"

    def test_registry_persistence(self, temp_registry_file):
        """Test that registry can be saved and reloaded."""
        AgentRegistry._reset_singleton()

        # Create and save registry
        config1 = AgentRegistryConfig(
            registry_path=temp_registry_file,
            auto_discover=True,
            use_agents=["deep_search", "gap_analysis"],
        )
        registry1 = AgentRegistry(config1)
        original_count = len(registry1.registry_data.agents)

        # Save it
        registry1._save_registry()

        # Reset and reload
        AgentRegistry._reset_singleton()

        config2 = AgentRegistryConfig(
            registry_path=temp_registry_file,
            auto_discover=False,  # Load from file
        )
        registry2 = AgentRegistry(config2)

        # Should have same agents
        assert len(registry2.registry_data.agents) == original_count
        assert "deep_search" in registry2.registry_data.agents
        assert "gap_analysis" in registry2.registry_data.agents

        # Verify schemas are preserved
        agent = registry2.get_agent("deep_search")
        assert len(agent.input_schema.fields) > 0
        assert len(agent.output_schema.fields) > 0


class TestRegistryWithFieldMapping:
    """Test registry integration with field mapping system."""

    def teardown_method(self):
        """Reset singleton state after each test."""
        AgentRegistry._reset_singleton()

    def test_registry_provides_schemas_for_mapping(self):
        """Test that registry provides schemas needed for field mapping."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["deep_search", "gap_analysis"])
        registry = AgentRegistry(config)

        # Get both agents
        deep_search = registry.get_agent("deep_search")
        gap_analysis = registry.get_agent("gap_analysis")

        assert deep_search is not None
        assert gap_analysis is not None

        # Verify both have complete schemas
        assert len(deep_search.output_schema.fields) > 0
        assert len(gap_analysis.input_schema.fields) > 0

        # Get field names for mapping
        deep_search_outputs = [f.name for f in deep_search.output_schema.fields]
        gap_analysis_inputs = [f.name for f in gap_analysis.input_schema.fields]

        # Verify we can access field details for mapping
        for field in deep_search.output_schema.fields:
            assert hasattr(field, "name")
            assert hasattr(field, "type")
            assert hasattr(field, "description")

    def test_registry_agent_compatibility_check(self):
        """Test checking if agents can be chained based on schemas."""
        AgentRegistry._reset_singleton()

        config = AgentRegistryConfig(auto_discover=True, use_agents=["deep_search", "gap_analysis"])
        registry = AgentRegistry(config)

        deep_search = registry.get_agent("deep_search")
        gap_analysis = registry.get_agent("gap_analysis")

        # Check if deep_search outputs can satisfy gap_analysis inputs
        output_fields = {f.name for f in deep_search.output_schema.fields}
        required_inputs = {f.name for f in gap_analysis.input_schema.fields if f.required}

        # With field mapping, we don't need exact matches
        # But we should have at least some output fields
        assert len(output_fields) > 0
        assert len(required_inputs) > 0


class TestRegisterUnregisterAgent:
    """Test register_agent and unregister_agent methods."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Auto-reset singleton before and after each test."""
        AgentRegistry._reset_singleton()
        yield
        AgentRegistry._reset_singleton()

    def test_register_agent_with_class(self):
        """Test registering an agent with class directly."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        entry = registry.register_agent(agent_class=IntentAgent)

        assert entry.agent_id == "intent_agent"
        assert entry.agent_class == "akd.agents.intents.IntentAgent"
        assert "external" in entry.tags
        assert entry.enabled is True

        # Verify it's in the registry
        assert registry.get_agent("intent_agent") is not None

    def test_register_agent_with_custom_id(self):
        """Test registering an agent with custom agent_id."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        entry = registry.register_agent(agent_id="my_custom_intent", agent_class=IntentAgent)

        assert entry.agent_id == "my_custom_intent"
        assert registry.get_agent("my_custom_intent") is not None

    def test_register_agent_with_module_path(self):
        """Test registering an agent with module_path and class_name."""
        registry = get_agent_registry()
        entry = registry.register_agent(
            agent_id="intent_from_path",
            module_path="akd.agents.intents",
            class_name="IntentAgent",
        )

        assert entry.agent_id == "intent_from_path"
        assert entry.agent_class == "akd.agents.intents.IntentAgent"

    def test_register_agent_with_custom_tags(self):
        """Test registering an agent with custom tags."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        entry = registry.register_agent(
            agent_class=IntentAgent,
            tags=["custom", "test", "external"],
        )

        assert "custom" in entry.tags
        assert "test" in entry.tags

    def test_register_agent_auto_id_from_class_name(self):
        """Test auto-generation of agent_id from class name."""
        from akd.agents.query import QueryAgent
        from akd.agents.relevancy import RelevancyAgent

        registry = get_agent_registry()

        # QueryAgent -> query_agent
        entry1 = registry.register_agent(agent_class=QueryAgent)
        assert entry1.agent_id == "query_agent"

        # RelevancyAgent -> relevancy_agent
        entry2 = registry.register_agent(agent_class=RelevancyAgent)
        assert entry2.agent_id == "relevancy_agent"

    def test_register_agent_duplicate_raises_error(self):
        """Test that registering duplicate agent_id raises ValueError."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        registry.register_agent(agent_id="duplicate_test", agent_class=IntentAgent)

        with pytest.raises(ValueError, match="already exists"):
            registry.register_agent(agent_id="duplicate_test", agent_class=IntentAgent)

    def test_register_agent_non_base_agent_raises_error(self):
        """Test that registering non-BaseAgent class raises TypeError."""
        registry = get_agent_registry()

        with pytest.raises(TypeError, match="must inherit from BaseAgent"):
            registry.register_agent(agent_id="invalid", agent_class=str)

    def test_register_agent_missing_args_raises_error(self):
        """Test that missing required args raises ValueError."""
        registry = get_agent_registry()

        with pytest.raises(ValueError, match="Provide either agent_class"):
            registry.register_agent(agent_id="no_class")

    def test_register_agent_extracts_schemas(self):
        """Test that schemas are correctly extracted from agent class."""
        from akd.agents.query import QueryAgent

        registry = get_agent_registry()
        entry = registry.register_agent(agent_class=QueryAgent)

        # Verify input schema has fields
        input_field_names = [f.name for f in entry.input_schema.fields]
        assert "query" in input_field_names

        # Verify output schema has fields
        output_field_names = [f.name for f in entry.output_schema.fields]
        assert len(output_field_names) > 0

    def test_register_agent_persist_false_by_default(self, temp_registry_file):
        """Test that persist=False is the default (transient registration)."""
        from akd.agents.intents import IntentAgent

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=True)
        registry = AgentRegistry(config)

        # Register without persist (default)
        registry.register_agent(agent_class=IntentAgent)

        # Read file - should not contain the new agent
        with open(temp_registry_file, "r") as f:
            data = json.load(f)

        assert "intent_agent" not in data.get("agents", {})

    def test_register_agent_persist_true(self, temp_registry_file):
        """Test that persist=True saves to JSON file."""
        from akd.agents.intents import IntentAgent

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=True)
        registry = AgentRegistry(config)

        # Register with persist=True
        registry.register_agent(agent_class=IntentAgent, persist=True)

        # Read file - should contain the new agent
        with open(temp_registry_file, "r") as f:
            data = json.load(f)

        assert "intent_agent" in data.get("agents", {})

    def test_unregister_agent_returns_entry(self):
        """Test unregistering an agent returns the removed entry."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        registry.register_agent(agent_class=IntentAgent)

        removed = registry.unregister_agent("intent_agent")

        assert removed is not None
        assert removed.agent_id == "intent_agent"
        assert registry.get_agent("intent_agent") is None

    def test_unregister_agent_not_found_returns_none(self):
        """Test unregistering non-existent agent returns None."""
        registry = get_agent_registry()

        removed = registry.unregister_agent("does_not_exist")

        assert removed is None

    def test_unregister_agent_persist_false_by_default(self, temp_registry_file):
        """Test that unregister persist=False is the default."""
        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=True)
        registry = AgentRegistry(config)

        # Get an existing agent from file
        agents_before = list(registry.registry_data.agents.keys())
        if agents_before:
            agent_to_remove = agents_before[0]
            registry.unregister_agent(agent_to_remove)

            # Read file - should still contain the agent
            with open(temp_registry_file, "r") as f:
                data = json.load(f)

            assert agent_to_remove in data.get("agents", {})

    def test_unregister_agent_persist_true(self, temp_registry_file):
        """Test that unregister with persist=True updates JSON file."""
        from akd.agents.intents import IntentAgent

        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=True)
        registry = AgentRegistry(config)

        # Register with persist
        registry.register_agent(agent_class=IntentAgent, persist=True)

        # Verify it's in file
        with open(temp_registry_file, "r") as f:
            data = json.load(f)
        assert "intent_agent" in data.get("agents", {})

        # Unregister with persist
        registry.unregister_agent("intent_agent", persist=True)

        # Verify it's removed from file
        with open(temp_registry_file, "r") as f:
            data = json.load(f)
        assert "intent_agent" not in data.get("agents", {})

    def test_register_unregister_cycle(self):
        """Test registering and unregistering multiple times."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()

        # Register
        entry1 = registry.register_agent(agent_class=IntentAgent)
        assert registry.get_agent("intent_agent") is not None

        # Unregister
        removed = registry.unregister_agent("intent_agent")
        assert removed.agent_id == entry1.agent_id
        assert registry.get_agent("intent_agent") is None

        # Register again
        entry2 = registry.register_agent(agent_class=IntentAgent)
        assert registry.get_agent("intent_agent") is not None
        assert entry2.agent_id == "intent_agent"

    def test_registered_agent_appears_in_enabled_agents(self):
        """Test that registered agent appears in get_enabled_agents()."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        registry.register_agent(agent_class=IntentAgent)

        enabled = registry.get_enabled_agents()
        enabled_ids = [a.agent_id for a in enabled]

        assert "intent_agent" in enabled_ids

    def test_registered_agent_with_disabled_flag(self):
        """Test registering an agent as disabled."""
        from akd.agents.intents import IntentAgent

        registry = get_agent_registry()
        entry = registry.register_agent(agent_class=IntentAgent, enabled=False)

        assert entry.enabled is False

        enabled = registry.get_enabled_agents()
        enabled_ids = [a.agent_id for a in enabled]

        assert "intent_agent" not in enabled_ids


if __name__ == "__main__":
    pytest.main([__file__])
