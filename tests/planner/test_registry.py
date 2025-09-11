"""
Unit tests for the AgentRegistry module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from akd.planner.config import AgentRegistryConfig
from akd.planner.registry import (
    AgentEntry,
    AgentRegistry,
    AgentRegistryData,
    AgentSchemaDefinition,
    FieldDefinition,
    get_agent_registry,
)


@pytest.fixture
def temp_registry_file():
    """Create a temporary registry file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
                            "items_type": None
                        }
                    ]
                },
                "output_schema": {
                    "fields": [
                        {
                            "name": "output_field",
                            "type": "string", 
                            "description": "Test output field",
                            "required": True,
                            "default": None,
                            "items_type": None
                        }
                    ]
                },
                "tags": ["test"],
                "use_cases": ["Testing"],
                "dependencies": []
            }
        }
    }




class TestAgentRegistryConfig:
    """Test AgentRegistryConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentRegistryConfig()
        # Path should end with the expected relative path
        assert config.registry_path.endswith("akd/mapping/agent_registry.json")
        assert config.auto_discover is True
        assert config.enabled_agents == []
        assert config.validate_schemas is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentRegistryConfig(
            registry_path="/custom/path.json",
            auto_discover=False,
            enabled_agents=["agent1", "agent2"],
            validate_schemas=False,
            debug=True
        )
        assert config.registry_path == "/custom/path.json"
        assert config.auto_discover is False
        assert config.enabled_agents == ["agent1", "agent2"]
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
        field = FieldDefinition(
            name="test_field",
            type="string",
            description="A test field",
            required=True
        )
        schema = AgentSchemaDefinition(fields=[field])
        assert len(schema.fields) == 1
        assert schema.fields[0].name == "test_field"


class TestFieldDefinition:
    """Test FieldDefinition class."""
    
    def test_field_creation(self):
        """Test creating a field definition."""
        field = FieldDefinition(
            name="test_field",
            type="string",
            description="A test field",
            required=True
        )
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
            output_schema=AgentSchemaDefinition()
        )
        assert entry.agent_id == "test_agent"
        assert entry.name == "Test Agent"
        assert entry.enabled is True  # default value


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    def teardown_method(self):
        """Reset singleton state after each test."""
        AgentRegistry._reset_singleton()
    
    def test_registry_with_existing_file(self, temp_registry_file, sample_registry_data):
        """Test loading registry from existing file."""
        # Write test data to file
        with open(temp_registry_file, 'w') as f:
            json.dump(sample_registry_data, f)
        
        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)
        
        assert len(registry.registry_data.agents) == 1
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
        
        with patch('akd.planner.registry.importlib.import_module') as mock_import:
            # Mock schema class with fields
            mock_schema_class = type('MockSchema', (), {
                'model_json_schema': lambda: {
                    'properties': {
                        'test_field': {
                            'type': 'string',
                            'description': 'Test field'
                        }
                    },
                    'required': ['test_field']
                }
            })
            
            # Mock a simple agent class
            mock_agent_class = type('MockAgent', (), {
                'input_schema': mock_schema_class,
                'output_schema': mock_schema_class,
                '__doc__': 'Mock agent for testing'
            })
            mock_module = type('Module', (), {'QueryAgent': mock_agent_class})
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
            "enabled": False
        }
        
        with open(temp_registry_file, 'w') as f:
            json.dump(sample_registry_data, f)
        
        config = AgentRegistryConfig(registry_path=temp_registry_file, auto_discover=False)
        registry = AgentRegistry(config)
        
        enabled_agents = registry.get_enabled_agents()
        assert len(enabled_agents) == 1
        assert enabled_agents[0].agent_id == "test_agent"
    
    def test_get_agents_by_tag(self, temp_registry_file, sample_registry_data):
        """Test getting agents by tag."""
        with open(temp_registry_file, 'w') as f:
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
        with open(temp_registry_file, 'w') as f:
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
        """Test loading only specific agents."""
        config = AgentRegistryConfig(
            registry_path=temp_registry_file,
            auto_discover=True,
            enabled_agents=["query", "extraction"]  # Only these agents
        )
        
        with patch('akd.planner.registry.importlib.import_module') as mock_import:
            # Mock schema class with fields
            mock_schema_class = type('MockSchema', (), {
                'model_json_schema': lambda: {
                    'properties': {
                        'test_field': {
                            'type': 'string',
                            'description': 'Test field'
                        }
                    },
                    'required': ['test_field']
                }
            })
            
            # Mock agent classes for the ones we want
            mock_agent_class = type('MockAgent', (), {
                'input_schema': mock_schema_class,
                'output_schema': mock_schema_class,
                '__doc__': 'Mock agent for testing'
            })
            
            # Mock different modules for different imports
            def mock_import_func(module_path):
                if module_path == 'akd.agents.query':
                    return type('Module', (), {'QueryAgent': mock_agent_class})
                elif module_path == 'akd.agents.extraction':
                    return type('Module', (), {'EstimationExtractionAgent': mock_agent_class})
                else:
                    return type('Module', (), {})
            
            mock_import.side_effect = mock_import_func
            
            registry = AgentRegistry(config)
            
            # Should only discover the enabled agents
            discovered_ids = list(registry.registry_data.agents.keys())
            assert "query" in discovered_ids
            assert "extraction" in discovered_ids


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


if __name__ == "__main__":
    pytest.main([__file__])