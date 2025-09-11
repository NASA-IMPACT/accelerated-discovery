"""
Agent Registry for the AKD Planner module.

This module provides agent registration and discovery capabilities for the AKD framework.
"""

import importlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import IOSchema

from .config import AgentRegistryConfig


class FieldDefinition(BaseModel):
    """Individual field definition for agent schemas."""
    
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type")
    description: str = Field(..., description="Field description")
    required: bool = Field(default=True, description="Whether field is required")
    default: str | int | float | bool | list[Any] | None = Field(default=None, description="Default value if any")
    items_type: Optional[str] = Field(default=None, description="Array item type")


class AgentSchemaDefinition(BaseModel):
    """Schema definition for agent inputs/outputs."""
    
    fields: list[FieldDefinition] = Field(
        default_factory=list,
        description="List of field definitions"
    )


class AgentEntry(BaseModel):
    """Registry entry for a single agent."""
    
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    agent_class: str = Field(description="Python import path to agent class")
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    input_schema: AgentSchemaDefinition = Field(description="Input schema definition")
    output_schema: AgentSchemaDefinition = Field(description="Output schema definition")
    tags: list[str] = Field(default_factory=list, description="Agent tags")
    use_cases: list[str] = Field(default_factory=list, description="Agent use cases")
    dependencies: list[str] = Field(default_factory=list, description="Agent dependencies")


class AgentRegistryData(BaseModel):
    """Top-level registry data structure."""
    
    version: str = Field(default="1.0.0", description="Registry format version")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agents: dict[str, AgentEntry] = Field(default_factory=dict, description="Agent entries")


class AgentRegistry:
    """
    Agent registry with auto-discovery capabilities.
    
    This registry can automatically discover agents from the akd.agents module
    and persist them to a JSON file for easy configuration.
    
    Implements singleton pattern to ensure only one registry instance exists.
    """
    
    # Singleton instance holder
    _instance = None
    _initialized = False
    
    # Known agent mappings for auto-discovery (currently hand-made)
    # Format: (agent_id, module_path, class_name)
    # TODO: Add filesystem scanning for automatic agent discovery in future iterations
    KNOWN_AGENTS: list[tuple[str, str, str]] = [
        ("query", "akd.agents.query", "QueryAgent"),
        ("followup_query", "akd.agents.query", "FollowUpQueryAgent"), 
        ("extraction", "akd.agents.extraction", "EstimationExtractionAgent"),
        ("relevancy", "akd.agents.relevancy", "MultiRubricRelevancyAgent"),
        ("intent", "akd.agents.intents", "IntentAgent"),
        ("controlled_search", "akd.agents.search.controlled", "ControlledSearchAgent"),
        ("deep_search", "akd.agents.search.deep_search", "DeepLitSearchAgent"),
        ("gap_analysis", "akd.agents.gap_analysis.gap_analysis", "GapAgent"),
        ("storm", "akd.agents.storm.storm", "StormAgent"),
        ("aspect_search", "akd.agents.search.aspect_search.aspect_search", "AspectSearchAgent"),
    ]
    
    def __new__(cls, config: Optional[AgentRegistryConfig] = None):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[AgentRegistryConfig] = None):
        """Initialize the agent registry (only once due to singleton pattern)."""
        if not self._initialized:
            self.config = config or AgentRegistryConfig()
            self.registry_data: AgentRegistryData = AgentRegistryData()
            self._load_or_discover()
            AgentRegistry._initialized = True
    
    def _load_or_discover(self) -> None:
        """Load registry from file, or auto-discover if missing/empty."""
        try:
            if self._should_auto_discover():
                logger.info("Auto-discovering agents...")
                self._discover_agents()
                self._save_registry()
            else:
                self._load_from_file()
        except Exception as e:
            logger.error(f"Failed to load/discover agents: {e}")
            if self.config.auto_discover:
                logger.info("Falling back to auto-discovery...")
                self._discover_agents()
    
    def _should_auto_discover(self) -> bool:
        """Check if we should auto-discover agents."""
        if not self.config.auto_discover:
            return False
        
        # Auto-discover if file doesn't exist or is empty
        if not os.path.exists(self.config.registry_path):
            return True
        
        try:
            with open(self.config.registry_path, "r") as f:
                data = json.load(f)
                return not data.get("agents", {})
        except (json.JSONDecodeError, FileNotFoundError):
            return True
    
    def _load_from_file(self) -> None:
        """Load registry from JSON file."""
        try:
            with open(self.config.registry_path, "r") as f:
                data = json.load(f)
            
            self.registry_data = AgentRegistryData(**data)
            
            logger.info(f"Loaded {len(self.registry_data.agents)} agents from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry from {self.config.registry_path}: {e}")
            raise
    
    def _discover_agents(self) -> None:
        """Auto-discover agents by scanning known agent classes."""
        discovered: dict[str, AgentEntry] = {}
        
        for agent_id, module_path, class_name in self.KNOWN_AGENTS:
            # Skip if we have specific enabled agents and this isn't one of them
            if self.config.enabled_agents and agent_id not in self.config.enabled_agents:
                continue
            
            try:
                # Import the module and get the class
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                
                # Extract schemas
                input_schema = self._extract_schema(getattr(agent_class, "input_schema", None))
                output_schema = self._extract_schema(getattr(agent_class, "output_schema", None))
                
                # Validate schemas if enabled
                if self.config.validate_schemas:
                    if not input_schema.fields and not output_schema.fields:
                        logger.warning(f"Agent {agent_id} has empty schemas, skipping")
                        continue
                
                # Get description from docstring
                description = agent_class.__doc__
                if description:
                    description = description.strip().split('\n')[0]
                else:
                    description = f"Agent for {agent_id.replace('_', ' ')}"
                
                discovered[agent_id] = AgentEntry(
                    agent_id=agent_id,
                    name=agent_id.replace("_", " ").title(),
                    description=description,
                    agent_class=f"{module_path}.{class_name}",
                    input_schema=input_schema,
                    output_schema=output_schema,
                    enabled=True,
                    tags=[agent_id.split("_")[0]],  # Simple tag based on first word
                    use_cases=[f"{description}"]
                )
                
                logger.debug(f"Discovered agent: {agent_id}")
                
            except Exception as e:
                logger.warning(f"Could not discover agent {agent_id}: {e}")
                continue
        
        self.registry_data = AgentRegistryData(agents=discovered)
        logger.info(f"Auto-discovered {len(discovered)} agents")
    
    def _extract_schema(self, schema_class: Optional[Type[IOSchema]]) -> AgentSchemaDefinition:
        """Extract schema from an agent IOSchema class (InputSchema/OutputSchema)."""
        if not schema_class:
            return AgentSchemaDefinition()
        
        try:
            # Get the JSON schema from the Pydantic model
            json_schema = schema_class.model_json_schema()
            properties: dict[str, Any] = json_schema.get("properties", {})
            required: list[str] = json_schema.get("required", [])
            
            fields = []
            for field_name, field_info in properties.items():
                field_def = FieldDefinition(
                    name=field_name,
                    type=field_info.get("type", "string"),
                    description=field_info.get("description", ""),
                    required=field_name in required,
                )
                
                # Handle array types
                if field_info.get("type") == "array":
                    items = field_info.get("items", {})
                    if items:
                        field_def.items_type = items.get("type", "string")
                
                # Handle default values
                if "default" in field_info:
                    field_def.default = field_info["default"]
                    field_def.required = False
                
                fields.append(field_def)
            
            return AgentSchemaDefinition(fields=fields)
            
        except Exception as e:
            logger.warning(f"Could not extract schema from {schema_class}: {e}")
            return AgentSchemaDefinition()
    
    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.registry_path), exist_ok=True)
            
            # Update timestamp
            self.registry_data.updated_at = datetime.now(timezone.utc).isoformat()
            
            with open(self.config.registry_path, "w") as f:
                json.dump(self.registry_data.model_dump(), f, indent=2)
            
            logger.info(f"Saved registry to {self.config.registry_path}")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def get_agent(self, agent_id: str) -> Optional[AgentEntry]:
        """Get a specific agent by ID."""
        return self.registry_data.agents.get(agent_id)
    
    def get_enabled_agents(self) -> list[AgentEntry]:
        """Get all enabled agents."""
        return [agent for agent in self.registry_data.agents.values() if agent.enabled]
    
    def get_agents_by_tag(self, tag: str) -> list[AgentEntry]:
        """Get all agents with a specific tag."""
        return [agent for agent in self.registry_data.agents.values() if tag in agent.tags and agent.enabled]
    
    def get_all_agents(self) -> list[AgentEntry]:
        """Get all agents (enabled and disabled)."""
        return list(self.registry_data.agents.values())
    
    def update_agent(self, agent_id: str, enabled: bool) -> bool:
        """Update an agent's enabled status."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.enabled = enabled
            self._save_registry()
            return True
        return False
    
    def reload(self) -> None:
        """Reload the registry from file or re-discover."""
        self._load_or_discover()


def get_agent_registry(config: Optional[AgentRegistryConfig] = None) -> AgentRegistry:
    """Get the singleton agent registry instance."""
    return AgentRegistry(config)