"""
Agent Registry for the AKD Planner module.

This module provides agent registration and discovery capabilities for the AKD framework.
"""

from __future__ import annotations

import enum
import importlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, Union, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import IOSchema
from akd.agents._base import BaseAgent
from akd.utils import to_snake_case

from .config import AgentRegistryConfig


class FieldDefinition(BaseModel):
    """Individual field definition for agent schemas."""

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type")
    description: str = Field(..., description="Field description")
    required: bool = Field(default=True, description="Whether field is required")
    default: str | int | float | bool | list[Any] | None = Field(default=None, description="Default value if any")
    items_type: str | None = Field(default=None, description="Array item type")
    allowed_values: list[str] | None = Field(default=None, description="Allowed values for enum/Literal fields")
    value: str | int | float | bool | list[Any] | None = Field(
        default=None,
        description="Current value (defaults to default)",
    )


class AgentSchemaDefinition(BaseModel):
    """Schema definition for agent inputs/outputs."""

    fields: list[FieldDefinition] = Field(default_factory=list, description="List of field definitions")


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

    # Available agent mappings for auto-discovery.
    # Format: agent_id -> (module_path, class_name).
    #
    # akd core ships no built-in agents here — downstream packages (backends,
    # akd_ext) register their own agents at runtime via
    # `AgentRegistry.register_agent(YourAgent)`.
    AVAILABLE_AGENTS: dict[str, tuple[str, str]] = {}

    def __new__(cls, config: AgentRegistryConfig | None = None):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: AgentRegistryConfig | None = None):
        """Initialize the agent registry (only once due to singleton pattern)."""
        if not self._initialized:
            self.config = config or AgentRegistryConfig()
            self.registry_data: AgentRegistryData = AgentRegistryData()
            self._load_or_discover()
            AgentRegistry._initialized = True

    def __repr__(self) -> str:
        """Return a detailed string representation of the registry."""
        lines = ["AgentRegistry:"]
        lines.append(f"  Version: {self.registry_data.version}")
        lines.append(f"  Updated: {self.registry_data.updated_at}")
        lines.append(f"  Total Agents: {len(self.registry_data.agents)}")
        lines.append(f"  Enabled: {len(self.get_enabled_agents())}")
        lines.append("")
        lines.append("  Registered Agents:")
        for agent_id, agent in self.registry_data.agents.items():
            status = "enabled" if agent.enabled else "disabled"
            tags = ", ".join(agent.tags) if agent.tags else "none"
            lines.append(f"    - {agent_id} [{status}]")
            lines.append(f"        Class: {agent.agent_class}")
            lines.append(f"        Description: {agent.description}")
            lines.append(f"        Tags: {tags}")
            input_fields = [f.name for f in agent.input_schema.fields]
            output_fields = [f.name for f in agent.output_schema.fields]
            lines.append(f"        Input: {input_fields}")
            lines.append(f"        Output: {output_fields}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a concise string representation."""
        enabled = len(self.get_enabled_agents())
        total = len(self.registry_data.agents)
        agent_ids = list(self.registry_data.agents.keys())
        return f"AgentRegistry({enabled}/{total} enabled): {agent_ids}"

    def _load_or_discover(self) -> None:
        """
        Load registry with priority order:
        1. USE_AGENTS (explicit override) - highest priority
        2. JSON cache (if exists and auto_discover=False)
        3. Auto-discovery (if auto_discover=True)
        """
        try:
            # Priority 1: USE_AGENTS explicit override
            if self.config.use_agents is not None:
                logger.info(f"Using explicit agent list: {self.config.use_agents}")
                self._discover_agents(filter_agents=self.config.use_agents)
                # Don't save to file when using explicit override
                return

            # Priority 2 & 3: Normal flow
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

    def _discover_agents(self, filter_agents: list[str] | None = None) -> None:
        """
        Auto-discover agents by scanning known agent classes.

        Args:
            filter_agents: Optional list of agent IDs to discover. If provided, only these agents are discovered.
        """
        discovered: dict[str, AgentEntry] = {}

        for agent_id, (module_path, class_name) in self.AVAILABLE_AGENTS.items():
            # Filter agents if explicit list provided
            if filter_agents is not None:
                if agent_id not in filter_agents:
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

                # Get description: config default → docstring → auto-generated
                config_desc = getattr(getattr(agent_class, "config_schema", None), "model_fields", {}).get(
                    "description",
                )
                description = " ".join(
                    (
                        (config_desc.default if config_desc and config_desc.default else None)
                        or agent_class.__doc__
                        or f"Agent for {agent_id.replace('_', ' ')}"
                    ).split(),
                )

                discovered[agent_id] = AgentEntry(
                    agent_id=agent_id,
                    name=agent_id.replace("_", " ").title(),
                    description=description,
                    agent_class=f"{module_path}.{class_name}",
                    input_schema=input_schema,
                    output_schema=output_schema,
                    enabled=True,
                    tags=[agent_id.split("_")[0]],  # Simple tag based on first word
                    use_cases=[f"{description}"],
                )

                logger.debug(f"Discovered agent: {agent_id}")

            except Exception as e:
                logger.warning(f"Could not discover agent {agent_id}: {e}")
                continue

        self.registry_data = AgentRegistryData(agents=discovered)
        logger.info(f"Auto-discovered {len(discovered)} agents")

    @staticmethod
    def _resolve_enum_or_literal(annotation: type | None) -> list[str] | None:
        """Extract allowed values from enum or Literal type annotations.

        Handles: plain Enum, plain Literal, Optional[Enum], Optional[Literal],
        and Union wrappers containing a single constrained type.
        """
        if annotation is None:
            return None

        origin = get_origin(annotation)

        # Unwrap Optional / Union[X, None] to the inner type
        if origin is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]
                origin = get_origin(annotation)

        if origin is Literal:
            return [str(v) for v in get_args(annotation)]

        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return [str(member.value) for member in annotation]

        return None

    def _extract_schema(self, schema_class: type[IOSchema] | None) -> AgentSchemaDefinition:
        """Extract schema from an agent IOSchema class (InputSchema/OutputSchema)."""
        if not schema_class:
            return AgentSchemaDefinition()

        try:
            json_schema = schema_class.model_json_schema()
            properties: dict[str, Any] = json_schema.get("properties", {})
            required: list[str] = json_schema.get("required", [])

            fields = []
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "string")

                # Enum/Literal detection via model_fields (avoids fragile JSON schema parsing)
                allowed_values = None
                model_field = getattr(schema_class, "model_fields", {}).get(field_name)
                if model_field is not None:
                    allowed_values = self._resolve_enum_or_literal(model_field.annotation)
                    if allowed_values:
                        field_type = "enum"

                # Resolve type from anyOf when JSON schema doesn't expose a direct "type"
                if field_type == "string" and "anyOf" in field_info and not allowed_values:
                    for option in field_info["anyOf"]:
                        if "type" in option:
                            field_type = option["type"]
                            break

                field_def = FieldDefinition(
                    name=field_name,
                    type=field_type,
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
                    field_def.value = field_info["default"]
                    field_def.required = False

                # Set allowed values if detected
                if allowed_values:
                    field_def.allowed_values = allowed_values

                fields.append(field_def)

            return AgentSchemaDefinition(fields=fields)

        except Exception as e:
            logger.warning(f"Could not extract schema from {schema_class}: {e}")
            return AgentSchemaDefinition()

    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        try:
            # Ensure directory exists
            registry_dir = os.path.dirname(self.config.registry_path)
            if registry_dir:  # Only create directory if there is a parent directory
                os.makedirs(registry_dir, exist_ok=True)

            # Update timestamp
            self.registry_data.updated_at = datetime.now(timezone.utc).isoformat()

            with open(self.config.registry_path, "w") as f:
                json.dump(self.registry_data.model_dump(), f, indent=2)

            logger.info(f"Saved registry to {self.config.registry_path}")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def get_agent(self, agent_id: str) -> AgentEntry | None:
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

    def register_agent(
        self,
        agent_class: type,
        agent_id: str | None = None,
        enabled: bool = True,
        tags: list[str] | None = None,
        persist: bool = False,
    ) -> AgentEntry:
        """
        Register a new agent with the registry at runtime.

        Args:
            agent_class: Agent class to register. Must inherit from BaseAgent.
            agent_id: Unique identifier for the agent. If not provided, auto-generated
                      from class name (e.g., QueryAgent -> "query_agent", CMRAgent -> "cmr_agent")
            enabled: Whether agent is enabled (default: True)
            tags: Optional tags for categorization
            persist: Save to JSON file (default: False, transient registration)

        Returns:
            The created AgentEntry

        Raises:
            ValueError: If agent_id already exists
            TypeError: If agent_class does not inherit from BaseAgent

        Example:
            registry.register_agent(CMRAgent)  # auto-generates id "cmr_agent"
            registry.register_agent(CMRAgent, agent_id="my_cmr")
        """
        # Auto-generate agent_id from class name if not provided
        if agent_id is None:
            agent_id = to_snake_case(agent_class.__name__)

        if agent_id in self.registry_data.agents:
            raise ValueError(f"Agent '{agent_id}' already exists")

        agent_class_path = f"{agent_class.__module__}.{agent_class.__name__}"

        # Type check: ensure agent inherits from BaseAgent (works with full inheritance chain)
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(
                f"Agent class '{agent_class.__name__}' must inherit from BaseAgent",
            )

        # Extract schemas using existing method
        input_schema = self._extract_schema(getattr(agent_class, "input_schema", None))
        output_schema = self._extract_schema(getattr(agent_class, "output_schema", None))

        # Get description: config default → docstring → auto-generated
        config_desc = getattr(getattr(agent_class, "config_schema", None), "model_fields", {}).get("description")
        description = (
            (config_desc.default if config_desc and config_desc.default else None)
            or agent_class.__doc__
            or f"Agent for {agent_id.replace('_', ' ')}"
        ).strip()

        # Create entry
        entry = AgentEntry(
            agent_id=agent_id,
            name=agent_id.replace("_", " ").title(),
            description=description,
            agent_class=agent_class_path,
            input_schema=input_schema,
            output_schema=output_schema,
            enabled=enabled,
            tags=tags or ["external"],
            use_cases=[description],
        )

        # Add to registry
        self.registry_data.agents[agent_id] = entry
        logger.info(f"Registered agent: {agent_id}")

        # Persist if requested
        if persist:
            self._save_registry()

        return entry

    def unregister_agent(self, agent_id: str, persist: bool = False) -> AgentEntry | None:
        """
        Unregister an agent from the registry.

        Args:
            agent_id: Unique identifier of the agent to remove
            persist: Save changes to JSON file (default: False)

        Returns:
            The removed AgentEntry, or None if agent_id not found

        Example:
            removed = registry.unregister_agent("cmr_agent")
            if removed:
                print(f"Removed: {removed.agent_id}")
        """
        if agent_id not in self.registry_data.agents:
            logger.warning(f"Agent '{agent_id}' not found in registry")
            return None

        entry = self.registry_data.agents.pop(agent_id)
        logger.info(f"Unregistered agent: {agent_id}")

        if persist:
            self._save_registry()

        return entry

    def reload(self) -> None:
        """Reload the registry from file or re-discover."""
        self._load_or_discover()

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing purposes only)."""
        cls._instance = None
        cls._initialized = False


def get_agent_registry(config: AgentRegistryConfig | None = None) -> AgentRegistry:
    """Get the singleton agent registry instance."""
    return AgentRegistry(config)
