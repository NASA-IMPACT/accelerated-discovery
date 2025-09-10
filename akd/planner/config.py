"""
Configuration for the AKD Planner module.
"""

from typing import List
from pydantic import Field

from akd._base import BaseConfig


class AgentRegistryConfig(BaseConfig):
    """Configuration for the Agent Registry."""
    
    registry_path: str = Field(
        default="akd/mapping/agent_registry.json",
        description="Path to the agent registry JSON file"
    )
    
    auto_discover: bool = Field(
        default=True, 
        description="Auto-populate registry if empty or missing"
    )
    
    enabled_agents: List[str] = Field(
        default_factory=list,
        description="List of enabled agent IDs. Empty list enables all agents"
    )
    
    validate_schemas: bool = Field(
        default=True,
        description="Validate agent schemas during discovery"
    )