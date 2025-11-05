"""
Configuration for the AKD Planner module.
"""

from pydantic import Field

from akd._base import BaseConfig
from akd.utils import get_akd_root


class AgentRegistryConfig(BaseConfig):
    """Configuration for the Agent Registry."""

    registry_path: str = Field(
        default=str(get_akd_root() / "akd" / "mapping" / "eie_agent_registry.json"),
        description="Path to the agent registry JSON file",
    )

    auto_discover: bool = Field(default=True, description="Auto-populate registry if empty or missing")

    use_agents: list[str] | None = Field(
        default=None,
        description="Explicit list of agent IDs to use. Overrides JSON and auto-discovery. None = use all discovered agents",
    )

    validate_schemas: bool = Field(default=True, description="Validate agent schemas during discovery")
