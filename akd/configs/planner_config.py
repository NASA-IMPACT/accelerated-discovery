from typing import List, Optional, Dict, Any
from pydantic import Field
from akd._base import BaseConfig
from akd.configs.project import ModelConfigSettings


class PlannerConfig(BaseConfig):
    """Enhanced configuration for LLM-based research planner with reasoning capabilities"""
    
    # Inherit from existing model config
    model_config_settings: ModelConfigSettings = Field(default_factory=ModelConfigSettings)
    
    # Agent discovery settings
    auto_discover_agents: bool = Field(
        True, description="Automatically discover available agents"
    )
    agent_packages: List[str] = Field(
        default=["akd.agents", "akd.nodes"],
        description="Packages to scan for agents"
    )
    
    # LLM Planning settings
    max_workflow_nodes: int = Field(
        20, description="Maximum nodes in a workflow"
    )
    planning_model: str = Field(
        default="gpt-4o-mini",
        description="Model for planning operations (prefer o3/o1 for reasoning)"
    )
    enable_reasoning_traces: bool = Field(
        True, description="Capture reasoning traces from o3/o1 models"
    )
    reasoning_model_preference: List[str] = Field(
        default=["o3-mini", "o3", "o1-preview", "o1-mini", "gpt-4o"],
        description="Preferred models for reasoning in order of preference"
    )
    
    # Agent analysis settings
    enable_agent_analysis: bool = Field(
        True, description="Enable LLM-based agent capability analysis"
    )
    agent_analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Model for agent capability analysis"
    )
    enable_compatibility_analysis: bool = Field(
        True, description="Enable agent compatibility analysis"
    )
    compatibility_threshold: float = Field(
        0.7, description="Minimum compatibility score for agent connections"
    )
    
    # LangGraph execution settings
    execution_strategy: str = Field(
        "langgraph", description="Execution strategy: mock, langgraph"
    )
    enable_checkpointing: bool = Field(
        True, description="Enable LangGraph checkpointing"
    )
    checkpoint_storage: str = Field(
        "memory", description="Checkpoint storage: memory, sqlite, postgres"
    )
    checkpoint_namespace: str = Field(
        "akd_planner", description="Namespace for checkpoints"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        True, description="Cache agent profiles and compatibility scores"
    )
    cache_ttl_hours: int = Field(
        24, description="Cache time-to-live in hours"
    )
    max_parallel_analysis: int = Field(
        10, description="Maximum parallel agent analysis tasks"
    )
    
    # Monitoring and visualization
    enable_monitoring: bool = Field(
        default=True,
        description="Enable execution monitoring"
    )
    enable_visualization: bool = Field(
        False, description="Generate workflow visualizations"
    )
    
    # Session management
    session_dir: Optional[str] = Field(
        "./planner_sessions", description="Directory for session persistence"
    )
    auto_save_session: bool = Field(
        True, description="Automatically save session state"
    )
    
    # React/streaming integration
    enable_streaming: bool = Field(
        True, description="Enable streaming for React interface"
    )
    stream_intermediate_steps: bool = Field(
        True, description="Stream intermediate node results"
    )
    
    # Advanced LangGraph features
    enable_time_travel: bool = Field(
        True, description="Enable time-travel debugging"
    )
    max_checkpoint_history: int = Field(
        100, description="Maximum checkpoints to keep in history"
    )
    enable_conditional_edges: bool = Field(
        True, description="Enable conditional workflow paths"
    )