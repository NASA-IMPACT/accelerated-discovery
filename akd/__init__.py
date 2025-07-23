"""
AKD (Accelerated Knowledge Discovery)

A human-centric Multi-Agent System (MAS) framework for scientific discovery.

This framework prioritizes scientific integrity, human control, and reproducible
research through:
- Human-in-the-loop control for researchers to direct discovery processes
- Scientific integrity with deep attribution and conflicting evidence identification
- Transparent & reproducible research with complete workflow tracking
- Framework agnostic core logic decoupled from orchestration engines
"""

__version__ = "0.1.0"
__author__ = "NASA IMPACT"
__email__ = "np0069@uah.edu,mr0051@uah.edu"

# Core base classes
from akd._base import (
    AbstractBase,
    BaseConfig,
    InputSchema,
    IOSchema,
    OutputSchema,
    UnrestrictedAbstractBase,
)

# Agent system
from akd.agents._base import BaseAgent, BaseAgentConfig

# Configuration
from akd.configs.project import CONFIG
from akd.nodes.states import GlobalState, NodeState
from akd.nodes.supervisor import BaseSupervisor

# Node template system
from akd.nodes.templates import AbstractNodeTemplate

# Core structures
from akd.structures import (
    ExtractionSchema,
    ResearchData,
    SearchResultItem,
    SingleEstimation,
    ToolSearchResult,
)

# Tool system
from akd.tools._base import BaseTool, BaseToolConfig

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Base classes
    "AbstractBase",
    "UnrestrictedAbstractBase",
    "BaseConfig",
    "IOSchema",
    "InputSchema",
    "OutputSchema",
    # Agent system
    "BaseAgent",
    "BaseAgentConfig",
    # Tool system
    "BaseTool",
    "BaseToolConfig",
    # Node template system
    "AbstractNodeTemplate",
    "GlobalState",
    "NodeState",
    "BaseSupervisor",
    # Core structures
    "SearchResultItem",
    "ResearchData",
    "ExtractionSchema",
    "SingleEstimation",
    "ToolSearchResult",
    # Configuration
    "CONFIG",
]
