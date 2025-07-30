"""
Literature Search Agents Module

This module contains specialized agents for literature search and research workflows,
including agentic search capabilities with embedded deep research components.
"""

from ._base import (
    LitBaseAgent,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
    LitSearchAgentConfig,
)
from .controlled_agentic import (
    ControlledAgenticLitSearchAgent,
    ControlledAgenticLitSearchAgentConfig,
)
from .deep_search import (
    DeepLitSearchAgent,
    DeepLitSearchAgentConfig,
)

__all__ = [
    # Base classes
    "LitBaseAgent",
    "LitSearchAgentInputSchema",
    "LitSearchAgentOutputSchema",
    "LitSearchAgentConfig",
    # Specific agents
    "ControlledAgenticLitSearchAgent",
    "ControlledAgenticLitSearchAgentConfig", 
    "DeepLitSearchAgent",
    "DeepLitSearchAgentConfig",
]