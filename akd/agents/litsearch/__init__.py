"""
Literature Search Agents Module

This module contains specialized agents for literature search and research workflows,
including agentic search capabilities with embedded deep research components.
"""

from ._base import (
    LitBaseAgent,
    LitSearchAgentConfig,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
    SearchAgent,
    SearchAgentConfig,
    SearchAgentInputSchema,
    SearchAgentOutputSchema,
)
from .controlled import ControlledSearchAgent, ControlledSearchAgentConfig
from .deep_search import DeepLitSearchAgent, DeepLitSearchAgentConfig

__all__ = [
    # Base classes
    "SearchAgent",
    "SearchAgentConfig",
    "SearchAgentInputSchema",
    "SearchAgentOutputSchema",
    "LitBaseAgent",
    "LitSearchAgentInputSchema",
    "LitSearchAgentOutputSchema",
    "LitSearchAgentConfig",
    # Specific agents
    "ControlledSearchAgent",
    "ControlledSearchAgentConfig",
    "DeepLitSearchAgent",
    "DeepLitSearchAgentConfig",
]
