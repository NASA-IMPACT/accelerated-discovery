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
    SearchMode,
)
from .answer import (
    QuestionAnsweringAgent,
    QuestionAnsweringAgentInputSchema,
    QuestionAnsweringAgentOutputSchema,
)
from .code_search import CodeSearchAgent, CodeSearchAgentConfig
from .controlled import ControlledSearchAgent, ControlledSearchAgentConfig
from .deep_search import DeepLitSearchAgent, DeepLitSearchAgentConfig

__all__ = [
    "SearchMode",
    # Base classes
    "SearchAgent",
    "SearchAgentConfig",
    "SearchAgentInputSchema",
    "SearchAgentOutputSchema",
    "LitBaseAgent",
    "LitSearchAgentInputSchema",
    "LitSearchAgentOutputSchema",
    "LitSearchAgentConfig",
    # Question Answering agents
    "QuestionAnsweringAgent",
    "QuestionAnsweringAgentInputSchema",
    "QuestionAnsweringAgentOutputSchema",
    # Other specific agents
    "ControlledSearchAgent",
    "ControlledSearchAgentConfig",
    "DeepLitSearchAgent",
    "DeepLitSearchAgentConfig",
    "CodeSearchAgent",
    "CodeSearchAgentConfig",
]
