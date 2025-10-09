"""
Base classes for repository-specific data search handlers.

Handlers encapsulate repository-specific logic for processing scientific
decompositions into data results.
"""

from abc import ABC, abstractmethod
from typing import Any

from akd.agents.data_search._base import DataSearchAgentInputSchema, DecompositionResult
from akd.agents.data_search.components.scientific_decomposition import (
    ScientificDecomposition,
)
from akd.agents.data_search.components.topic_splitting import Topic


class BaseHandler(ABC):
    """
    Abstract base class for repository-specific data search handlers.

    Each repository (CMR, PDS4, SPASE, etc.) implements its own handler
    that knows how to:
    - Extract parameters from scientific decompositions
    - Generate repository-specific queries
    - Search for and rank data results
    - Return standardized decomposition results

    Handlers are instantiated with repository-specific configurations and
    maintain their own component instances and tool connections.
    """

    def __init__(self, config: Any, debug: bool = False):
        """
        Initialize the handler with repository-specific configuration.

        Args:
            config: Handler-specific configuration object (e.g., CMRHandlerConfig)
            debug: Enable debug logging
        """
        self.config = config
        self.debug = debug

    @abstractmethod
    async def process_decomposition(
        self,
        decomposition: ScientificDecomposition,
        topic: Topic,
        original_query: str,
        params: DataSearchAgentInputSchema,
    ) -> DecompositionResult:
        """
        Process a scientific decomposition and return data results.

        This method implements the complete repository-specific pipeline:
        1. Extract known parameters from decomposition
        2. Generate searchable query variations
        3. Execute searches against repository API
        4. Filter and rank results
        5. Return standardized decomposition result

        Args:
            decomposition: Scientific decomposition to process
            topic: Parent topic providing context
            original_query: Original user research question
            params: Search parameters (temporal/spatial constraints, etc.)

        Returns:
            DecompositionResult with repository-specific data results

        Raises:
            NotImplementedError: If repository handler is not yet implemented
            RuntimeError: If processing fails after retries
        """
        pass
