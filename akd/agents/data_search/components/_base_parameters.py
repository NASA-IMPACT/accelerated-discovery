"""
Base classes for parameter extraction components.

Defines abstract interfaces for repository-specific parameter extraction:
- Known parameters: Hard filters directly identifiable from research context
- Searchable parameters: Search terms/keywords for dataset discovery
"""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from akd._base import InputSchema

from .scientific_decomposition import ScientificDecomposition
from .topic_splitting import Topic

# Type variables for generic base classes
TQueryApproach = TypeVar("TQueryApproach", bound=BaseModel)
TSearchableQuery = TypeVar("TSearchableQuery", bound=BaseModel)


class BaseKnownParametersOutput(BaseModel, Generic[TQueryApproach]):
    """Base output schema for known parameters extraction."""

    query_approaches: List[TQueryApproach]
    reasoning: str


class BaseKnownParametersInputSchema(InputSchema):
    """Base input schema for known parameters extraction."""

    original_query: str
    topic: Topic
    decomposition: ScientificDecomposition


class BaseKnownParametersComponent(ABC, Generic[TQueryApproach]):
    """
    Abstract base class for known parameters extraction.

    Repository-specific implementations (CMR, PDS4, etc.) inherit from this
    to extract hard filters that can be directly identified from scientific context.

    Each repository defines its own QueryApproach schema with repository-specific
    parameter fields (e.g., CMR has instrument/platform, PDS4 has target/mission).
    """

    @abstractmethod
    async def process(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
    ) -> BaseKnownParametersOutput[TQueryApproach]:
        """
        Extract known parameters from research context.

        Args:
            original_query: Original research question for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification

        Returns:
            Output with repository-specific query approaches
        """
        pass


class BaseSearchableParametersOutput(BaseModel, Generic[TSearchableQuery]):
    """Base output schema for searchable parameters generation."""

    searchable_queries: List[TSearchableQuery]
    keyword_strategy: str


class BaseSearchableParametersComponent(ABC, Generic[TQueryApproach, TSearchableQuery]):
    """
    Abstract base class for searchable parameters generation.

    Repository-specific implementations inherit from this to generate
    search terms/keywords that combine with known parameters for dataset discovery.

    Each repository defines:
    - TQueryApproach: Known parameter structure for that repository
    - TSearchableQuery: Complete query with known + searchable parameters
    """

    @abstractmethod
    async def process(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
        query_approaches: List[TQueryApproach],
    ) -> BaseSearchableParametersOutput[TSearchableQuery]:
        """
        Generate searchable parameters for query approaches.

        Args:
            original_query: Original research question for context
            topic: Topic being processed
            decomposition: Scientific decomposition with justification
            query_approaches: Known parameter approaches to enhance

        Returns:
            Output with complete searchable queries
        """
        pass
