"""Data search components for query processing and result synthesis."""

# Base component
from ._base import BaseDataSearchComponent

# Base classes for parameters and ranking
from ._base_parameters import (
    BaseKnownParametersComponent,
    BaseSearchableParametersComponent,
)
from ._base_ranking import BaseApproachFilteringComponent, BaseFinalRankingComponent

# Shared components (for creating repository-specific implementations)
from ._shared_parameters import (
    SharedKnownParametersComponent,
    SharedSearchableParametersComponent,
)
from ._shared_ranking import (
    SharedApproachFilteringComponent,
    SharedFinalRankingComponent,
)

# Universal workflow components (used for all repositories)
from .repository_router import RepositoryRouterComponent
from .scientific_decomposition import (
    ScientificDecomposition,
    ScientificDecompositionComponent,
)
from .topic_splitting import Topic, TopicSplittingComponent

__all__ = [
    # Base components
    "BaseDataSearchComponent",
    "BaseKnownParametersComponent",
    "BaseSearchableParametersComponent",
    "BaseApproachFilteringComponent",
    "BaseFinalRankingComponent",
    # Shared components
    "SharedKnownParametersComponent",
    "SharedSearchableParametersComponent",
    "SharedApproachFilteringComponent",
    "SharedFinalRankingComponent",
    # Universal workflow components
    "TopicSplittingComponent",
    "Topic",
    "RepositoryRouterComponent",
    "ScientificDecompositionComponent",
    "ScientificDecomposition",
]
