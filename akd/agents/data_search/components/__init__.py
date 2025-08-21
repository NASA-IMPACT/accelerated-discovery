"""Data search components for query processing and result synthesis."""

from .scientific_expansion import ScientificExpansionComponent
from .scientific_angles import ScientificAnglesComponent
from .cmr_query_generation import CMRQueryGenerationComponent

# Deprecated - kept for backward compatibility
from .query_decomposition import QueryDecompositionComponent

__all__ = [
    "ScientificExpansionComponent",
    "ScientificAnglesComponent",
    "CMRQueryGenerationComponent",
    # Deprecated
    "QueryDecompositionComponent",
]
