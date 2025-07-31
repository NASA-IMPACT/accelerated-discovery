"""Search tools for the AKD framework."""

# Re-export SearchResultItem from structures for backward compatibility
from akd.structures import SearchResultItem

from ._base import (
    QueryFocusStrategy,
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from .searxng_search import (
    SearxNGSearchTool,
    SearxNGSearchToolInputSchema,
    SearxNGSearchToolOutputSchema,
)
from .semantic_scholar_search import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolConfig,
    SemanticScholarSearchToolInputSchema,
    SemanticScholarSearchToolOutputSchema,
)

__all__ = [
    # Re-exported structures
    "SearchResultItem",
    # Base classes
    "SearchTool",
    "SearchToolConfig",
    "SearchToolInputSchema",
    "SearchToolOutputSchema",
    "QueryFocusStrategy",
    # SearxNG
    "SearxNGSearchTool",
    "SearxNGSearchToolInputSchema",
    "SearxNGSearchToolOutputSchema",
    # Semantic Scholar
    "SemanticScholarSearchTool",
    "SemanticScholarSearchToolConfig",
    "SemanticScholarSearchToolInputSchema",
    "SemanticScholarSearchToolOutputSchema",
]
