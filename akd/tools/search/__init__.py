"""Search tools for the AKD framework."""

# Re-export SearchResultItem from structures for backward compatibility
from akd.structures import SearchResultItem

from ._base import (
    AgenticSearchTool,
    QueryFocusStrategy,
    SearchTool, 
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from .searxng_search import (
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
    SearxNGSearchToolInputSchema,
    SearxNGSearchToolOutputSchema,
)
from .semantic_scholar_search import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolConfig,
    SemanticScholarSearchToolInputSchema,
    SemanticScholarSearchToolOutputSchema,
)
from .agentic_search import (
    ControlledAgenticLitSearchTool,
    ControlledAgenticLitSearchToolConfig,
    DeepLitSearchTool,
    DeepLitSearchToolConfig,
)

__all__ = [
    # Re-exported structures
    "SearchResultItem",
    # Base classes
    "SearchTool",
    "AgenticSearchTool", 
    "SearchToolConfig",
    "SearchToolInputSchema",
    "SearchToolOutputSchema",
    "QueryFocusStrategy",
    # SearxNG
    "SearxNGSearchTool",
    "SearxNGSearchToolConfig", 
    "SearxNGSearchToolInputSchema",
    "SearxNGSearchToolOutputSchema",
    # Semantic Scholar
    "SemanticScholarSearchTool",
    "SemanticScholarSearchToolConfig",
    "SemanticScholarSearchToolInputSchema", 
    "SemanticScholarSearchToolOutputSchema",
    # Agentic Search
    "ControlledAgenticLitSearchTool",
    "ControlledAgenticLitSearchToolConfig",
    "DeepLitSearchTool",
    "DeepLitSearchToolConfig",
]