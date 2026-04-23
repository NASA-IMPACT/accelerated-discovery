"""Search tools for the AKD framework."""

import importlib
from typing import TYPE_CHECKING

# Re-export SearchResultItem from structures for backward compatibility
from akd.structures import SearchResultItem

from ._base import (
    QueryFocusStrategy,
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from .composite import CompositeSearchTool, CompositeSearchToolConfig
from .searxng import (
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
    SearxNGSearchToolInputSchema,
    SearxNGSearchToolOutputSchema,
)
from .semantic_scholar import (
    SemanticScholarSearchTool,
    SemanticScholarSearchToolConfig,
    SemanticScholarSearchToolInputSchema,
    SemanticScholarSearchToolOutputSchema,
)
from .serper import (
    SerperSearchTool,
    SerperSearchToolConfig,
    SerperSearchToolInputSchema,
    SerperSearchToolOutputSchema,
)

# Lazy imports for heavy dependencies (pipeline depends on scrapers.omni -> docling)
if TYPE_CHECKING:
    from .pipeline import (
        SearchPipeline,
        SearchPipelineConfig,
        SearchPipelineScrapingMode,
    )

_LAZY_IMPORTS = {
    "SearchPipeline": ".pipeline",
    "SearchPipelineConfig": ".pipeline",
    "SearchPipelineScrapingMode": ".pipeline",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Re-exported structures
    "SearchResultItem",
    # Base classes
    "SearchTool",
    "SearchToolConfig",
    "SearchToolInputSchema",
    "SearchToolOutputSchema",
    "QueryFocusStrategy",
    # Composite
    "CompositeSearchTool",
    "CompositeSearchToolConfig",
    # SearxNG
    "SearxNGSearchTool",
    "SearxNGSearchToolInputSchema",
    "SearxNGSearchToolOutputSchema",
    "SearxNGSearchToolConfig",
    # Serper
    "SerperSearchTool",
    "SerperSearchToolInputSchema",
    "SerperSearchToolOutputSchema",
    "SerperSearchToolConfig",
    # Semantic Scholar
    "SemanticScholarSearchTool",
    "SemanticScholarSearchToolInputSchema",
    "SemanticScholarSearchToolOutputSchema",
    "SemanticScholarSearchToolConfig",
    # Text Search Pipeline (lazy loaded)
    "SearchPipeline",
    "SearchPipelineConfig",
    "SearchPipelineScrapingMode",
]
