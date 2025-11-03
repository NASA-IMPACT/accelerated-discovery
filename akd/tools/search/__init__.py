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
from .code_search import (
    CodeSearchTool,
    CodeSearchToolConfig,
    CodeSearchToolInputSchema,
    CodeSearchToolOutputSchema,
    CompositeCodeSearchTool,
    CompositeCodeSearchToolConfig,
    GitHubCodeSearchTool,
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
)
from .composite import CompositeSearchTool, CompositeSearchToolConfig
from .pipeline import SearchPipeline, SearchPipelineConfig, SearchPipelineScrapingMode
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
    # Text Search Pipeline
    "SearchPipeline",
    "SearchPipelineConfig",
    "SearchPipelineScrapingMode",
    # Code Search
    "CodeSearchTool",
    "CodeSearchToolConfig",
    "CodeSearchToolInputSchema",
    "CodeSearchToolOutputSchema",
    "CompositeCodeSearchTool",
    "CompositeCodeSearchToolConfig",
    "LocalRepoCodeSearchTool",
    "LocalRepoCodeSearchToolConfig",
    "GitHubCodeSearchTool",
    "SDECodeSearchTool",
    "SDECodeSearchToolConfig",
]
