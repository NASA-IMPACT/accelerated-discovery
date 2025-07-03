# flake8: noqa: E501
"""
Refactored data structures and schemas for AKD project.

This module contains core data models, schemas, and type definitions
organized into logical sections for better maintainability.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, computed_field

from akd.configs.project import CONFIG

# Import guards for optional dependencies
try:
    import langchain_core  # noqa: F401

    LANGCHAIN_CORE_INSTALLED = True
except ImportError:
    LANGCHAIN_CORE_INSTALLED = False

if TYPE_CHECKING or LANGCHAIN_CORE_INSTALLED:
    try:
        from langchain_core.messages import BaseMessage
        from langchain_core.tools.structured import StructuredTool
    except ImportError:
        BaseMessage = BaseModel
        StructuredTool = None
else:
    BaseMessage = BaseModel
    StructuredTool = None

from .agents._base import BaseAgent
from .tools._base import BaseTool

# =============================================================================
# Search and Data Models
# =============================================================================


class SearchResultItem(BaseModel):
    """Represents a single search result item with metadata."""

    # Required fields
    url: HttpUrl = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    query: str = Field(..., description="The query used to obtain the search result")

    # Optional metadata
    pdf_url: Optional[HttpUrl] = Field(
        None,
        description="The PDF URL of the search paper",
    )
    content: Optional[str] = Field(
        None,
        description="The content snippet of the search result",
    )
    category: Optional[str] = Field(
        None,
        description="Category of the search result",
    )
    doi: Optional[str] = Field(
        None,
        description="Digital Object Identifier (DOI) of the search result",
    )
    published_date: Optional[str] = Field(
        None,
        description="Publication date for the search result",
    )
    engine: Optional[str] = Field(
        None,
        description="Engine that fetched the search result",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for the search result",
    )
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Extra information from the search result",
    )

    @computed_field
    @property
    def title_augmented(self) -> str:
        """Returns title with publication date if available."""
        if self.published_date:
            return f"{self.title} - (Published {self.published_date})"
        return self.title


class ResearchData(BaseModel):
    """
    Represents the dataset used in scientific research.

    Captures key metadata about data sources including format, origin,
    and accessibility for better reproducibility and documentation.
    """

    data_format: str = Field(
        ...,
        description="Type of data used (e.g: HDF5/CSV/JSON) in the research",
    )
    origin: str = Field(
        ...,
        description="Mission/Instrument/Model the data is derived from (e.g., HLS, MERRA-2)",
    )
    data_url: Optional[HttpUrl] = Field(
        None,
        description="Valid URL to download data referenced in research. Leave None if unavailable.",
    )


# =============================================================================
# Extraction Schemas
# =============================================================================


class ExtractionSchema(BaseModel):
    """Base schema for information extraction tasks."""

    answer: str = Field(
        CONFIG.model_config_settings.default_no_answer,
        description="Direct, concise answer to the input query",
    )
    related_knowledge: Optional[List[str]] = Field(
        None,
        description="List of concise related information supporting the query answer",
    )


class SingleEstimation(ExtractionSchema):
    """
    Represents an estimation extracted from research literature.

    Used for extracting specific values, parameters, or results based on
    scientific data and methodologies. Captures estimation process details
    including methodology, assumptions, and validation.
    """

    research_data: ResearchData = Field(
        ...,
        description="Data used for the estimation in the research",
    )
    methodology: str = Field(
        ...,
        description="Methodology used for the estimation",
    )
    assumptions: Optional[List[str]] = Field(
        None,
        description="Key assumptions made during the estimation process",
    )
    confidence_level: Optional[float] = Field(
        None,
        description="Confidence level of the estimation (e.g., probability or margin of error)",
    )
    validation_method: Optional[str] = Field(
        None,
        description="How the estimation was validated or cross-checked",
    )


class ExtractionDTO(BaseModel):
    """Data Transfer Object for extraction results."""

    source: str = Field(..., description="Source of the extraction")
    result: Any = Field(..., description="Extracted result data")


# =============================================================================
# Tool System Models
# =============================================================================


class ToolSearchResult(BaseModel):
    """Represents the result of a tool search operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Optional["Tool"] = Field(
        None,
        description="Tool found during search",
    )
    args: Optional[Dict[str, Any]] = Field(
        None,
        description="Input arguments extracted when tool is found",
    )
    result: Optional[Any] = Field(
        None,
        description="Result when tool is executed",
    )

    @property
    def name(self) -> str:
        """Returns the name of the tool or its class name."""
        if self.tool is None:
            return "Unknown"
        return getattr(self.tool, "name", self.tool.__class__.__name__)


# =============================================================================
# Type Definitions
# =============================================================================

# Tool types - conditional based on langchain availability
if TYPE_CHECKING:
    # For type checking, include all possible types
    from langchain_core.tools.structured import StructuredTool as _StructuredTool

    Tool: TypeAlias = Union[BaseTool, BaseAgent, _StructuredTool]
else:
    # At runtime, check what's actually available
    if LANGCHAIN_CORE_INSTALLED and StructuredTool is not None:
        Tool: TypeAlias = Union[BaseTool, BaseAgent, "StructuredTool"]  # type: ignore
    else:
        Tool: TypeAlias = Union[BaseTool, BaseAgent]

# Guardrail types
GuardrailType: TypeAlias = Union[BaseTool, Callable, Coroutine]

# Callable specifications
AnyCallable: TypeAlias = Union[BaseTool, BaseAgent, Callable[..., Any]]
CallableSpec: TypeAlias = Union[AnyCallable, Tuple[AnyCallable, Dict[str, str]]]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Search and Data Models
    "SearchResultItem",
    "ResearchData",
    # Extraction Schemas
    "ExtractionSchema",
    "SingleEstimation",
    "ExtractionDTO",
    # Tool Models
    "ToolSearchResult",
    # Type Definitions
    "Tool",
    "GuardrailType",
    "AnyCallable",
    "CallableSpec",
]
