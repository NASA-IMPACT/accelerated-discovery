# flake8: noqa: E501
"""
Refactored data structures and schemas for AKD project.

This module contains core data models, schemas, and type definitions
organized into logical sections for better maintainability.
"""

from typing import Any, Dict, List, Optional

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, computed_field

from akd._base import IOSchema

# from akd.common_types import ToolType
from akd.configs.project import CONFIG

# Import guards for optional dependencies
try:
    import langchain_core  # noqa: F401

    LANGCHAIN_CORE_INSTALLED = True
except ImportError:
    LANGCHAIN_CORE_INSTALLED = False


# =============================================================================
# Search and Data Models
# =============================================================================


class SearchResultItem(IOSchema):
    """Represents a single search result item with metadata."""

    # Required fields
    url: AnyUrl = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    query: str = Field(..., description="The query used to obtain the search result")

    # Optional metadata
    pdf_url: AnyUrl | None = Field(
        None,
        description="The PDF URL of the search paper",
    )
    content: str | None = Field(
        None,
        description="The content snippet of the search result",
    )
    category: str | None = Field(
        None,
        description="Category of the search result",
    )
    doi: str | None = Field(
        None,
        description="Digital Object Identifier (DOI) of the search result",
    )
    published_date: str | None = Field(
        None,
        description="Publication date for the search result",
    )
    engine: str | None = Field(
        None,
        description="Engine that fetched the search result",
    )
    tags: list[str] | None = Field(
        None,
        description="Tags for the search result",
    )

    authors: list[str] | None = Field(
        None,
        description="List of authors for DOI resolution by title and author",
    )
    extra: dict[str, Any] | None = Field(
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
    data_url: Optional[AnyUrl] = Field(
        None,
        description="Valid URL to download data referenced in research. Leave None if unavailable.",
    )


class PaperDataItem(BaseModel):
    """Represents a single paper data object retrieved from Semantic Scholar."""

    paper_id: Optional[str] = Field(
        ...,
        description="Semantic Scholar’s primary unique identifier for a paper.",
    )
    corpus_id: Optional[int] = Field(
        ...,
        description="Semantic Scholar’s secondary unique identifier for a paper.",
    )
    external_ids: Optional[object] = Field(
        None,
        description="Valid URL to download data referenced in research. Leave None if unavailable.",
    )
    url: Optional[str] = Field(
        ...,
        description="URL of the paper on the Semantic Scholar website.",
    )
    title: Optional[str] = Field(
        ...,
        description="Title of the paper.",
    )
    abstract: Optional[str] = Field(
        ...,
        description="The paper's abstract. Note that due to legal reasons, this may be missing even if we display an abstract on the website.",
    )
    venue: Optional[str] = Field(
        ...,
        description="The name of the paper’s publication venue.",
    )
    publication_venue: Optional[object] = Field(
        ...,
        description="An object that contains the following information about the journal or conference in which this paper was published: id (the venue’s unique ID), name (the venue’s name), type (the type of venue), alternate_names (an array of alternate names for the venue), and url (the venue’s website).",
    )
    year: Optional[int] = Field(
        ...,
        description="The year the paper was published.",
    )
    reference_count: Optional[int] = Field(
        ...,
        description="The total number of papers this paper references.",
    )
    citation_count: Optional[int] = Field(
        ...,
        description="The total number of papers that references this paper.",
    )
    influential_citation_count: Optional[int] = Field(
        ...,
        description="A subset of the citation count, where the cited publication has a significant impact on the citing publication.",
    )
    is_open_access: Optional[bool] = Field(
        ...,
        description="Whether the paper is open access.",
    )
    open_access_pdf: Optional[object] = Field(
        ...,
        description="An object that contains the following parameters: url (a link to the paper’s PDF), status, the paper's license, and a legal disclaimer.",
    )
    fields_of_study: Optional[list[str]] = Field(
        ...,
        description="A list of the paper’s high-level academic categories from external sources.",
    )
    s2_fields_of_study: Optional[list[object]] = Field(
        ...,
        description="An array of objects. Each object contains the following parameters: category (a field of study. The possible fields are the same as in fieldsOfStudy), and source (specifies whether the category was classified by Semantic Scholar or by an external source.",
    )
    publication_types: Optional[list[str]] = Field(
        ...,
        description="The type of this publication.",
    )
    publication_date: Optional[str] = Field(
        ...,
        description="The date when this paper was published, in YYYY-MM-DD format.",
    )
    journal: Optional[object] = Field(
        ...,
        description="An object that contains the following parameters, if available: name (the journal name), volume (the journal’s volume number), and pages (the page number range)",
    )
    citation_styles: Optional[object] = Field(
        ...,
        description="The BibTex bibliographical citation of the paper.",
    )
    authors: Optional[list[object]] = Field(
        ...,
        description="List of authors corresponding to the paper.",
    )
    citations: Optional[list[object]] = Field(
        ...,
        description="List of citations the paper has.",
    )
    references: Optional[list[object]] = Field(
        ...,
        description="List of references used in the paper.",
    )
    embedding: Optional[object] = Field(
        ...,
        description="The paper's embedding.",
    )
    tldr: Optional[object] = Field(
        ...,
        description="Tldr version of the paper.",
    )
    external_id: Optional[str] = Field(
        ...,
        description="The external id of the paper from the query.",
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

    # causes serialization issues as arbitrary type
    tool: Optional[Any] = Field(
        None,
        description="Tool found during search",
    )  # Should be ToolType, but circular import issues
    # tool: Optional["ToolType"] = Field(...)
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
]
