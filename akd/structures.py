# flake8: noqa: E501
"""
Refactored data structures and schemas for AKD project.

This module contains core data models, schemas, and type definitions
organized into logical sections for better maintainability.
"""

from typing import Any

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from akd._base import IOSchema

# from akd.common_types import ToolType
from akd.configs.project import CONFIG

# =============================================================================
# Search and Data Models
# =============================================================================


class SearchResult(IOSchema):
    """Base class for all search result types across different search domains.

    Provides common interface for search results including query context,
    title, content/description, and relevance scoring. Subclasses should
    add domain-specific fields (e.g., url, doi for literature; decompositions
    for hierarchical data search).
    """

    query: str = Field(..., description="Query that produced this search result")
    title: str = Field(..., description="Title or name of the search result")
    content: str = Field(default="", description="Content snippet, description, or summary")
    score: float | None = Field(None, description="Relevance or ranking score")
    extra: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @computed_field
    def relevancy_score(self) -> float | None:
        """Alias for score field for consistency."""
        return self.score

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        """Convert None to empty string for content field."""
        return v if v is not None else ""

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v):
        """Convert numpy float types to Python float for JSON serialization."""
        if v is None:
            return None
        # Handle numpy float types (float32, float64, etc.)
        if hasattr(v, "item"):  # numpy scalar types have .item() method
            return float(v.item())
        return float(v)


class SearchResultItem(SearchResult):
    """Represents a single search result item with metadata."""

    # Required fields
    url: AnyUrl = Field(..., description="The URL of the search result")

    # Optional metadata
    pdf_url: AnyUrl | None = Field(
        None,
        description="The PDF URL of the search paper",
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
    data_url: AnyUrl | None = Field(
        None,
        description="Valid URL to download data referenced in research. Leave None if unavailable.",
    )


class PaperDataItem(BaseModel):
    """Represents a single paper data object retrieved from Semantic Scholar."""

    paper_id: str | None = Field(
        ...,
        description="Semantic Scholar’s primary unique identifier for a paper.",
    )
    corpus_id: int | None = Field(
        ...,
        description="Semantic Scholar’s secondary unique identifier for a paper.",
    )
    external_ids: object | None = Field(
        None,
        description="Valid URL to download data referenced in research. Leave None if unavailable.",
    )
    url: str | None = Field(
        ...,
        description="URL of the paper on the Semantic Scholar website.",
    )
    title: str | None = Field(
        ...,
        description="Title of the paper.",
    )
    abstract: str | None = Field(
        ...,
        description="The paper's abstract. Note that due to legal reasons, this may be missing even if we display an abstract on the website.",
    )
    venue: str | None = Field(
        ...,
        description="The name of the paper’s publication venue.",
    )
    publication_venue: object | None = Field(
        ...,
        description="An object that contains the following information about the journal or conference in which this paper was published: id (the venue’s unique ID), name (the venue’s name), type (the type of venue), alternate_names (an array of alternate names for the venue), and url (the venue’s website).",
    )
    year: int | None = Field(
        ...,
        description="The year the paper was published.",
    )
    reference_count: int | None = Field(
        ...,
        description="The total number of papers this paper references.",
    )
    citation_count: int | None = Field(
        ...,
        description="The total number of papers that references this paper.",
    )
    influential_citation_count: int | None = Field(
        ...,
        description="A subset of the citation count, where the cited publication has a significant impact on the citing publication.",
    )
    is_open_access: bool | None = Field(
        ...,
        description="Whether the paper is open access.",
    )
    open_access_pdf: object | None = Field(
        ...,
        description="An object that contains the following parameters: url (a link to the paper’s PDF), status, the paper's license, and a legal disclaimer.",
    )
    fields_of_study: list[str] | None = Field(
        ...,
        description="A list of the paper’s high-level academic categories from external sources.",
    )
    s2_fields_of_study: list[object] | None = Field(
        ...,
        description="An array of objects. Each object contains the following parameters: category (a field of study. The possible fields are the same as in fieldsOfStudy), and source (specifies whether the category was classified by Semantic Scholar or by an external source.",
    )
    publication_types: list[str] | None = Field(
        ...,
        description="The type of this publication.",
    )
    publication_date: str | None = Field(
        ...,
        description="The date when this paper was published, in YYYY-MM-DD format.",
    )
    journal: object | None = Field(
        ...,
        description="An object that contains the following parameters, if available: name (the journal name), volume (the journal’s volume number), and pages (the page number range)",
    )
    citation_styles: object | None = Field(
        ...,
        description="The BibTex bibliographical citation of the paper.",
    )
    authors: list[object] | None = Field(
        ...,
        description="List of authors corresponding to the paper.",
    )
    citations: list[object] | None = Field(
        ...,
        description="List of citations the paper has.",
    )
    references: list[object] | None = Field(
        ...,
        description="List of references used in the paper.",
    )
    embedding: object | None = Field(
        ...,
        description="The paper's embedding.",
    )
    tldr: object | None = Field(
        ...,
        description="Tldr version of the paper.",
    )
    external_id: str | None = Field(
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
    related_knowledge: list[str] | None = Field(
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
    assumptions: list[str] | None = Field(
        None,
        description="Key assumptions made during the estimation process",
    )
    confidence_level: float | None = Field(
        None,
        description="Confidence level of the estimation (e.g., probability or margin of error)",
    )
    validation_method: str | None = Field(
        None,
        description="How the estimation was validated or cross-checked",
    )


# =============================================================================
# Tool System Models
# =============================================================================


class ToolSearchResult(BaseModel):
    """Represents the result of a tool search operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # causes serialization issues as arbitrary type
    tool: Any | None = Field(
        None,
        description="Tool found during search",
    )  # Should be ToolType, but circular import issues
    # tool: Optional["ToolType"] = Field(...)
    args: dict[str, Any] | None = Field(
        None,
        description="Input arguments extracted when tool is found",
    )
    result: Any | None = Field(
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

# Type alias for semantic clarity in literature search contexts
LitSearchResult = SearchResultItem

__all__ = [
    # Search and Data Models
    "SearchResult",
    "SearchResultItem",
    "LitSearchResult",
    "ResearchData",
    # Extraction Schemas
    "ExtractionSchema",
    "SingleEstimation",
    # Tool Models
    "ToolSearchResult",
]
