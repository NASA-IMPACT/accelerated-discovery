from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl

from .config import CONFIG


class SearchResultItem(BaseModel):
    """This schema represents a single search result item"""

    url: HttpUrl = Field(..., description="The URL of the search result")
    pdf_url: Optional[HttpUrl] = Field(
        None,
        description="The PDF URL of the search paper",
    )
    title: str = Field(..., description="The title of the search result")
    content: Optional[str] = Field(
        None,
        description="The content snippet of the search result",
    )
    query: str = Field(..., description="The query used to obtain the search result")
    category: Optional[str] = Field(None, description="Category of the search result")
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
    tags: Optional[List[str]] = Field(None, description="Tags for the search result")
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Extra information from the search result",
    )


class ExtractionSchema(BaseModel):
    """
    Base schema for information extraction
    """

    answer: str = Field(
        CONFIG.model_config_.default_no_answer,
        description="Direct, concise answer to the input query",
    )
    related_knowledge: List[str] = Field(
        None,
        description="List of related information that can support "
        "answering the query. Should be very concise.",
    )


class ResearchData(BaseModel):
    """
    Represents the dataset used in scientific research.

    This schema captures key metadata about the data sources utilized in research,
    including their format, origin, and accessibility. It ensures structured
    documentation of data provenance, allowing for better reproducibility
    and understanding of the research findings.
    """

    data_format: str = Field(
        ...,
        description="Type of data used (e.g: HDF5/CSV/JSON or other) in the research",
    )
    origin: str = Field(
        ...,
        description="Mission/Instrument/Model the data is derived from "
        "(e.g., HLS, MERRA-2) in the research",
    )
    # source: str = Field(..., description="Source for the literature research")
    data_url: Optional[HttpUrl] = Field(
        None,
        description="URL to download data that the research references/uses. If not available, leave empty/None",
    )


class SingleEstimation(ExtractionSchema):
    """
    Represents an estimation extracted from research literature.
    This schema is used when the extraction involves estimating a specific
    value, parameter, or result based on scientific data and methodologies.
    It captures essential details about the estimation process, including
    the research data used, the methodology applied, and any relevant
    assumptions or validation steps.
    """

    research_data: ResearchData = Field(
        ...,
        description="Data being used for the estimation in the research",
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
        description="Confidence level of the estimation, if available "
        "(e.g., probability or margin of error)",
    )
    validation_method: Optional[str] = Field(
        None,
        description="How the estimation was validated or cross-checked",
    )


class ExtractionDTO(BaseModel):
    source: str
    result: Any
