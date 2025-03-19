from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl

from .config import CONFIG


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
    data_url: Optional[HttpUrl] = Field(None, description="URL to download data")


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
