"""
Structured output schemas for data search components.

Defines Pydantic models for LLM-generated scientific angles and CMR search parameters.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ScientificAngle(BaseModel):
    """Individual scientific angle for data discovery."""

    title: str = Field(..., description="Concise title for this scientific angle")
    scientific_justification: str = Field(
        ...,
        description="Scientific reasoning explaining why this angle is relevant to the research question",
    )


class ScientificAnglesOutput(BaseModel):
    """Output from scientific angles generation component."""

    angles: List[ScientificAngle] = Field(
        ...,
        description="List of scientific angles for data discovery (2-6 angles)",
        min_items=2,
        max_items=6,
    )


class CMRCollectionSearchParams(BaseModel):
    """Parameters for CMR collection search that map directly to MCP interface."""

    keyword: Optional[str] = Field(
        None,
        description="Text search across collection metadata",
    )
    short_name: Optional[str] = Field(
        None,
        description="Collection short name (e.g., MOD09A1)",
    )
    version: Optional[str] = Field(
        None,
        description="Collection version (e.g., 6.1)",
    )
    provider: Optional[str] = Field(
        None,
        description="Data provider (e.g., LPDAAC_ECS, NSIDC_ECS)",
    )
    platform: Optional[str] = Field(
        None,
        description="Single platform/satellite name (e.g., Terra, Aqua)",
    )
    instrument: Optional[str] = Field(
        None,
        description="Single instrument name (e.g., MODIS, VIIRS)",
    )
    temporal: Optional[str] = Field(
        None,
        description="Temporal range in ISO format: YYYY-MM-DDTHH:mm:ssZ,YYYY-MM-DDTHH:mm:ssZ",
    )
    bounding_box: Optional[str] = Field(
        None,
        description="Spatial bounding box as comma-separated string: west,south,east,north",
    )


class CMRQueryOutput(BaseModel):
    """Output from CMR query generation component."""

    search_queries: List[CMRCollectionSearchParams] = Field(
        ...,
        description="List of CMR search parameter sets for this scientific angle",
        min_items=1,
        max_items=5,
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these specific search parameters were chosen",
    )
