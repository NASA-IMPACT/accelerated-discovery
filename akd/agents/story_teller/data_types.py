from pydantic import Field, BaseModel
from typing import Optional, List, Tuple

class STACItem(BaseModel):
  id: str = Field(..., description="The STAC Item Identifier.")
  description: str = Field(..., description="The description of the item.")
  temporal_resolution: Optional[Tuple[str, str]] = Field(default=None, description="The start datetime and end datetime of the item.")
  spatial_resolution: Optional[str] = Field(default=None, description="The location where this item belongs to. Name of the location using geodini.")

class STACCollection(BaseModel):
  id: str = Field(..., description="The STAC Collection Identifier.")
  description: str = Field(..., description="The description of the collection.")
  items: List[STACItem] = Field(default=[], description="The list of relevant STAC Items for the collection.")
  temporal_resolution: Optional[Tuple[str, str]] = Field(default=None, description="The start datetime and end datetime of the collection.")
  spatial_resolution: Optional[str] = Field(default=None, description="The location where this collection belongs to. Name of the location using geodini.")

class SearchedSTACData(BaseModel):
  collections: List[STACCollection] = Field(default=[], description="The list of relevant collections received from Data Search Agent.")
