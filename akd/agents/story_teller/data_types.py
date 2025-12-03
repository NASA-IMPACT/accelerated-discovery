from pydantic import Field, BaseModel
from typing import List

class CollectionItem(BaseModel):
  collection_id: str = Field(..., description="The collection which the item belongs to.")
  collection_description: str = Field(..., description="The description of the STAC collection which the item belongs to.")
  collection_title: str = Field(..., description="The title of the STAC collection which the item belongs to.")
  item_id: str = Field(..., description="The item id which represents a collection asset.")
  location: List[int] = Field(..., description="The longitue and latitude of the asset.") # TODO: later use tuple. OpenAI api limitation
  date: str = Field(..., description="date time corresponding to when the item was captured using some instrument.")
  location_name: str = Field(..., description="The name of the location to where the item assets belongs to.")
  item_description: str = Field(..., description="The description of the STAC item. It is the combined assets description.")
  item_title: str = Field(..., description="The title of the STAC item. It is the combined assets title.")

class SearchedSTACData(BaseModel):
  collections: List[CollectionItem] = Field(default=[], description="The list of relevant collections received from Data Search Agent.") # TODO: later use tuple. OpenAI api limitation

class ComparableItem(BaseModel):
  items: List[CollectionItem] = Field(..., description="This is the pair of items that can be temporally compared across multiple days")
  reasoning: str = Field(..., description="The reason why this items are comparable")

class RelatedItem(BaseModel):
  items: List[CollectionItem] = Field(..., description="The STAC collection items that are related")
  reasoning: str = Field(..., description="The reason behind why hte items are related.")

