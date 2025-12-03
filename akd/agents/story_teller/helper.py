from typing import List, Dict
import json
import os
from geopy.geocoders import Nominatim
from data_types import CollectionItem


class LocationCache:
    """Persistent cache for reverse geocoding results."""
    
    def __init__(self, cache_file: str = None, user_agent: str = "stac-processor"):
        """
        Initialize the location cache.
        
        Args:
            cache_file: Path to the cache file. Defaults to .location_cache.json in the same directory.
            user_agent: User agent string for Nominatim API.
        """
        self._cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), ".location_cache.json"
        )
        self._cache: Dict[str, str] = {}
        self._geolocator = Nominatim(user_agent=user_agent)
        self._load()
    
    def _load(self) -> None:
        """Load cache from file if it exists."""
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
    
    def _save(self) -> None:
        """Save cache to file."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError:
            pass
    
    def get(self, lat: float, lon: float) -> str:
        """
        Get location name from coordinates with caching.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Location name string or empty string if not found
        """
        cache_key = f"{lat},{lon}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            location = self._geolocator.reverse(f"{lat}, {lon}")
            location_name = location.address if location else ""
        except Exception:
            location_name = ""
        
        self._cache[cache_key] = location_name
        self._save()
        return location_name
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache = {}
        self._save()
    
    def __len__(self) -> int:
        """Return the number of cached locations."""
        return len(self._cache)
    
    def __contains__(self, key: tuple) -> bool:
        """Check if a location is in the cache."""
        lat, lon = key
        return f"{lat},{lon}" in self._cache


# Default instance for convenience
_location_cache = LocationCache()


def get_location_name(lat: float, lon: float) -> str:
    """Convenience function using the default cache instance."""
    return _location_cache.get(lat, lon)


def help_def() -> str:
  return "this is a help text"

def parse_stac_items_to_collection_items(stac_json: dict, stac_collection: dict = None) -> List[CollectionItem]:
    """
    Parse STAC FeatureCollection JSON and convert features to CollectionItem objects.
    
    Args:
        stac_json: A STAC FeatureCollection JSON containing features
        stac_collection: Optional STAC Collection JSON to extract collection metadata
        
    Returns:
        List of CollectionItem objects
    """
    collection_items = []
    
    # Extract collection metadata if provided
    collection_description = stac_collection.get("description", "") if stac_collection else ""
    collection_title = stac_collection.get("title", "") if stac_collection else ""
    
    for feature in stac_json.get("features", []):
        bbox = feature.get("bbox", [0, 0, 0, 0])
        center_lon = int((bbox[0] + bbox[2]) / 2)
        center_lat = int((bbox[1] + bbox[3]) / 2)
        
        # Get datetime from properties
        date = feature.get("properties", {}).get("datetime", "")
        
        # Get collection info
        collection_id = feature.get("collection", "")
        item_id = feature.get("id", "")
        
        # Get descriptions and titles from all assets
        assets = feature.get("assets", {})
        descriptions = [asset["description"] for asset in assets.values() if "description" in asset]
        titles = [asset["title"] for asset in assets.values() if "title" in asset]
        
        item_description = ", ".join(descriptions)
        item_title = ", ".join(titles)
        
        collection_item = CollectionItem(
            collection_id=collection_id,
            collection_description=collection_description,
            collection_title=collection_title,
            item_id=item_id,
            location=(center_lon, center_lat),
            date=date,
            location_name=get_location_name(center_lat, center_lon),
            item_description=item_description,
            item_title=item_title
        )
        collection_items.append(collection_item)
    
    return collection_items
