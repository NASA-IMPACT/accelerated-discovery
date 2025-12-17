from typing import List, Dict
import json
import os
from geopy.geocoders import Nominatim
from data_types import CollectionItem
from scraper import get_default_scraper
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

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

def parse_stac_items_to_collection_items(stac_items: List[dict], stac_collection: dict = None) -> List[CollectionItem]:
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
    
    for feature in stac_items:
        bbox = feature.get("bbox", [0, 0, 0, 0])
        center_lon = int((bbox[0] + bbox[2]) / 2)
        center_lat = int((bbox[1] + bbox[3]) / 2)
        
        # Get datetime from properties and format to yyyy-mm-dd
        date_str = feature.get("properties", {}).get("datetime", "") or feature.get("properties", {}).get("start_datetime", "")
        date = ""
        if date_str:
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                date = parsed_date.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                date = date_str
        
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

def download_stac_data(stac_url: str, stac_collection_id: str, output_dir: str = "./data") -> tuple[str, str, dict, list]:
    """
    Download STAC collection JSON and all its items.
    
    Args:
        stac_url: Base URL of the STAC API
        stac_collection_id: ID of the collection to download
        output_dir: Directory to save the JSON files (default: ./data)
    
    Returns:
        tuple: (collection_filepath, items_filepath, collection_data, items_list)
    """
    import json
    from pathlib import Path
    import requests
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Construct collection URL
    collection_url = f"{stac_url.rstrip('/')}/collections/{stac_collection_id}"
    
    # Download collection JSON
    print(f"Downloading STAC collection from: {collection_url}")
    response = requests.get(collection_url)
    response.raise_for_status()
    collection_data = response.json()
    
    # Save collection JSON
    collection_file = output_path / "stac_collection.json"
    with open(collection_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    print(f"Saved collection to: {collection_file}")
    
    # Download all items
    items = []
    links = collection_data.get('links', [])
    items_url = None
    
    # Look for items link
    for link in links:
        if link.get('rel') == 'items':
            items_url = link.get('href')
            break
    
    if items_url:
        print(f"Downloading items from: {items_url}")
        
        # Handle pagination
        while items_url:
            response = requests.get(items_url)
            response.raise_for_status()
            items_data = response.json()
            
            if 'features' in items_data:
                items.extend(items_data['features'])
                print(f"  Downloaded {len(items_data['features'])} items (total: {len(items)})")
            
            # Check for next page
            items_url = None
            for link in items_data.get('links', []):
                if link.get('rel') == 'next':
                    items_url = link.get('href')
                    break
    else:
        if 'features' in collection_data:
            items = collection_data['features']
            print(f"Found {len(items)} inline items")
    
    # Save items JSON
    items_file = output_path / "stac_items.json"
    with open(items_file, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": items}, f, indent=2)
    print(f"Saved {len(items)} items to: {items_file}")
    
    return str(collection_file), str(items_file), collection_data, items

class MDXValidator:
    """Validates MDX (Markdown with JSX) tag structure.

    This validator checks that all opening and closing tags are properly paired
    in MDX content. It tracks tag positions to help identify mismatched tags.
    Self-closing tags (e.g., <Component />) are handled correctly.
    """

    def __init__(self):
        """Initialize the MDX validator with an empty stack."""
        self.stack: list[tuple[str, int]] = [] # (tag, position)

    def validate_mdx(self, mdx: str) -> tuple[bool, list[tuple[str, int]]]:
        """Validate MDX content for properly paired tags.

        Args:
            mdx: The MDX string to validate.

        Returns:
            A tuple containing:
            - bool: True if the MDX is valid (all tags properly paired), False otherwise.
            - list[tuple[str, int]]: List of unpaired tags with their positions. Empty if valid.
              Each tuple contains (tag_string, position_index) where position is the
              enumerated index of the tag in the sequence.

        Examples:
            >>> validator = MDXValidator()
            >>> validator.validate_mdx("<Block><Prose>text</Prose></Block>")
            (True, [])
            >>> validator.validate_mdx("<Block><Prose>text</Block>")
            (False, [('<Block>', 0), ('<Prose>', 1), ('</Block>', 2)])
        """
        mdx_blocks = self._get_mdx_blocks(mdx)
        first_tag = next(mdx_blocks, None)
        if not first_tag: # a valid mdx can have no tags
            return True, []
        self.stack.append((first_tag, 0))
        for idx, block in enumerate(mdx_blocks, start=1):
            if self._self_closing_tag(self._sanitize_tag(block)):
                continue
            elif len(self.stack) == 0:
                self.stack.append((block, idx))
            elif self._check_pair(self._sanitize_tag(self.stack[-1][0]), self._sanitize_tag(block)):
                self.stack.pop()
            else:
                self.stack.append((block, idx))
        return not self.stack, [(tag, pos) for (tag, pos) in self.stack]

    def _check_pair(self, tag1: str, tag2: str) -> bool:
        """Check if two tags form a valid opening/closing pair.

        Args:
            tag1: First tag string (e.g., '<Block>' or '</Block>').
            tag2: Second tag string (e.g., '<Block>' or '</Block>').

        Returns:
            True if the tags are a matching opening/closing pair, False otherwise.
            Self-closing tags always return False as they don't pair with other tags.

        Examples:
            >>> validator._check_pair('<Block>', '</Block>')
            True
            >>> validator._check_pair('<Block>', '</Prose>')
            False
            >>> validator._check_pair('<Block />', '</Block>')
            False
        """
        if (self._self_closing_tag(tag1) or self._self_closing_tag(tag2)):
            return False

        open_tag = tag1
        close_tag = tag2

        if (open_tag[-1] == ">" and open_tag[-2] == "/"):
            close_tag, open_tag = open_tag, close_tag

        open_tag_name = open_tag[1:-1]
        close_tag_name = close_tag[2:-1]

        return open_tag_name == close_tag_name

    def _self_closing_tag(self, tag: str) -> bool:
        """Check if a tag is self-closing.

        Args:
            tag: The tag string to check.

        Returns:
            True if the tag is self-closing (ends with '/>'), False otherwise.

        Examples:
            >>> validator._self_closing_tag('<Component />')
            True
            >>> validator._self_closing_tag('<Component>')
            False
        """
        return tag.startswith("<") and tag[-2:] == "/>"

    def _sanitize_tag(self, tag: str) -> str:
        """Remove attributes and whitespace from a tag, keeping only the tag name.

        This method strips out any attributes, props, or whitespace from a tag,
        leaving only the tag name and appropriate closing characters.

        Args:
            tag: The tag string to sanitize (e.g., '<Block className="foo">').

        Returns:
            The sanitized tag with only the tag name (e.g., '<Block>').
            Preserves self-closing syntax if present.

        Examples:
            >>> validator._sanitize_tag('<Block className="foo">')
            '<Block>'
            >>> validator._sanitize_tag('<Component prop="value" />')
            '<Component />'
        """
        tag_split: list[str] = tag.split(" ")
        if len(tag_split) < 2:
            return tag
        tag_name: str = tag_split[0].split("\n")[0]
        tag_end: str = ">"
        if tag[-2] == "/":
            tag_end = "/>"
        return tag_name+tag_end

    def _get_mdx_blocks(self, mdx: str):
        """Generator that yields all MDX tags from the input string.

        Parses the MDX string and yields each tag (opening, closing, or self-closing)
        in the order they appear. This is a generator function for memory efficiency.

        Args:
            mdx: The MDX string to parse.

        Yields:
            str: Each tag found in the MDX content (e.g., '<Block>', '</Block>', '<Component />').

        Examples:
            >>> list(validator._get_mdx_blocks('<Block><Prose>text</Prose></Block>'))
            ['<Block>', '<Prose>', '</Prose>', '</Block>']
        """
        start: int = -1
        for idx, c in enumerate(mdx):
            if c == "<":
                start = idx
            elif c == ">" and start != -1:
                yield mdx[start:idx+1]
                start = -1

def get_collection_items(stac_collection_ids: list[str], stac_url:str="https://earth.gov/ghgcenter/api/stac") -> List[CollectionItem]:
    """Get collection items from the collection file."""
    all_collection_items: List[CollectionItem] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(get_collection_item, cid, stac_url)
            for cid in stac_collection_ids
        ]
        # collect the results as they complete
        for future in as_completed(futures):
            try:
                collection_items: List[CollectionItem] = future.result()
                all_collection_items.extend(collection_items)
            except Exception as e:
                print(f"Error getting collection item: {e}")
    return all_collection_items

def get_collection_item(stac_collection_id: str, stac_url:str) -> List[CollectionItem]:
    """Get collection item from the collection file."""
    [collection_file_path, items_file_path, stac_collection, stac_collection_items] = download_stac_data(stac_url=stac_url, stac_collection_id=stac_collection_id, output_dir="./data")
    collection_items: List[CollectionItem] = parse_stac_items_to_collection_items(stac_collection_items, stac_collection)
    return collection_items

async def scrape_text_from_url(url: str) -> str:
    default_scraper = get_default_scraper(debug=False)
    scraper_input = default_scraper.input_schema(
      url = url
    )
    scraped_output = await default_scraper.arun(scraper_input)
    scraped_text = scraped_output.content
    return scraped_text

async def scrape_text_from_urls(urls: List[str]) -> str:
    tasks = [scrape_text_from_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return "\n\n`````\n\n".join(results)
