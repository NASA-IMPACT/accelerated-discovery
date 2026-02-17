import asyncio
import re
import time
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dateparser
import gdown
import requests
from loguru import logger
from pydantic import BaseModel, HttpUrl, create_model

if TYPE_CHECKING:
    pass


def async_lru_cache(maxsize: int = 128):
    """
    LRU cache decorator for async functions using Python's built-in lru_cache.

    Caches the result (not the coroutine) to avoid 'cannot reuse awaited coroutine' errors.

    Args:
        maxsize: Maximum size of the cache. Defaults to 128.

    Example:
        ```python
        @async_lru_cache(maxsize=256)
        async def fetch_data(query: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://api.example.com/{query}")
                return response.json()

        result1 = await fetch_data("test")  # First call - fetches from API
        result2 = await fetch_data("test")  # Second call - returns cached result
        ```

    Note:
        All arguments to the decorated function must be hashable.
    """

    def decorator(async_fn):
        @lru_cache(maxsize=maxsize)
        def _cached_result_key(*args, **kwargs):
            """Create a unique cache key."""
            return (args, tuple(sorted(kwargs.items())))

        _cache: dict = {}

        @wraps(async_fn)
        async def wrapper(*args, **kwargs):
            cache_key = _cached_result_key(*args, **kwargs)

            if cache_key in _cache:
                return _cache[cache_key]

            result = await async_fn(*args, **kwargs)
            _cache[cache_key] = result
            return result

        wrapper.cache_info = _cached_result_key.cache_info
        wrapper.cache_clear = lambda: (_cached_result_key.cache_clear(), _cache.clear())

        return wrapper

    return decorator


class RateLimiter:
    """
    Simple rate limiter for API requests.

    This class ensures that API requests are spaced out according to a specified
    rate limit to avoid exceeding API quotas. It's thread-safe and async-friendly.

    Example:
        ```python
            rate_limiter = RateLimiter(max_calls_per_second=1.0)

            async def make_request():
                await rate_limiter.acquire()
                # Make your API request here
                response = await api_client.get("/endpoint")
                return response
        ```

    Args:
        max_calls_per_second (float): Maximum number of API calls per second.
            Default is 1.0 (one call per second).
    """

    def __init__(self, max_calls_per_second: float = 1.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_called = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request, blocking if necessary.

        This method will sleep if needed to ensure requests don't exceed
        the configured rate limit. It's safe to call from multiple
        concurrent tasks.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_called

            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                self.last_called = time.time()
            else:
                self.last_called = now


def get_akd_root() -> Path:
    """
    Returns the root directory of the AKD project.
    """
    return Path(__file__).parent.parent.resolve()


def google_drive_downloader(
    file_id: str,
    output_path: str,
    quiet: bool = False,
) -> None:
    """
    Download a file from Google Drive using the file ID.

    Args:
        file_id (str): The ID of the file on Google Drive.
        output_path (str): The path to save the downloaded file.
        quiet (bool): Whether to suppress download output.
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=quiet)
        logger.info(f"Downloaded file from Google Drive to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        raise


def is_server_available(url: str | HttpUrl) -> bool:
    """
    Check if a url is available.

    Args:
        url: The URL to test

    Returns:
        bool: True if server is reachable, False otherwise
    """
    # Sanity check on URL
    if not (url.startswith("http://") or url.startswith("https://")):
        logger.warning(f"URL {url} is not a valid URL.")
        return False

    try:
        # Check if the URL is reachable
        requests.head(url, timeout=5, allow_redirects=True)
        logger.info(f"URL {url} is reachable.")
        return True
    except requests.RequestException:
        logger.warning(f"URL {url} is not reachable.")
        return False


def get_model_fields(
    model_class: type[BaseModel],
    skip_no_description: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract field information from a Pydantic model using the most Pydantic-native approach.
    Uses Pydantic's built-in model_json_schema() method.

    Args:
        model_class: Pydantic model class to extract fields from
        skip_no_description: If True, skip fields without descriptions

    Returns:
        List of dictionaries containing field information with all schema properties
        plus 'name' and 'is_required' keys
    """
    if not model_class or not hasattr(model_class, "model_json_schema"):
        return []

    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    fields_info = []
    for field_name, field_schema in properties.items():
        if skip_no_description and not field_schema.get("description"):
            continue

        field_data = {
            "name": field_name,
            "is_required": field_name in required_fields,
            **field_schema,  # Include all schema properties
        }
        fields_info.append(field_data)

    return fields_info


def parse_date(date_input: str | int | None) -> datetime | None:
    """
    Parse various date formats to datetime object.

    Handles:
    - Human-readable dates: "4 days ago", "yesterday", "2 days ago"
    - Year integers: 2007, 2025
    - ISO dates: "2025-10-05"
    - Partial dates: "Oct 2025"

    Args:
        date_input: Date string, year integer, or None

    Returns:
        datetime object or None if parsing fails

    Examples:
        >>> parse_date("4 days ago")
        datetime.datetime(2025, 10, 4, ...)
        >>> parse_date(2007)
        datetime.datetime(2007, 1, 1, 0, 0)
        >>> parse_date("2025-10-05")
        datetime.datetime(2025, 10, 5, 0, 0)
    """
    parsed_date = None

    # Handle year integers
    if isinstance(date_input, int):
        parsed_date = datetime(date_input, 1, 1)

    # Handle string dates
    elif isinstance(date_input, str):
        parsed_date = dateparser.parse(date_input)

    return parsed_date


def to_snake_case(name: str) -> str:
    """
    Convert CamelCase/PascalCase to snake_case, handling acronyms.

    Examples:
        SearxNGSearchTool -> searxng_search_tool
        QueryAgent -> query_agent
        CMRDataExtractor -> cmr_data_extractor
    """
    # Insert _ between lowercase and uppercase: deepLit -> deep_Lit
    result = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    # Insert _ between acronym and next word: CMRAgent -> CMR_Agent
    result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", result)
    return result.lower()


class PartialModel[T: BaseModel]:
    """Partial schema generator - all fields become Optional.

    Creates a Pydantic model where all fields are Optional, useful for
    streaming PARTIAL events where output builds progressively.

    Usage:
        PartialModel[MySchema]           # Returns the partial model class
        PartialModel[MySchema](field=v)  # Creates instance

    Example:
        from akd.utils import PartialModel
        from akd.agents.search._base import LitSearchAgentOutputSchema

        # Create partial with only some fields
        partial = PartialModel[LitSearchAgentOutputSchema](
            results=[...],
            extra={"key_findings": [...]},
        )

        # Serialize for frontend
        partial.model_dump()  # {'answer': None, 'report': None, 'results': [...], 'extra': {...}}
    """

    _cache: dict[type[BaseModel], type[BaseModel]] = {}

    def __class_getitem__(cls, model: type[T]) -> type[T]:
        """Generate partial version of model with all fields Optional."""
        if model not in cls._cache:
            fields = {}
            for name, field_info in model.model_fields.items():
                fields[name] = (field_info.annotation | None, None)
            cls._cache[model] = create_model(
                f"Partial{model.__name__}",
                __doc__=model.__doc__,
                __base__=model,
                **fields,
            )
        return cls._cache[model]
