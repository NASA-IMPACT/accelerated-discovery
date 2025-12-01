import asyncio
import functools
import operator
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dateparser
import gdown
import requests
from loguru import logger
from pydantic import BaseModel, HttpUrl

if TYPE_CHECKING:
    pass


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


### Recursive attribute access ####
def rgetattr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Recursive GetAttr: Gets a nested attribute using a dot-separated string.

    Uses operator.attrgetter for clean, efficient attribute access.

    Example:
        >>> rgetattr(obj, 'a.b.c')
        # equivalent to obj.a.b.c

        >>> rgetattr(obj, 'a.b.c', default=0)
        # returns 0 if any attribute in the path doesn't exist

    Args:
        obj: The object to get the attribute from
        attr: Dot-separated attribute path (e.g., 'component.config.temperature')
        default: Default value to return if attribute doesn't exist

    Returns:
        The value of the nested attribute, or default if not found
    """
    return operator.attrgetter(attr)(obj)


def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive SetAttr: Sets a nested attribute using a dot-separated string.

    Example:
        >>> rsetattr(obj, 'a.b.c', 10)
        # equivalent to obj.a.b.c = 10

    Args:
        obj: The object to set the attribute on
        attr: Dot-separated attribute path (e.g., 'component.config.temperature')
        val: The value to set

    Raises:
        AttributeError: If any intermediate attribute in the path doesn't exist
    """
    pre, _, post = attr.rpartition(".")

    # If there is a dot (nested path)
    if pre:
        # Traverse down to the parent object (e.g., go to obj.a.b)
        parent = functools.reduce(getattr, pre.split("."), obj)
        # Set the attribute on that parent
        setattr(parent, post, val)

    # If there is no dot (simple attribute)
    else:
        setattr(obj, attr, val)
