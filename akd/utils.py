import asyncio
import time
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import dateparser
import gdown
import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel, HttpUrl

if TYPE_CHECKING:
    from akd.structures import SearchResultItem

try:
    from langchain_core.tools.structured import StructuredTool

    LANGCHAIN_CORE_INSTALLED = True
except Exception:
    LANGCHAIN_CORE_INSTALLED = False


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class AsyncRunMixin:
    """
    Mixin for adding interface to run
    async methods in a sync context.
    """

    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses should implement this method")

    # async def ainvoke(self, *args, **kwargs) -> Any:
    #     return await self.arun(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        Runs the async method in a sync context.
        """
        if not hasattr(self, "arun"):
            raise AttributeError("Method 'arun' not implemented in the class")
        try:
            # Check if there's a running event loop
            loop = get_event_loop()
            # If we're already in an event loop, we need to use create_task and wait for it
            if loop and loop.is_running():
                # This creates a new task in the current event loop
                future = asyncio.ensure_future(self.arun(*args, **kwargs))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.arun(*args, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.arun(*args, **kwargs))


class LangchainToolMixin:
    """
    Mixin for converting tools/agents to Langchain structured tools.

    Preconditions:
        Each tool/agent should have:
        - input schema (input_schema)
        - output schema (output_schema)
        - async run method (arun)
    """

    def to_langchain_structured_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> StructuredTool:
        if not LANGCHAIN_CORE_INSTALLED:
            raise ImportError("langchain-core is required to use this method")

        async def _wrapped_arun(**input_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            A wrapper around the tool's run method to accept JSON-like input and return JSON-like output.
            """
            # Validate and parse input using Pydantic
            validated_input = self.input_schema(**input_data)

            # Execute the tool
            output_obj = await self.arun(validated_input)

            # Convert output Pydantic object to dictionary
            return output_obj.model_dump()

        def _wrapped_run(**input_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            A wrapper around the tool's run method to accept JSON-like input and return JSON-like output.
            """
            # Validate and parse input using Pydantic
            validated_input = self.input_schema(**input_data)

            # Execute the tool
            output_obj = self.run(validated_input)

            # Convert output Pydantic object to dictionary
            return output_obj.model_dump()

        name = name or self.__class__.__name__
        doc = (self.__class__.__doc__ or "").strip()
        description = description or f"A tool that executes {name}." + (f" Description: {doc}" if doc else "")
        return StructuredTool.from_function(
            func=_wrapped_run,
            coroutine=_wrapped_arun,
            name=name,
            description=description,
            args_schema=self.input_schema,
        )


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


def reciprocal_rank_fusion(
    *results: list["SearchResultItem"],
    k: int = 60,
    keys: list[str] | str | None = None,
    value_normalizers: dict[str, Callable] | None = None,
    normalize: bool = True,
    debug: bool = False,
) -> list["SearchResultItem"]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF) with multi-key matching and value normalization.

    Formula: RRF_score(item) = Σ(1 / (k + rank)) across all lists containing item

    Args:
        results: Variable number of ranked result lists (one per query).
        k: RRF constant (default 60, standard value from literature).
        keys: List of attribute names for deduplication (cascaded OR logic).
              Default: ["doi", "title", "url"] - matches if ANY key matches.
              Can also pass single string for backward compatibility.
        value_normalizers: Optional dict mapping key names to normalizer functions.
              Each function takes a SearchResultItem and returns a normalized string or None.
              Default normalizers: doi (extract from URL), title (lowercase, no punctuation), url (no protocol/www).
              Pass empty dict {} to disable normalization.
              Example: {"url": lambda item: my_custom_normalizer(item.url)}
        normalize: If True, apply min-max normalization to scores [0.1, 1.0] (default True).
                  Raw RRF scores are always preserved in item.extra["rrf_score"].
        debug: Enable debug logging showing matched keys and values.

    Returns:
        Fused list sorted by RRF score (descending). Original item values are preserved.

    Notes:
        - Normalization is applied ONLY for matching/deduplication, not for output values
        - Items with URLs "https://github.com/foo" and "https://www.github.com/foo/" will match
        - Items with titles "Deep Learning: A Survey" and "deep learning a survey" will match
        - The first encountered item's original values are kept in the output

    Examples:
        >>> # Default normalization (recommended)
        >>> results_q1 = [item_a, item_b, item_c]
        >>> results_q2 = [item_b, item_d, item_a]
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2)
        # item_b and item_a appear in both, so they get boosted

        >>> # URL normalization matches different URL formats
        >>> item1 = SearchResultItem(url="https://github.com/foo/bar")
        >>> item2 = SearchResultItem(url="https://www.github.com/foo/bar/")
        >>> fused = reciprocal_rank_fusion([item1], [item2], keys=["url"])
        # Result: 1 item (matched via normalized URL), original URL preserved

        >>> # Custom normalizer
        >>> custom = {"url": lambda item: item.url.lower().strip()}
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2, value_normalizers=custom)

        >>> # Disable normalization
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2, value_normalizers={})
    """
    # Handle default and backward compatibility
    keys = keys or ["doi", "title", "url"]
    if isinstance(keys, str):
        keys = [keys]

    # Return empty if no results
    if not results:
        return []

    # Track: identifier -> cumulative RRF score
    # Track: identifier -> SearchResultItem
    # Track: identifier -> set of all (key, value) identifiers for this item
    identifier_to_score = defaultdict(float)
    identifier_to_item = {}
    identifier_groups = defaultdict(set)

    # Import normalizer functions (avoid circular import at module level)
    from akd.tools.search.utils import get_doi, normalize_title, normalize_url

    # Use provided normalizers or defaults
    value_normalizers = value_normalizers or {
        "doi": lambda item: get_doi(item),
        "title": lambda item: normalize_title(item.title) if item.title else None,
        "url": lambda item: normalize_url(item.url) if item.url else None,
    }

    # Calculate RRF scores with multi-key matching
    for rank_list in results:
        for rank, item in enumerate(rank_list, 1):
            # Collect all non-None identifiers for this item (using normalized values)
            item_identifiers = set()
            for key in keys:
                # Apply normalizer if available, otherwise use original value
                if key in value_normalizers:
                    value = value_normalizers[key](item)
                else:
                    value = getattr(item, key, None)

                if value:
                    item_identifiers.add((key, str(value)))

            if not item_identifiers:
                continue  # Skip items with no valid identifiers

            # Pick canonical identifier (prefer first key in priority order)
            canonical = None
            for key in keys:
                for item_key, item_val in item_identifiers:
                    if item_key == key:
                        canonical = (item_key, item_val)
                        break
                if canonical:
                    break

            if not canonical:
                canonical = next(iter(item_identifiers))

            # Check if any identifier matches existing groups
            matched_canonical = None
            for existing_canonical, group in identifier_groups.items():
                if item_identifiers & group:  # Set intersection - shared identifier?
                    matched_canonical = existing_canonical
                    break

            increment = 1.0 / (rank + k)
            if matched_canonical:
                # Merge with existing item - accumulate RRF score
                identifier_to_score[matched_canonical] += increment

                if debug:
                    # Show which keys matched (BEFORE updating the group!)
                    shared = item_identifiers & identifier_groups[matched_canonical]
                    matched_keys = [k for k, _ in shared]
                    canonical_key, canonical_val = matched_canonical
                    logger.debug(
                        f"[RRF] MERGED: rank={rank} matched via {matched_keys} | "
                        f"primary_key={canonical_key} value='{canonical_val[:40]}...' | +score={increment:.6f}",
                    )

                # Merge all identifiers into the group (AFTER logging!)
                identifier_groups[matched_canonical].update(item_identifiers)
            else:
                # New item - create new group
                identifier_to_score[canonical] += increment
                identifier_to_item[canonical] = item.model_copy()
                identifier_groups[canonical] = item_identifiers
                if debug:
                    available_keys = [k for k, _ in item_identifiers]
                    canonical_key, canonical_val = canonical
                    logger.debug(
                        f"[RRF] NEW: rank={rank} | "
                        f"primary_key={canonical_key} value='{canonical_val[:40]}...' | "
                        f"available_keys={available_keys}",
                    )

    # Build results with RRF scores
    fused_results = []
    for identifier, rrf_score in sorted(identifier_to_score.items(), key=lambda x: x[1], reverse=True):
        item = identifier_to_item[identifier]
        item.score = rrf_score
        item.extra = item.extra or {}
        item.extra["rrf_score"] = rrf_score
        # Track which keys were used for matching (useful for debugging)
        item.extra["rrf_matched_keys"] = list(set(k for k, _ in identifier_groups[identifier]))
        fused_results.append(item)

    if not fused_results:
        return []

    # Normalize scores if requested
    if normalize:
        scores = np.array([item.score for item in fused_results])
        score_range = scores.max() - scores.min()

        if score_range > 0:
            # Scale to [0.1, 1.0] range to avoid 0.0 scores
            normalized_scores = 0.1 + 0.9 * (scores - scores.min()) / score_range
        else:
            # All scores are the same
            normalized_scores = np.ones_like(scores)

        for item, norm_score in zip(fused_results, normalized_scores):
            item.score = float(norm_score)

    return fused_results
