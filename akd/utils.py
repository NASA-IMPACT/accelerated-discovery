import asyncio
import time
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

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
    key: str = "url",
    normalize: bool = True,
) -> list["SearchResultItem"]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(item) = Σ(1 / (k + rank)) across all lists containing item

    Args:
        results: Variable number of ranked result lists (one per query).
        k: RRF constant (default 60, standard value from literature).
        key: Attribute name to use as deduplication key (default "url").
        normalize: If True, apply min-max normalization to scores [0.1, 1.0] (default True).
                  Raw RRF scores are always preserved in item.extra["rrf_score"].

    Returns:
        Fused list sorted by RRF score (descending).

    Examples:
        >>> results_q1 = [item_a, item_b, item_c]
        >>> results_q2 = [item_b, item_d, item_a]
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2)
        # item_b and item_a appear in both, so they get boosted
    """
    # Return empty if no results
    if not results:
        return []

    rrf_map = defaultdict(float)
    item_map = {}

    # Calculate RRF scores
    for rank_list in results:
        for rank, item in enumerate(rank_list, 1):
            identifier = getattr(item, key, None)
            if identifier:
                rrf_map[identifier] += 1.0 / (rank + k)
                if identifier not in item_map:
                    item_map[identifier] = item.model_copy()  # Copy to avoid mutation

    # Build results with RRF scores
    fused_results = []
    for identifier, rrf_score in sorted(rrf_map.items(), key=lambda x: x[1], reverse=True):
        item = item_map[identifier]
        item.score = rrf_score
        item.extra = item.extra or {}
        item.extra["rrf_score"] = rrf_score
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
