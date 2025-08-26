import asyncio
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import gdown
from loguru import logger

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
        description = description or f"A tool that executes {name}." + (
            f" Description: {doc}" if doc else ""
        )
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
