"""
Base classes for data search tools.
"""

from typing import Optional

from pydantic import Field
from pydantic.networks import HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig


class DataSearchToolConfig(BaseToolConfig):
    """
    Base configuration for data search tools.
    Common settings for tools that interact with data repositories.
    """

    mcp_endpoint: HttpUrl = Field(
        default="http://localhost:8080/mcp/cmr/mcp/",
        description="MCP server endpoint URL",
    )
    timeout_seconds: float = Field(
        default=30.0, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Base delay between retries in seconds"
    )
    page_size: int = Field(
        default=25, description="Default page size for paginated requests"
    )


class DataSearchToolInputSchema(InputSchema):
    """
    Base input schema for data search tools.
    Common parameters used across different data search tools.
    """

    temporal: Optional[str] = Field(
        None,
        description="Temporal range in ISO format (YYYY-MM-DDTHH:mm:ssZ,YYYY-MM-DDTHH:mm:ssZ)",
    )
    bounding_box: Optional[str] = Field(
        None, description="Spatial bounding box as 'west,south,east,north'"
    )
    page_size: Optional[int] = Field(None, description="Number of results per page")
    page_num: Optional[int] = Field(None, description="Page number (1-based)")


class DataSearchToolOutputSchema(OutputSchema):
    """
    Base output schema for data search tools.
    Standardized response format for all data search tools.
    """

    results: dict = Field(..., description="Search results in native format")
    total_hits: int = Field(..., description="Total number of matching results")
    query_time_ms: int = Field(..., description="Query execution time in milliseconds")
    page_info: dict = Field(default_factory=dict, description="Pagination information")


class BaseDataSearchTool[
    TInput: DataSearchToolInputSchema,
    TOutput: DataSearchToolOutputSchema,
](BaseTool[TInput, TOutput]):
    """
    Abstract base class for data search tools.

    Provides common functionality for tools that search scientific data repositories:
    - HTTP client setup and configuration
    - Request retry logic with exponential backoff
    - Error handling and logging
    - Response format standardization
    """

    input_schema = DataSearchToolInputSchema
    output_schema = DataSearchToolOutputSchema
    config_schema = DataSearchToolConfig

    def _prepare_request_headers(self) -> dict:
        """Prepare common HTTP headers for requests."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "AKD-DataSearch/1.0.0",
        }

    def _build_mcp_request(self, tool_name: str, arguments: dict) -> dict:
        """Build standardized MCP JSON-RPC request."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

    def _parse_mcp_response(self, response_text: str) -> dict:
        """Parse MCP server response from SSE format."""
        import json

        # Handle Server-Sent Events format
        content = response_text.strip()
        for line in content.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line.split("data: ", 1)[1])

                if "error" in data:
                    raise Exception(f"MCP Error: {data['error']}")

                # Parse the tool result
                if "result" in data and "content" in data["result"]:
                    tool_result = json.loads(data["result"]["content"][0]["text"])
                    return tool_result

        raise Exception("Invalid MCP response format")

    def _validate_spatial_bounds(self, bounding_box: Optional[str]) -> Optional[list]:
        """Validate and parse spatial bounding box."""
        if not bounding_box:
            return None

        try:
            coords = [float(x.strip()) for x in bounding_box.split(",")]
            if len(coords) != 4:
                raise ValueError("Bounding box must have exactly 4 coordinates")

            west, south, east, north = coords
            if not (-180 <= west <= 180 and -180 <= east <= 180):
                raise ValueError("Longitude values must be between -180 and 180")
            if not (-90 <= south <= 90 and -90 <= north <= 90):
                raise ValueError("Latitude values must be between -90 and 90")
            if west >= east:
                raise ValueError("West longitude must be less than east longitude")
            if south >= north:
                raise ValueError("South latitude must be less than north latitude")

            return coords
        except Exception as e:
            raise ValueError(f"Invalid bounding box format: {e}")

    def _validate_temporal_range(self, temporal: Optional[str]) -> Optional[str]:
        """Validate temporal range format."""
        if not temporal:
            return None

        # Basic validation - could be enhanced with proper datetime parsing
        if "," not in temporal:
            raise ValueError(
                "Temporal range must contain start and end times separated by comma"
            )

        parts = temporal.split(",")
        if len(parts) != 2:
            raise ValueError("Temporal range must have exactly two datetime values")

        return temporal.strip()

    async def _make_http_request(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """Make HTTP request to MCP server with retry logic."""
        import asyncio
        import httpx
        from loguru import logger

        config = self.config
        headers = self._prepare_request_headers()
        request_data = self._build_mcp_request(tool_name, arguments)

        for attempt in range(config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                    response = await client.post(
                        str(config.mcp_endpoint),
                        json=request_data,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        return self._parse_mcp_response(response.text)
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if attempt < config.max_retries:
                            wait_time = config.retry_delay * (2**attempt)
                            if self.debug:
                                logger.warning(
                                    f"Request failed (attempt {attempt + 1}): {error_msg}. "
                                    f"Retrying in {wait_time}s..."
                                )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise Exception(error_msg)

            except httpx.TimeoutException:
                error_msg = f"Request timeout after {config.timeout_seconds}s"
                if attempt < config.max_retries:
                    wait_time = config.retry_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Request timeout (attempt {attempt + 1}). "
                            f"Retrying in {wait_time}s..."
                        )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(error_msg)

            except Exception as e:
                if attempt < config.max_retries:
                    wait_time = config.retry_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e

        raise Exception("Maximum retries exceeded")
