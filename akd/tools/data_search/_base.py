"""
Base classes for data search tools.
"""

import json
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic.networks import HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils.logging import ContextualLogger


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
        default=30.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds",
    )
    page_size: int = Field(
        default=25,
        description="Default page size for paginated requests",
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
        None,
        description="Spatial bounding box as 'west,south,east,north'",
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_logger = ContextualLogger(self.__class__.__name__)

    def _prepare_request_headers(self) -> dict:
        """Prepare common HTTP headers for requests."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "AKD-DataSearch/1.0.0",
        }

    def _build_mcp_request(self, tool_name: str, arguments: dict) -> dict:
        """Build standardized MCP JSON-RPC request with validation."""
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool name must be a non-empty string")

        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary")

        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

    def _parse_mcp_response(self, response_text: str) -> dict:
        """Parse MCP server response from SSE format with validation."""
        try:
            # Handle Server-Sent Events format
            content = response_text.strip()
            for line in content.split("\n"):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line.split("data: ", 1)[1])
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in MCP response: {e}")

                    if "error" in data:
                        error_details = data.get("error", {})
                        if isinstance(error_details, dict):
                            error_msg = error_details.get("message", str(error_details))
                        else:
                            error_msg = str(error_details)
                        raise Exception(f"MCP Error: {error_msg}")

                    # Validate and parse the tool result
                    if "result" in data:
                        result = data["result"]
                        if not isinstance(result, dict):
                            raise Exception(
                                f"Invalid MCP result format: expected dict, got {type(result)}",
                            )

                        if "content" not in result:
                            raise Exception("Missing 'content' field in MCP result")

                        content_list = result["content"]
                        if not isinstance(content_list, list) or len(content_list) == 0:
                            raise Exception(
                                "Invalid 'content' field: expected non-empty list",
                            )

                        content_item = content_list[0]
                        if (
                            not isinstance(content_item, dict)
                            or "text" not in content_item
                        ):
                            raise Exception(
                                "Invalid content item: expected dict with 'text' field",
                            )

                        try:
                            tool_result = json.loads(content_item["text"])
                            return self._validate_tool_result(tool_result)
                        except json.JSONDecodeError as e:
                            raise Exception(f"Invalid JSON in tool result: {e}")

            raise Exception("No valid data found in MCP response")
        except Exception as e:
            self.tool_logger.error(f"MCP response parsing failed: {e}")
            if self.debug:
                self.tool_logger.debug(f"Raw response: {response_text[:500]}...")
            raise

    def _validate_tool_result(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool result structure and add fallback values if needed."""
        if not isinstance(tool_result, dict):
            raise Exception(
                f"Tool result must be a dictionary, got {type(tool_result)}",
            )

        # Ensure required fields exist with fallback values
        validated_result = {
            "total_hits": tool_result.get("total_hits", 0),
            "query_time_ms": tool_result.get("query_time_ms", 0),
            "page_size": tool_result.get("page_size", 0),
            "page_number": tool_result.get("page_number", 1),
            **tool_result,  # Keep all original fields
        }

        return validated_result

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
                "Temporal range must contain start and end times separated by comma",
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
        """Make HTTP request to MCP server with retry logic and circuit breaker."""
        import asyncio

        import httpx

        config = self.config
        headers = self._prepare_request_headers()
        request_data = self._build_mcp_request(tool_name, arguments)

        last_exception = None

        for attempt in range(config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                    self.tool_logger.debug(
                        f"Making MCP request (attempt {attempt + 1}): {tool_name}",
                    )

                    response = await client.post(
                        str(config.mcp_endpoint),
                        json=request_data,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        try:
                            return self._parse_mcp_response(response.text)
                        except Exception as parse_error:
                            self.tool_logger.error(
                                f"Failed to parse MCP response: {parse_error}",
                            )
                            last_exception = parse_error
                            if attempt == config.max_retries:
                                raise
                            continue
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        last_exception = Exception(error_msg)
                        if attempt < config.max_retries:
                            wait_time = config.retry_delay * (2**attempt)
                            self.tool_logger.warning(
                                f"Request failed (attempt {attempt + 1}): {error_msg}. "
                                f"Retrying in {wait_time}s...",
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise last_exception

            except httpx.TimeoutException:
                error_msg = f"Request timeout after {config.timeout_seconds}s"
                last_exception = Exception(error_msg)
                if attempt < config.max_retries:
                    wait_time = config.retry_delay * (2**attempt)
                    self.tool_logger.warning(
                        f"Request timeout (attempt {attempt + 1}). "
                        f"Retrying in {wait_time}s...",
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise last_exception

            except Exception as e:
                last_exception = e
                if attempt < config.max_retries:
                    wait_time = config.retry_delay * (2**attempt)
                    self.tool_logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait_time}s...",
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e

        raise Exception(f"Maximum retries exceeded. Last error: {last_exception}")
