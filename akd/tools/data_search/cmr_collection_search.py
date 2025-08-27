"""
CMR Collection Search Tool - Search for Earth science collections/datasets.
"""

from typing import Optional

from pydantic import Field

from akd.utils.logging import log_api_request

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
    DataSearchToolOutputSchema,
)


class CMRCollectionSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for CMR collection search operations."""

    keyword: Optional[str] = Field(
        None,
        description="Text search across collection metadata (AND search)",
    )
    short_name: Optional[str] = Field(
        None,
        description="Collection short name (e.g., MOD09A1)",
    )
    platform: Optional[str] = Field(
        None,
        description="Platform/satellite name (e.g., Terra, Aqua)",
    )
    instrument: Optional[str] = Field(
        None,
        description="Instrument name (e.g., MODIS, VIIRS)",
    )


class CMRCollectionSearchOutputSchema(DataSearchToolOutputSchema):
    """Output schema for CMR collection search operations."""

    collections: list = Field(..., description="List of found collections")


class CMRCollectionSearchTool(
    BaseDataSearchTool[
        CMRCollectionSearchInputSchema,
        CMRCollectionSearchOutputSchema,
    ],
):
    """
    Tool for searching Earth science collections in NASA's CMR.

    Wraps the CMR MCP server's search_collections endpoint to discover
    datasets based on scientific parameters like platform, instrument,
    processing level, and temporal/spatial constraints.
    """

    input_schema = CMRCollectionSearchInputSchema
    output_schema = CMRCollectionSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: CMRCollectionSearchInputSchema,
        **kwargs,
    ) -> CMRCollectionSearchOutputSchema:
        """
        Execute CMR collection search using MCP server.

        Args:
            params: Search parameters including keywords, platform, instrument, etc.
            **kwargs: Additional parameters

        Returns:
            CMRCollectionSearchOutputSchema with collection results
        """
        self.tool_logger.debug(f"Starting CMR collection search with params: {params}")

        # Prepare MCP arguments
        arguments = {}

        # Add search parameters
        if params.keyword:
            arguments["keyword"] = params.keyword
        if params.short_name:
            arguments["short_name"] = params.short_name
        if params.platform:
            arguments["platform"] = params.platform
        if params.instrument:
            arguments["instrument"] = params.instrument

        # Add temporal constraint
        if params.temporal:
            validated_temporal = self._validate_temporal_range(params.temporal)
            if validated_temporal:
                arguments["temporal"] = validated_temporal

        # Add spatial constraint
        if params.bounding_box:
            validated_bbox = self._validate_spatial_bounds(params.bounding_box)
            if validated_bbox:
                arguments["bounding_box"] = ",".join(map(str, validated_bbox))

        # Add pagination
        page_size = params.page_size or self.config.page_size
        arguments["page_size"] = min(page_size, 50)  # MCP server limit
        if params.page_num:
            arguments["page_num"] = params.page_num

        try:
            # Make request to MCP server
            result = await self._make_http_request("search_collections", arguments)

            if "error" in result:
                raise Exception(f"CMR search error: {result['error']}")

            # Extract response data
            total_hits = result.get("total_hits", 0)
            query_time_ms = result.get("query_time_ms", 0)
            collections = result.get("collections", [])
            page_size_returned = result.get("page_size", page_size)
            page_number = result.get("page_number", 1)

            # Log API request details
            log_api_request("POST", str(self.config.mcp_endpoint), 200, query_time_ms)
            self.tool_logger.info(
                f"CMR collection search completed: {total_hits} total hits, "
                f"{len(collections)} collections returned, "
                f"query time: {query_time_ms}ms",
            )

            return CMRCollectionSearchOutputSchema(
                results=result,  # Return full MCP response
                total_hits=total_hits,
                query_time_ms=query_time_ms,
                collections=collections,
                page_info={
                    "page_size": page_size_returned,
                    "page_number": page_number,
                    "total_pages": (total_hits + page_size_returned - 1)
                    // page_size_returned
                    if page_size_returned > 0
                    else 0,
                },
            )

        except Exception as e:
            error_msg = f"CMR collection search failed: {e}"
            log_api_request("POST", str(self.config.mcp_endpoint), status_code=500)
            self.tool_logger.error(error_msg)
            raise Exception(error_msg)

    def _build_search_summary(self, params: CMRCollectionSearchInputSchema) -> str:
        """Build a human-readable summary of search parameters."""
        summary_parts = []

        if params.keyword:
            summary_parts.append(f"keyword: '{params.keyword}'")
        if params.platform:
            summary_parts.append(f"platform: {params.platform}")
        if params.instrument:
            summary_parts.append(f"instrument: {params.instrument}")
        if params.processing_level:
            summary_parts.append(f"level: {params.processing_level}")
        if params.temporal:
            summary_parts.append(f"temporal: {params.temporal}")
        if params.bounding_box:
            summary_parts.append(f"bbox: {params.bounding_box}")

        if summary_parts:
            return f"CMR collection search: {', '.join(summary_parts)}"
        else:
            return "CMR collection search: all collections"

    @classmethod
    def from_params(
        cls,
        keyword: Optional[str] = None,
        platform: Optional[str] = None,
        instrument: Optional[str] = None,
        temporal: Optional[str] = None,
        bounding_box: Optional[str] = None,
        page_size: int = 50,
        debug: bool = False,
        **kwargs,
    ) -> "CMRCollectionSearchTool":
        """
        Convenience constructor for creating tool with common parameters.

        Args:
            keyword: Text search terms
            platform: Satellite platform name
            instrument: Instrument name
            processing_level: Data processing level
            temporal: Temporal range
            bounding_box: Spatial bounds
            page_size: Results per page
            debug: Enable debug logging
            **kwargs: Additional configuration parameters

        Returns:
            Configured CMRCollectionSearchTool instance
        """
        config = DataSearchToolConfig(
            page_size=page_size,
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
