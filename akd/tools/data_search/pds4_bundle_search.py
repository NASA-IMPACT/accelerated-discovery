"""
PDS4 Bundle Search Tool - Search for dataset bundles with faceting support.
"""

from typing import Optional

from pydantic import Field

from akd._base import OutputSchema

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
)


class PDS4BundleSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for PDS4 bundle search operations."""

    title_query: Optional[str] = Field(
        None,
        description="Search query for bundle titles (e.g., 'Mars', 'Lunar')",
    )
    limit: int = Field(
        0,
        description="Number of actual products to return (set to 0 for facets only)",
    )
    facet_fields: Optional[str] = Field(
        None,
        description="Comma-separated list of fields to facet on (e.g., 'pds:Identification_Area.pds:title,lidvid')",
    )
    facet_limit: int = Field(
        25,
        description="Maximum number of facet values to return",
    )


class PDS4BundleSearchOutputSchema(OutputSchema):
    """Output schema for PDS4 bundle search operations."""

    total_hits: int = Field(..., description="Total number of matching bundles")
    query_time_ms: int = Field(default=0, description="Query execution time in milliseconds")
    query: str = Field(default="", description="Query string used")
    limit: int = Field(default=0, description="Maximum number of results requested")
    bundles: list = Field(..., description="List of found bundles")
    facets: dict = Field(default_factory=dict, description="Faceted metadata results")


class PDS4BundleSearchTool(
    BaseDataSearchTool[
        PDS4BundleSearchInputSchema,
        PDS4BundleSearchOutputSchema,
    ],
):
    """
    Tool for searching PDS4 bundles with comprehensive faceted results.

    Bundles are top-level PDS4 products containing collections. This tool supports
    both title-based searching and faceted navigation of bundle metadata.

    Use for exploratory searches, metadata analysis, or supplementary data discovery
    beyond context-based search.
    """

    input_schema = PDS4BundleSearchInputSchema
    output_schema = PDS4BundleSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: PDS4BundleSearchInputSchema,
        **kwargs,
    ) -> PDS4BundleSearchOutputSchema:
        """
        Execute PDS4 bundle search using MCP server.

        Args:
            params: Search parameters including title_query, limit, facet options
            **kwargs: Additional parameters

        Returns:
            PDS4BundleSearchOutputSchema with bundle results and facets
        """
        # Prepare MCP arguments
        arguments = {
            "title_query": params.title_query,
            "limit": params.limit,
            "facet_fields": params.facet_fields,
            "facet_limit": params.facet_limit,
        }

        # Call MCP server
        result = await self._make_http_request(
            tool_name="search_bundles",
            arguments=arguments,
        )

        # Return structured output
        return PDS4BundleSearchOutputSchema(
            total_hits=result.get("total_hits") or 0,
            query_time_ms=result.get("query_time_ms") or 0,
            query=result.get("query") or "",
            limit=result.get("limit") or params.limit,
            bundles=result.get("bundles") or [],
            facets=result.get("facets") or {},
        )

    @classmethod
    def from_params(
        cls,
        debug: bool = False,
        **kwargs,
    ) -> "PDS4BundleSearchTool":
        """
        Convenience constructor for creating tool with configuration parameters.

        Args:
            debug: Enable debug logging
            **kwargs: Additional configuration parameters (mcp_endpoint, timeout_seconds, etc.)

        Returns:
            Configured PDS4BundleSearchTool instance
        """
        config = DataSearchToolConfig(
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
