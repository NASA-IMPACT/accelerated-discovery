"""
PDS4 Collection Search Tool - Search for data collections filtered by context references.
"""

from typing import Optional

from pydantic import Field

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
    DataSearchToolOutputSchema,
)


class PDS4CollectionSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for PDS4 collection search operations."""

    ref_lid_instrument: str = Field(
        "",
        description="URN identifier for instrument (e.g., urn:nasa:pds:context:instrument:mars2020.mastcamz)",
    )
    ref_lid_target: str = Field(
        "",
        description="URN identifier for target (e.g., urn:nasa:pds:context:target:planet.mars)",
    )
    ref_lid_instrument_host: str = Field(
        "",
        description="URN identifier for instrument host (e.g., urn:nasa:pds:context:instrument_host:spacecraft.mars2020)",
    )
    ref_lid_investigation: str = Field(
        "",
        description="URN identifier for investigation (e.g., urn:nasa:pds:context:investigation:mission.mars2020)",
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return",
    )


class PDS4CollectionSearchOutputSchema(DataSearchToolOutputSchema):
    """Output schema for PDS4 collection search operations."""

    total_hits: int = Field(..., description="Total number of matching collections")
    collections: list = Field(..., description="List of found collections")


class PDS4CollectionSearchTool(
    BaseDataSearchTool[
        PDS4CollectionSearchInputSchema,
        PDS4CollectionSearchOutputSchema,
    ],
):
    """
    Tool for searching PDS data collections filtered by context references.

    Collections are data products filtered by instrument, target, instrument host,
    or investigation URNs. This tool is typically used after context discovery to
    find actual data.

    Example: Mars Reconnaissance Orbiter HiRISE data collections targeting Mars.
    """

    input_schema = PDS4CollectionSearchInputSchema
    output_schema = PDS4CollectionSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: PDS4CollectionSearchInputSchema,
        **kwargs,
    ) -> PDS4CollectionSearchOutputSchema:
        """
        Execute PDS4 collection search using MCP server.

        Args:
            params: Search parameters including context URN references and limit
            **kwargs: Additional parameters

        Returns:
            PDS4CollectionSearchOutputSchema with collection results
        """
        # Prepare MCP arguments
        arguments = {
            "ref_lid_instrument": params.ref_lid_instrument,
            "ref_lid_target": params.ref_lid_target,
            "ref_lid_instrument_host": params.ref_lid_instrument_host,
            "ref_lid_investigation": params.ref_lid_investigation,
            "limit": params.limit,
        }

        # Call MCP server
        result = await self._make_http_request(
            tool_name="search_collections",
            arguments=arguments,
        )

        # Return structured output
        return PDS4CollectionSearchOutputSchema(
            total_hits=result.get("total_hits", 0),
            collections=result.get("collections", []),
        )

    @classmethod
    def from_params(
        cls,
        debug: bool = False,
        **kwargs,
    ) -> "PDS4CollectionSearchTool":
        """
        Convenience constructor for creating tool with configuration parameters.

        Args:
            debug: Enable debug logging
            **kwargs: Additional configuration parameters (mcp_endpoint, timeout_seconds, etc.)

        Returns:
            Configured PDS4CollectionSearchTool instance
        """
        config = DataSearchToolConfig(
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
