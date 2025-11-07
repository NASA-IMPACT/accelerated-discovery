"""
PDS4 Instrument Search Tool - Search for scientific instruments.
"""

from typing import Optional

from pydantic import Field

from akd._base import OutputSchema

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
)


class PDS4InstrumentSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for PDS4 instrument search operations."""

    keywords: str = Field(
        "",
        description="Space-delimited search terms (e.g., 'camera mars', 'spectrometer cassini')",
    )
    instrument_type: str = Field(
        "",
        description="Filter by instrument type (e.g., 'Spectrometer', 'Imager', 'Radio-Radar')",
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return",
    )


class PDS4InstrumentSearchOutputSchema(OutputSchema):
    """Output schema for PDS4 instrument search operations."""

    total_hits: int = Field(..., description="Total number of matching instruments")
    query_time_ms: int = Field(default=0, description="Query execution time in milliseconds")
    query: str = Field(default="", description="Query string used")
    limit: int = Field(default=10, description="Maximum number of results requested")
    instruments: list = Field(..., description="List of found instruments")


class PDS4InstrumentSearchTool(
    BaseDataSearchTool[
        PDS4InstrumentSearchInputSchema,
        PDS4InstrumentSearchOutputSchema,
    ],
):
    """
    Tool for searching PDS Context products that are Instruments.

    Instruments are scientific devices used to collect measurements and observations.
    Example: ChemCam - urn:nasa:pds:context:instrument:msl.chemcam

    Use for queries about specific instruments, instrument types, or finding instruments
    used on particular missions or platforms.
    """

    input_schema = PDS4InstrumentSearchInputSchema
    output_schema = PDS4InstrumentSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: PDS4InstrumentSearchInputSchema,
        **kwargs,
    ) -> PDS4InstrumentSearchOutputSchema:
        """
        Execute PDS4 instrument search using MCP server.

        Args:
            params: Search parameters including keywords, instrument_type, and limit
            **kwargs: Additional parameters

        Returns:
            PDS4InstrumentSearchOutputSchema with instrument results
        """
        # Prepare MCP arguments
        arguments = {
            "keywords": params.keywords,
            "limit": params.limit,
        }

        # Add instrument_type if provided
        if params.instrument_type:
            arguments["instrument_type"] = params.instrument_type

        # Call MCP server
        result = await self._make_http_request(
            tool_name="search_instruments",
            arguments=arguments,
        )

        # Return structured output
        return PDS4InstrumentSearchOutputSchema(
            total_hits=result.get("total_hits") or 0,
            query_time_ms=result.get("query_time_ms") or 0,
            query=result.get("query") or "",
            limit=result.get("limit") or params.limit,
            instruments=result.get("instruments") or [],
        )

    @classmethod
    def from_params(
        cls,
        debug: bool = False,
        **kwargs,
    ) -> "PDS4InstrumentSearchTool":
        """
        Convenience constructor for creating tool with configuration parameters.

        Args:
            debug: Enable debug logging
            **kwargs: Additional configuration parameters (mcp_endpoint, timeout_seconds, etc.)

        Returns:
            Configured PDS4InstrumentSearchTool instance
        """
        config = DataSearchToolConfig(
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
