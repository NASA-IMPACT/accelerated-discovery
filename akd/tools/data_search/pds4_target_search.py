"""
PDS4 Target Search Tool - Search for celestial bodies and phenomena.
"""

from typing import Optional

from pydantic import Field

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
    DataSearchToolOutputSchema,
)


class PDS4TargetSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for PDS4 target search operations."""

    keywords: str = Field(
        "",
        description="Space-delimited search terms (e.g., 'jupiter moon', 'asteroid belt')",
    )
    target_type: str = Field(
        "",
        description="Filter by target type (e.g., 'Planet', 'Satellite', 'Asteroid')",
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return",
    )


class PDS4TargetSearchOutputSchema(DataSearchToolOutputSchema):
    """Output schema for PDS4 target search operations."""

    total_hits: int = Field(..., description="Total number of matching targets")
    targets: list = Field(..., description="List of found targets")


class PDS4TargetSearchTool(
    BaseDataSearchTool[
        PDS4TargetSearchInputSchema,
        PDS4TargetSearchOutputSchema,
    ],
):
    """
    Tool for searching PDS Context products that are Targets (celestial bodies, phenomena).

    Targets are objects of scientific study: planets, moons, asteroids, comets, etc.
    Example: Mars - urn:nasa:pds:context:target:planet.mars

    Use for queries about specific celestial bodies, finding targets by type,
    or targets studied by missions.
    """

    input_schema = PDS4TargetSearchInputSchema
    output_schema = PDS4TargetSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: PDS4TargetSearchInputSchema,
        **kwargs,
    ) -> PDS4TargetSearchOutputSchema:
        """
        Execute PDS4 target search using MCP server.

        Args:
            params: Search parameters including keywords, target_type, and limit
            **kwargs: Additional parameters

        Returns:
            PDS4TargetSearchOutputSchema with target results
        """
        # Prepare MCP arguments
        arguments = {
            "keywords": params.keywords,
            "target_type": params.target_type,
            "limit": params.limit,
        }

        # Call MCP server
        result = await self._call_mcp_tool(
            tool_name="search_targets",
            arguments=arguments,
        )

        # Return structured output
        return PDS4TargetSearchOutputSchema(
            total_hits=result.get("total_hits", 0),
            targets=result.get("targets", []),
        )

    @classmethod
    def from_params(
        cls,
        debug: bool = False,
        **kwargs,
    ) -> "PDS4TargetSearchTool":
        """
        Convenience constructor for creating tool with configuration parameters.

        Args:
            debug: Enable debug logging
            **kwargs: Additional configuration parameters (mcp_endpoint, timeout_seconds, etc.)

        Returns:
            Configured PDS4TargetSearchTool instance
        """
        config = DataSearchToolConfig(
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
