"""
PDS4 Investigation Search Tool - Search for missions and projects.
"""

from typing import Optional

from pydantic import Field

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig,
    DataSearchToolInputSchema,
    DataSearchToolOutputSchema,
)


class PDS4InvestigationSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for PDS4 investigation search operations."""

    keywords: str = Field(
        "",
        description="Space-delimited search terms (e.g., 'mars rover', 'jupiter cassini')",
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return",
    )


class PDS4InvestigationSearchOutputSchema(DataSearchToolOutputSchema):
    """Output schema for PDS4 investigation search operations."""

    total_hits: int = Field(..., description="Total number of matching investigations")
    investigations: list = Field(..., description="List of found investigations")


class PDS4InvestigationSearchTool(
    BaseDataSearchTool[
        PDS4InvestigationSearchInputSchema,
        PDS4InvestigationSearchOutputSchema,
    ],
):
    """
    Tool for searching PDS Context products that are Investigations (missions/projects).

    Investigations are organized missions or projects that collect scientific data.
    Example: Cassini-Huygens - urn:nasa:pds:context:investigation:mission.cassini-huygens

    Use for queries about space missions, mission timelines, or finding missions
    that studied specific targets.
    """

    input_schema = PDS4InvestigationSearchInputSchema
    output_schema = PDS4InvestigationSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: PDS4InvestigationSearchInputSchema,
        **kwargs,
    ) -> PDS4InvestigationSearchOutputSchema:
        """
        Execute PDS4 investigation search using MCP server.

        Args:
            params: Search parameters including keywords and limit
            **kwargs: Additional parameters

        Returns:
            PDS4InvestigationSearchOutputSchema with investigation results
        """
        # Prepare MCP arguments
        arguments = {
            "keywords": params.keywords,
            "limit": params.limit,
        }

        # Call MCP server
        result = await self._call_mcp_tool(
            tool_name="search_investigations",
            arguments=arguments,
        )

        # Return structured output
        return PDS4InvestigationSearchOutputSchema(
            total_hits=result.get("total_hits", 0),
            investigations=result.get("investigations", []),
        )

    @classmethod
    def from_params(
        cls,
        debug: bool = False,
        **kwargs,
    ) -> "PDS4InvestigationSearchTool":
        """
        Convenience constructor for creating tool with configuration parameters.

        Args:
            debug: Enable debug logging
            **kwargs: Additional configuration parameters (mcp_endpoint, timeout_seconds, etc.)

        Returns:
            Configured PDS4InvestigationSearchTool instance
        """
        config = DataSearchToolConfig(
            debug=debug,
            **kwargs,
        )

        return cls(config=config, debug=debug)
