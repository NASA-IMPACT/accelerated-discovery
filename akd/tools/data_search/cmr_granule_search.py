"""
CMR Granule Search Tool - Retrieve data files (granules) for collections.
"""

from typing import Optional

from pydantic import Field
from loguru import logger

from ._base import (
    BaseDataSearchTool,
    DataSearchToolConfig, 
    DataSearchToolInputSchema,
    DataSearchToolOutputSchema,
)


class CMRGranuleSearchInputSchema(DataSearchToolInputSchema):
    """Input schema for CMR granule search operations."""

    collection_concept_id: str = Field(
        ...,
        description="Collection concept ID (e.g., C123456-LPDAAC_ECS)"
    )
    point: Optional[str] = Field(
        None,
        description="Point search as 'longitude,latitude'"
    )
    producer_granule_id: Optional[str] = Field(
        None,
        description="Producer-assigned granule ID for specific file search"
    )
    online_only: Optional[bool] = Field(
        True,
        description="Return only online-accessible granules"
    )
    downloadable: Optional[bool] = Field(
        None,
        description="Return only downloadable granules"
    )
    cloud_cover: Optional[str] = Field(
        None,
        description="Cloud cover percentage range (e.g., '0,10' for 0-10%)"
    )


class CMRGranuleSearchOutputSchema(DataSearchToolOutputSchema):
    """Output schema for CMR granule search operations."""
    
    granules: list = Field(..., description="List of found granules")
    collection_concept_id: str = Field(..., description="Collection that was searched")


class CMRGranuleSearchTool(
    BaseDataSearchTool[
        CMRGranuleSearchInputSchema,
        CMRGranuleSearchOutputSchema,
    ]
):
    """
    Tool for retrieving data files (granules) from NASA's CMR.
    
    Wraps the CMR MCP server's get_granules endpoint to find actual
    data files within a specific collection, applying temporal, spatial,
    and access constraints.
    """

    input_schema = CMRGranuleSearchInputSchema
    output_schema = CMRGranuleSearchOutputSchema
    config_schema = DataSearchToolConfig

    async def _arun(
        self,
        params: CMRGranuleSearchInputSchema,
        **kwargs,
    ) -> CMRGranuleSearchOutputSchema:
        """
        Execute CMR granule search using MCP server.
        
        Args:
            params: Search parameters including collection ID and constraints
            **kwargs: Additional parameters
            
        Returns:
            CMRGranuleSearchOutputSchema with granule results
        """
        if self.debug:
            logger.debug(
                f"Starting CMR granule search for collection {params.collection_concept_id}"
            )

        # Prepare MCP arguments
        arguments = {
            "collection_concept_id": params.collection_concept_id
        }
        
        # Add temporal constraint
        if params.temporal:
            validated_temporal = self._validate_temporal_range(params.temporal)
            if validated_temporal:
                arguments["temporal"] = validated_temporal

        # Add spatial constraints
        if params.bounding_box:
            validated_bbox = self._validate_spatial_bounds(params.bounding_box)
            if validated_bbox:
                arguments["bounding_box"] = ",".join(map(str, validated_bbox))

        if params.point:
            validated_point = self._validate_point(params.point)
            if validated_point:
                arguments["point"] = ",".join(map(str, validated_point))

        # Add filter parameters
        if params.producer_granule_id:
            arguments["producer_granule_id"] = params.producer_granule_id
        if params.online_only is not None:
            arguments["online_only"] = params.online_only
        if params.downloadable is not None:
            arguments["downloadable"] = params.downloadable
        if params.cloud_cover:
            arguments["cloud_cover"] = params.cloud_cover

        # Add pagination
        page_size = params.page_size or self.config.page_size
        arguments["page_size"] = min(page_size, 50)  # MCP server limit
        if params.page_num:
            arguments["page_num"] = params.page_num

        try:
            # Make request to MCP server
            result = await self._make_http_request("get_granules", arguments)
            
            if "error" in result:
                raise Exception(f"CMR granule search error: {result['error']}")

            # Extract response data
            total_hits = result.get("total_hits", 0)
            query_time_ms = result.get("query_time_ms", 0)
            granules = result.get("granules", [])
            page_size_returned = result.get("page_size", page_size)
            page_number = result.get("page_number", 1)

            if self.debug:
                logger.debug(
                    f"CMR granule search completed: {total_hits} total hits, "
                    f"{len(granules)} granules returned, "
                    f"query time: {query_time_ms}ms"
                )

            return CMRGranuleSearchOutputSchema(
                results=result,  # Return full MCP response
                total_hits=total_hits,
                query_time_ms=query_time_ms,
                granules=granules,
                collection_concept_id=params.collection_concept_id,
                page_info={
                    "page_size": page_size_returned,
                    "page_number": page_number,
                    "total_pages": (total_hits + page_size_returned - 1) // page_size_returned if page_size_returned > 0 else 0
                }
            )

        except Exception as e:
            error_msg = f"CMR granule search failed for {params.collection_concept_id}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _validate_point(self, point: Optional[str]) -> Optional[list]:
        """Validate and parse point coordinates."""
        if not point:
            return None
            
        try:
            coords = [float(x.strip()) for x in point.split(",")]
            if len(coords) != 2:
                raise ValueError("Point must have exactly 2 coordinates (longitude, latitude)")
            
            longitude, latitude = coords
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude must be between -90 and 90")
                
            return coords
        except Exception as e:
            raise ValueError(f"Invalid point format: {e}")

    def _build_search_summary(self, params: CMRGranuleSearchInputSchema) -> str:
        """Build a human-readable summary of search parameters."""
        summary_parts = [f"collection: {params.collection_concept_id}"]
        
        if params.temporal:
            summary_parts.append(f"temporal: {params.temporal}")
        if params.bounding_box:
            summary_parts.append(f"bbox: {params.bounding_box}")
        if params.point:
            summary_parts.append(f"point: {params.point}")
        if params.online_only:
            summary_parts.append("online_only: true")
        if params.cloud_cover:
            summary_parts.append(f"cloud_cover: {params.cloud_cover}")

        return f"CMR granule search: {', '.join(summary_parts)}"

    @classmethod
    def from_params(
        cls,
        collection_concept_id: str,
        temporal: Optional[str] = None,
        bounding_box: Optional[str] = None,
        point: Optional[str] = None,
        online_only: bool = True,
        page_size: int = 50,
        debug: bool = False,
        **kwargs
    ) -> "CMRGranuleSearchTool":
        """
        Convenience constructor for creating tool with common parameters.
        
        Args:
            collection_concept_id: Target collection ID
            temporal: Temporal range constraint
            bounding_box: Spatial bounds constraint
            point: Point location constraint
            online_only: Filter for online accessible data
            page_size: Results per page
            debug: Enable debug logging
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured CMRGranuleSearchTool instance
        """
        config = DataSearchToolConfig(
            page_size=page_size,
            debug=debug,
            **kwargs
        )
        
        return cls(config=config, debug=debug)