from datetime import date
from typing import Dict, List, Optional

import httpx
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent
from akd.agents.eie_agents.collection_search import STACSearchAgentConfig


class ItemSearchInputSchema(InputSchema):
    """Input schema for Item Search agent"""

    bbox: Optional[List[float]] = Field(
        default=None,
        description="A GeoJSON string representing the bounding box coordinates of the location passed from extraction agent",
    )
    temporal_extent: Optional[Dict] = Field(
        default={"dates": {"start": "1900-01-01T00:00:00Z", "end": date.today().isoformat()}},
        description="Time interval or dates passed from extraction agent",
    )
    collections: List[str] = Field(
        ...,
        description="List of collections to search for items.",
    )


class ItemSearchOutputSchema(OutputSchema):
    """Output schema for Item Search agent"""

    items: Dict = Field(
        ...,
        description="List of items matching the search.",
    )


class ItemSearchAgent(BaseAgent):
    input_schema = ItemSearchInputSchema
    output_schema = ItemSearchOutputSchema
    config_schema = STACSearchAgentConfig

    async def _fetch_items(
        self,
        client: httpx.AsyncClient,
        root: str,
        collections: List[str],
        params: ItemSearchInputSchema,
    ) -> Dict[str, List[str]]:
        """Fetch all STAC items from a root /search endpoint for given collections, following pagination."""
        results: Dict[str, List[str]] = {}
        base_url = f"{root}/search"

        # Base query
        base_query = {
            "collections": ",".join(collections),
            "limit": 100,
        }

        if params.bbox:
            base_query["bbox"] = ",".join(str(x) for x in params.bbox)

        if params.temporal_extent and "dates" in params.temporal_extent:
            start = params.temporal_extent["dates"].get("start")
            end = params.temporal_extent["dates"].get("end")
            if start or end:
                base_query["datetime"] = f"{start or ''}/{end or ''}"

        next_url = base_url
        query = base_query

        try:
            while next_url:
                r = await client.get(next_url, params=query)
                r.raise_for_status()
                data = r.json()

                features = data.get("features", [])
                for f in features:
                    coll = f.get("collection", "unknown")
                    hrefs = [
                        link["href"]
                        for link in f.get("links", [])
                        if link.get("rel") == "self" or link.get("href", "").endswith(".json")
                    ]
                    if hrefs:
                        results.setdefault(coll, []).extend(hrefs)

                # pagination
                next_link = next((link for link in data.get("links", []) if link.get("rel") == "next"), None)
                next_url = next_link["href"] if next_link else None
                query = None  # only first call gets query params

        except Exception as e:
            logger.error(f"Failed to fetch items from {root}: {e}")

        return results

    async def _arun(self, params: ItemSearchInputSchema, **kwargs) -> ItemSearchOutputSchema:
        return await self.get_response_async(params, **kwargs)

    async def get_response_async(self, params: ItemSearchInputSchema, **kwargs) -> ItemSearchOutputSchema:
        """Filter STAC collections based on extracted parameters."""
        stac_roots = [r.rstrip("/") for r in self.config.stac_roots]
        # group collections by STAC root prefix
        grouped: Dict[str, List[str]] = {}
        for c in params.collections:
            match = next((r for r in stac_roots if c.startswith(r)), None)
            if match:
                grouped.setdefault(match, []).append(c.split("/")[-1])
            else:
                logger.warning(f"No matching STAC root found for collection {c}")

        final_results: Dict[str, List[str]] = {}

        async with httpx.AsyncClient(timeout=60.0) as client:
            for root, coll_list in grouped.items():
                logger.info(f"Fetching items from root {root} for collections {coll_list}")
                items = await self._fetch_items(client, root, coll_list, params)
                for k, v in items.items():
                    final_results.setdefault(k, []).extend(v[:5])

        return ItemSearchOutputSchema(items=final_results)
