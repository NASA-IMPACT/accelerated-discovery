import json
import re
from datetime import date
from typing import Dict, List, Optional

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig


class CollectionSearchInputSchema(InputSchema):
    """Input schema for Collection Search agent"""

    dataset_type: str = Field(..., description="Dataset type passed from extraction agent")
    location: str = Field(..., description="Location passed from extraction agent")
    bbox: Optional[List[float]] = Field(
        default=None,
        description="A GeoJSON string representing the bounding box coordinates of the location passed from extraction agent",
    )
    frequency: str = Field(..., description="Frequency passed from extraction agent")
    temporal_extent: Optional[Dict] = Field(
        default={"dates": {"start": "1900-01-01T00:00:00Z", "end": date.today().isoformat()}},
        description="Time interval or dates passed from extraction agent",
    )


class CollectionSearchOutputSchema(OutputSchema):
    """Output schema for Collection Search agent"""

    response: str = Field(
        default=None,
        description="Response from the collection search",
    )
    collections: List[str] = Field(
        ...,
        description="List of collections matching the search.",
    )


class STACSearchAgentConfig(BaseAgentConfig):
    """Config  for Collection Search agent"""

    stac_roots: List[str] = Field(
        default_factory=lambda: [
            "https://earth.gov/ghgcenter/api/stac",
            "https://openveda.cloud/api/stac",
        ],
        description="List of STAC API endpoints to query.",
    )


class CollectionSearchAgent(BaseAgent):
    input_schema = CollectionSearchInputSchema
    output_schema = CollectionSearchOutputSchema
    config_schema = STACSearchAgentConfig

    async def _fetch_items(
        self,
        client: httpx.AsyncClient,
        root: str,
        params: CollectionSearchInputSchema,
    ) -> Dict[str, List[str]]:
        """Fetch all STAC items from a root /search endpoint, returning results grouped by collection URL."""

        results: Dict[str, List[str]] = {}
        base_url = f"{root}/search"

        # Base query (no collections filter)
        base_query = {
            "limit": 100,
        }

        # if params.bbox:
        #     base_query["bbox"] = params.bbox

        if params.temporal_extent and "dates" in params.temporal_extent:
            start = params.temporal_extent["dates"].get("start")
            end = params.temporal_extent["dates"].get("end")
            if start or end:
                base_query["datetime"] = f"{start or ''}/{end or ''}"

        next_url = base_url
        query = base_query

        try:
            while next_url:
                print("in while loop")
                r = await client.get(next_url, params=query)
                r.raise_for_status()
                data = r.json()

                features = data.get("features", [])
                for f in features:
                    # Extract the collection URL from the 'collection' link
                    collection_link = next(
                        (link["href"] for link in f.get("links", []) if link.get("rel") == "collection"),
                        None,
                    )
                    coll_url = collection_link or f"{root}/collections/{f.get('collection', 'unknown')}"

                    # Extract self/item URLs
                    hrefs = [
                        link["href"]
                        for link in f.get("links", [])
                        if link.get("rel") == "self" or link.get("href", "").endswith(".json")
                    ]

                    if hrefs:
                        results.setdefault(coll_url, []).extend(hrefs)

                # Handle pagination
                next_link = next((link for link in data.get("links", []) if link.get("rel") == "next"), None)
                next_url = next_link["href"] if next_link else None
                query = None  # only first call gets query params

        except Exception as e:
            logger.error(f"Failed to fetch items from {root}: {e}")

        return results

    async def _fetch_collections(self, client: httpx.AsyncClient, collection_url):
        r = await client.get(collection_url)
        r.raise_for_status()
        data = r.json()
        return {
            "title": data.get("title"),
            "description": data.get("description", ""),
            "collection_url": collection_url,
        }

    async def _arun(self, params: CollectionSearchInputSchema, **kwargs) -> CollectionSearchOutputSchema:
        return await self.get_response_async(params, **kwargs)

    async def fetch_all_collections(self, client: httpx.AsyncClient) -> List[Dict]:
        """Fetch and paginate collections from all STAC roots."""

        stac_roots = [r.rstrip("/") for r in self.config.stac_roots]

        results = []  # final list

        for root in stac_roots:
            next_url = f"{root}/collections"
            params = None  # first call only

            try:
                while next_url:
                    r = await client.get(next_url, params=params)
                    r.raise_for_status()
                    data = r.json()

                    # Extract collections in this page
                    for c in data.get("collections", []):
                        coll_id = c.get("id")
                        if not coll_id:
                            continue

                        collection_url = f"{root}/collections/{coll_id}"

                        results.append(
                            {
                                "title": c.get("title"),
                                "description": c.get("description", ""),
                                "collection_url": collection_url,
                                "periodicity": c.get("dashboard:time_density", ""),
                            },
                        )

                    # Pagination: find "next" link
                    next_link = next(
                        (link for link in data.get("links", []) if link.get("rel") == "next"),
                        None,
                    )

                    if next_link:
                        next_url = next_link["href"]
                        params = None  # next pages must NOT resend params
                    else:
                        next_url = None

            except Exception as e:
                logger.error(f"Failed to fetch collections from {root}: {e}")

        return results

    async def get_response_async(self, params: CollectionSearchInputSchema, **kwargs) -> CollectionSearchOutputSchema:
        """Filter STAC collections based on extracted parameters."""

        # searching all items and then filtering collection took a lot of time
        # all_items: Dict[str, List[str]] = {}
        # async with httpx.AsyncClient(timeout=60.0) as client:
        #     for root in stac_roots:
        #         items = await self._fetch_items(client, root, params)
        #         # merge items into all_items
        #         for coll_url, hrefs in items.items():
        #             all_items.setdefault(coll_url, []).extend(hrefs)

        #     collection_urls = list(all_items.keys())
        #     all_collections = []

        #     for collection_url in collection_urls:
        #         all_collections.append(await self._fetch_collections(client, collection_url))

        async with httpx.AsyncClient(timeout=30) as client:
            all_collections = await self.fetch_all_collections(client)

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500, api_key=self.config.api_key)
            llm_prompt = f"""
            You are a geospatial data expert.

            The user is looking for datasets matching:
            - dataset_type: {params.dataset_type}
            - periodicity: {params.frequency}

            You have access to several STAC collections across multiple STAC roots.
            Each collection has the following fields:
            - id: unique collection ID
            - title: human-readable name
            - description: text about what it contains
            - root: STAC API root URL
            - collection_url: direct link to the STAC collection

            Here are the available STAC collections:
            {json.dumps(all_collections, ensure_ascii=False, indent=2)}

            ### Task
            Analyze all collections and pick **the top 10 most relevant** to the user's query.

            Return **strictly valid JSON** (nothing else) with this exact structure:

            {{
            "collections": [ "collection1_url", "collection2_url"],
            "reasoning": "2-3 lines explaining why these collections were selected"
            }}

            Rules:
            - Prefer collections that mention the topic (e.g., sulphur) or related chemical species in title/description.
            - If none directly mention it, choose the most conceptually related environmental or emission datasets.
            - Never fabricate or rename collections.
            - Respond with JSON only — no markdown, no extra text.
            """

            response = await llm.ainvoke(llm_prompt)
            raw = response.content.strip()

            # Remove markdown formatting if the LLM wrapped output in ```json ... ```
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*", "", raw)
                raw = raw.replace("```", "").strip()

            data = json.loads(raw)
            return CollectionSearchOutputSchema(response=data["reasoning"], collections=data["collections"])
