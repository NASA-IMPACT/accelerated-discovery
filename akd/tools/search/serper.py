from __future__ import annotations

import asyncio
import os
from typing import List, Literal, Optional
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import Field, SecretStr
from pydantic.networks import HttpUrl

from akd.structures import SearchResultItem

from ._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)


class SerperSearchToolInputSchema(SearchToolInputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content using Serper API.
    """

    pass


class SerperSearchToolOutputSchema(SearchToolOutputSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content using Serper API."""

    pass


class SerperSearchToolConfig(SearchToolConfig):
    """Configuration for SerperSearchTool."""

    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("SERPER_API_KEY", "")),
        description="Serper API key for authentication",
    )
    base_url: HttpUrl = Field(
        default=os.getenv("SERPER_BASE_URL", "https://google.serper.dev"),
        description="Base URL for Serper API (without endpoint path)",
    )
    category: Literal["search", "scholar", "news", "images", "places"] = Field(
        default=os.getenv("SERPER_CATEGORY", "scholar"),
        description="Default Serper category/endpoint: 'search' (general Google), 'scholar' (Google Scholar), 'news', 'images', 'places'",
    )
    max_results: int = Field(
        default=int(os.getenv("SERPER_MAX_RESULTS", "10")),
        description="Maximum number of search results to return",
    )
    num_per_page: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Number of results per API call (max 100)",
    )
    score_cutoff: float = Field(
        default=float(os.getenv("SERPER_SCORE_CUTOFF", "0.0")),
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for filtering results",
    )
    gl: str = Field(
        default="us",
        description="Country code for localized results (e.g., 'us', 'uk')",
    )
    hl: str = Field(
        default="en",
        description="Language code for results (e.g., 'en', 'es')",
    )
    autocorrect: bool = Field(
        default=True,
        description="Whether to enable query autocorrection",
    )
    max_pages: int = Field(
        default=int(os.getenv("SERPER_MAX_PAGES", "5")),
        gt=0,
        le=10,
        description="Maximum number of pages to fetch per query",
    )
    debug: bool = Field(
        default=False,
        description="Whether to enable debug mode",
    )


class SerperSearchTool(SearchTool):
    """
    Tool for performing searches using Serper API (Google Search API alternative).

    Serper provides fast, reliable Google Search results through a simple REST API.
    This tool implements the same interface as SearxNGSearchTool for drop-in replacement.

    Attributes:
        input_schema (SerperSearchToolInputSchema): The schema for the input data.
        output_schema (SerperSearchToolOutputSchema): The schema for the output data.
        config_schema (SerperSearchToolConfig): Configuration schema for the tool.
    """

    input_schema = SerperSearchToolInputSchema
    output_schema = SerperSearchToolOutputSchema
    config_schema = SerperSearchToolConfig

    def __init__(
        self,
        config: SerperSearchToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initializes the SerperSearchTool.

        Args:
            config (SerperSearchToolConfig): Configuration for the tool, including
                API key, max results, score cutoff, and other parameters.
            debug (bool): Enable debug logging.
        """
        config = config or SerperSearchToolConfig()
        super().__init__(config, debug)

        # Validate API key
        if not self.api_key or not self.api_key.get_secret_value():
            raise ValueError(
                "SERPER_API_KEY environment variable must be set or provided in config",
            )

    def _validate_connection(self) -> None:
        """
        Validates the Serper API connection and authentication.

        Performs a minimal POST request with a test query to verify the API key is valid.

        Raises:
            RuntimeError: If authentication fails or connection cannot be established.
        """
        endpoint = urljoin(str(self.base_url), "search")
        headers = {
            "X-API-KEY": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        # Minimal test payload
        payload = {"q": "test", "num": 1}

        try:
            response = httpx.post(endpoint, json=payload, headers=headers, timeout=5.0)
            if response.status_code not in range(200, 300):
                if response.status_code in (401, 403):
                    raise RuntimeError("Invalid or unauthorized SERPER_API_KEY")
                raise RuntimeError(
                    f"Serper API connection test failed: {response.status_code} {response.reason_phrase}",
                )
        except httpx.RequestError as e:
            raise RuntimeError(f"Cannot connect to Serper API: {e}") from e

    def _post_init(self) -> None:
        """Post-initialization hook to validate API connection."""
        super()._post_init()
        self._validate_connection()

    def _get_search_endpoint(self, input_category: Optional[str] = None) -> str:
        """
        Determine the Serper API endpoint based on input category or default category.

        Input category mapping (from SearchToolInputSchema):
        - "science" -> scholar (Google Scholar for academic papers)
        - "general" -> search (general Google search)
        - "technology" -> search (general Google search)
        - None -> uses configured default category

        Args:
            input_category: Optional category from input parameters

        Returns:
            Full endpoint URL for Serper API
        """
        # Map input category to Serper endpoint if provided
        if input_category:
            category_map = {
                "science": "scholar",
                "general": "search",
                "technology": "search",
            }
            serper_category = category_map.get(input_category.lower(), self.category)
        else:
            serper_category = self.category

        return urljoin(str(self.base_url), serper_category)

    async def _fetch_serper_results(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: Optional[str] = None,
        page: int = 1,
    ) -> List[dict]:
        """
        Fetches search results from Serper API for a single query.

        Args:
            client: The httpx async client to use for the request
            query: The search query string
            category: Optional category filter (mapped to Serper endpoint)
            page: Page number (1-indexed)

        Returns:
            List of search result dictionaries from Serper API

        Raises:
            Exception: If the API request fails
        """
        # Get the appropriate endpoint based on category
        endpoint = self._get_search_endpoint(category)

        # Build request payload
        payload = {
            "q": query,
            "num": self.num_per_page,
            "page": page,
            "gl": self.gl,
            "hl": self.hl,
            "autocorrect": self.autocorrect,
        }

        headers = {
            "X-API-KEY": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        if self.debug:
            logger.debug(
                f"Fetching Serper results from {endpoint} for query '{query}'. Page: {page}, Num: {self.num_per_page}",
            )

        try:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    f"HTTP Error fetching Serper results for query '{query}': "
                    f"{response.status_code} {response.reason_phrase}\n{error_text}",
                )
                raise Exception(
                    f"Failed to fetch search results for query '{query}': {response.status_code} {response.reason_phrase}",
                )

            data = response.json()

            # Results key varies by endpoint:
            # - /search: "organic"
            # - /scholar: "organic"
            # - /news: "news"
            # - /images: "images"
            results = data.get("organic", data.get("news", data.get("images", [])))

            # Add query and category to each result for consistency with SearxNG
            for result in results:
                result["query"] = query
                if category:
                    result["category"] = category

            # Add search metadata if available
            if "searchParameters" in data:
                search_params = data["searchParameters"]
                if self.debug:
                    logger.debug(f"Search parameters: {search_params}")

            return results

        except httpx.RequestError as e:
            logger.error(f"Network error fetching Serper results for query '{query}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Serper results for query '{query}': {e}")
            raise

    def _calculate_score(self, result: dict, position: int) -> float:
        """
        Calculate a normalized score for a search result.

        Serper doesn't provide explicit scores, so we calculate based on:
        - Position in results (higher is better)
        - Presence of knowledge graph data
        - Result metadata quality

        Args:
            result: The result dictionary from Serper
            position: Position in the result list (0-indexed)

        Returns:
            Normalized score between 0 and 1
        """
        # Base score from position (inverse rank)
        # First result gets highest score, decaying logarithmically
        base_score = 1.0 / (1.0 + position * 0.1)

        # Bonus for rich metadata
        bonus = 0.0
        if result.get("snippet"):
            bonus += 0.05
        if result.get("sitelinks"):
            bonus += 0.05
        if result.get("date"):
            bonus += 0.03

        return min(base_score + bonus, 1.0)

    async def _process_results(
        self,
        results: List[dict],
    ) -> List[dict]:
        """
        Process and filter search results.

        - Applies score cutoff filtering
        - Removes duplicates by URL
        - Sorts by score (descending)

        Args:
            results: Raw results from Serper API

        Returns:
            Processed and filtered results
        """
        # Add scores if not present
        for i, result in enumerate(results):
            if "score" not in result:
                result["score"] = self._calculate_score(result, i)

        # Filter by score cutoff
        n_orig = len(results)
        results = [r for r in results if r.get("score", 0) >= self.score_cutoff]

        if self.debug and n_orig > len(results):
            logger.debug(
                f"Filtered {n_orig - len(results)} results based on score cutoff {self.score_cutoff}",
            )

        # Sort by score (descending)
        sorted_results = sorted(
            results,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in sorted_results:
            url = result.get("link")
            if url and url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)

        return unique_results

    async def _fetch_serper_results_paginated(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: Optional[str],
        target_results: int,
    ) -> List[dict]:
        """
        Fetches search results across multiple pages to reach target number.

        Args:
            client: The httpx async client to use
            query: The search query
            category: Optional category filter
            target_results: Target number of results to fetch

        Returns:
            List of search result dictionaries
        """
        all_results = []
        current_page = 1

        while len(all_results) < target_results and current_page <= self.max_pages:
            try:
                if self.debug:
                    logger.debug(f"Fetching page {current_page} for query: {query}")

                results = await self._fetch_serper_results(
                    client,
                    query,
                    category,
                    current_page,
                )

                if self.debug:
                    logger.debug(
                        f"Fetched {len(results)} results for page {current_page}",
                    )

                if not results:
                    # No more results available
                    break

                all_results.extend(results)
                all_results = await self._process_results(all_results)
                current_page += 1

                # Rate limiting: small delay between requests
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(
                    f"Error fetching page {current_page} for query '{query}': {e}",
                )
                break

        if self.debug and current_page > 1:
            logger.debug(
                f"Fetched {len(all_results)} results across {current_page - 1} pages for query: {query}",
            )

        return all_results

    async def _arun(
        self,
        params: SerperSearchToolInputSchema,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> SerperSearchToolOutputSchema:
        """
        Runs the SerperSearchTool asynchronously with the given parameters.

        This method implements the same interface as SearxNGSearchTool for
        drop-in compatibility.

        Args:
            params: Input parameters with queries, category, and max_results
            max_results: Override for maximum results to return
            **kwargs: Additional keyword arguments

        Returns:
            SerperSearchToolOutputSchema with search results

        Raises:
            ValueError: If API key is missing
            Exception: If API requests fail
        """
        max_results = max_results or params.max_results or self.max_results

        # Calculate target results per query (with multiplier for filtering)
        multiplier = 1.5
        target_results_per_query = min(
            int((max_results * multiplier) / len(params.queries)),
            self.max_pages * self.num_per_page,  # Don't exceed max possible results
        )

        # Log queries being sent to Serper
        if self.debug:
            logger.info(f"🔍 SERPER SEARCH QUERIES ({len(params.queries)} total):")
            for i, query in enumerate(params.queries, 1):
                logger.info(f"  {i}. '{query}'")
            logger.info(f"🎯 Target results per query: {target_results_per_query}")
            logger.info(f"📂 Category: {params.category}")
            logger.info(f"🔬 Search type: {self._get_search_endpoint(params.category)}")

        async with httpx.AsyncClient() as client:
            tasks = [
                self._fetch_serper_results_paginated(
                    client,
                    query,
                    params.category,
                    target_results_per_query,
                )
                for query in params.queries
            ]
            results = await asyncio.gather(*tasks)

        # Flatten and process final results
        flat_results = [item for sublist in results for item in sublist]
        filtered_results = await self._process_results(flat_results)
        filtered_results = filtered_results[:max_results]

        if self.debug:
            logger.debug(f"Returning {len(filtered_results)} total results")

        # Transform to SearchResultItem format
        search_results = [
            SearchResultItem(
                url=result.get("link"),
                title=result.get("title") or "Untitled",
                content=result.get("snippet", ""),
                query=result.get("query") or "Unknown query",
                category=result.get("category"),
                published_date=result.get("date"),
                engine="serper",
                score=result.get("score"),
                extra={
                    k: v
                    for k, v in result.items()
                    if k not in ["link", "title", "snippet", "query", "category", "date", "score"]
                },
            )
            for result in filtered_results
        ]

        return SerperSearchToolOutputSchema(
            results=search_results,
            category=params.category,
        )
