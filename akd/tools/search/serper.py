from __future__ import annotations

import asyncio
import os
from typing import Literal
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
        default=20,
        gt=0,
        le=100,
        description="Number of results per API call (max 10 for general search, max 20 for scholar)",
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
        default=int(os.getenv("SERPER_MAX_PAGES", "1")),
        gt=0,
        le=10,
        description="Maximum number of pages to fetch per query. Defaults to 1 to limit credit usage.",
    )

    pre_authenticate: bool = Field(
        default=False,
        description="Whether to validate serper connection on initialization, which will gulp 1 credit. Disable to save credits if needed.",
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

    _category_map = {
        "science": "scholar",
        "general": "search",
        "technology": "search",
    }

    # Maximum results per page for each Serper endpoint
    _max_results_map = {
        "search": 10,
        "scholar": 20,
        "news": 10,
        "images": 10,
        "places": 10,
    }

    # Result key mapping for each Serper endpoint
    _result_key_map = {
        "search": "organic",
        "scholar": "organic",
        "news": "news",
        "images": "images",
        "places": "places",
    }

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
        if self.pre_authenticate:
            self._validate_connection()

    @property
    def headers(self) -> dict:
        """
        Returns the headers to include in API requests.
        """
        return {
            "X-API-KEY": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    def _get_search_endpoint(self, category: str) -> str:
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
        category = category or self.category or ""
        serper_category = self._category_map.get(category.lower(), self.category)
        return urljoin(str(self.base_url), serper_category)

    async def _fetch_serper_results(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: str | None = None,
        page: int = 1,
    ) -> tuple[list[SearchResultItem], dict]:
        """
        Fetches search results from Serper API for a single query.

        Args:
            client: The httpx async client to use for the request
            query: The search query string
            category: Optional category filter (mapped to Serper endpoint)
            page: Page number (1-indexed)

        Returns:
            Tuple of (list of SearchResultItem objects, metadata dict containing extra fields)

        Raises:
            Exception: If the API request fails
        """
        # Get the appropriate endpoint based on category
        endpoint = self._get_search_endpoint(category)

        # Determine max results allowed for this endpoint
        category_for_endpoint = category or self.category or ""
        serper_category = self._category_map.get(category_for_endpoint.lower(), self.category)
        max_allowed = self._max_results_map.get(serper_category, 20)

        # Enforce endpoint-specific limits
        num_to_fetch = min(self.num_per_page, max_allowed)

        # Build request payload
        payload = {
            "q": query,
            "num": num_to_fetch,
            "page": page,
            "gl": self.gl,
            "hl": self.hl,
            "autocorrect": self.autocorrect,
        }

        if self.debug:
            logger.debug(
                f"Fetching Serper results from {endpoint} for query '{query}'. Page: {page}, Num: {num_to_fetch}",
            )

        try:
            response = await client.post(
                endpoint,
                json=payload,
                headers=self.headers,
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

            # Determine which key contains results based on the endpoint being used
            # This logic mirrors _get_search_endpoint to ensure consistency
            category_for_endpoint = category or self.category
            serper_category = self._category_map.get(category_for_endpoint.lower(), self.category) or "scholar"
            result_key = self._result_key_map.get(serper_category, "organic")

            # Extract results using the appropriate key
            raw_results = data.get(result_key, [])

            # Convert to SearchResultItem objects using pop pattern
            search_results = []
            for result in raw_results:
                # Calculate score for this result (before popping fields used in calculation)
                score = result.get("score") or self._calculate_score(result, len(search_results))

                search_results.append(
                    SearchResultItem(
                        url=result.pop("link", None),
                        title=result.pop("title", "Untitled"),
                        content=result.pop("snippet", ""),
                        query=query,
                        pdf_url=result.pop("pdfUrl", None),
                        category=category,
                        published_date=result.pop("date", None),
                        engine="serper",
                        score=result.pop("score", score),  # Pop score or use calculated
                        extra={**result, "page": page},  # Everything else + page goes in extra
                    ),
                )

            # Extract additional metadata fields for the extra field
            # Include everything except the results themselves
            # Exclude all possible result keys from metadata
            excluded_keys = set(self._result_key_map.values())
            metadata = {key: value for key, value in data.items() if key not in excluded_keys}

            if self.debug and metadata:
                logger.debug(f"Captured metadata fields: {list(metadata.keys())}")

            return search_results, metadata

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

    def _process_results(
        self,
        results: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """
        Process and filter search results.

        - Applies score cutoff filtering
        - Removes duplicates by URL
        - Sorts by score (descending)

        Args:
            results: Search result items from Serper API

        Returns:
            Processed and filtered results
        """
        # Filter by score cutoff
        n_orig = len(results)
        filtered_results = [r for r in results if (r.score or 0) >= self.score_cutoff]

        if self.debug and n_orig > len(filtered_results):
            logger.debug(
                f"Filtered {n_orig - len(filtered_results)} results based on score cutoff {self.score_cutoff}",
            )

        # Sort by score (descending)
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.score or 0,
            reverse=True,
        )

        # Remove duplicates by URL while preserving order
        seen_urls = set()
        unique_results = []
        for result in sorted_results:
            if not result.url:
                continue
            url_str = str(result.url)
            if url_str not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url_str)

        return unique_results

    async def _fetch_serper_results_paginated(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: str | None,
        target_results: int,
    ) -> tuple[list[SearchResultItem], dict]:
        """
        Fetches search results across multiple pages to reach target number.

        Args:
            client: The httpx async client to use
            query: The search query
            category: Optional category filter
            target_results: Target number of results to fetch

        Returns:
            Tuple of (list of SearchResultItem objects, aggregated metadata dict)
        """
        all_results = []
        all_page_metadata = []  # Collect metadata from each page
        current_page = 1

        while len(all_results) < target_results and current_page <= self.max_pages:
            try:
                if self.debug:
                    logger.debug(f"Fetching page {current_page} for query: {query}")

                results, metadata = await self._fetch_serper_results(
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
                all_results = self._process_results(all_results)

                # Collect metadata from each page for proper merging
                if metadata:
                    all_page_metadata.append(metadata)

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

        # Merge metadata from all pages using the dedicated merge method
        aggregated_metadata = self._merge_metadata(all_page_metadata)

        return all_results, aggregated_metadata

    def _merge_metadata(self, all_metadata: list[dict]) -> dict:
        """
        Merge metadata from multiple API calls/pages.

        This method implements different merge strategies based on field type:
        - credits: Sum across all API calls to track total credit usage
        - Lists: Extend/aggregate items from all calls
        - Other fields: Keep first occurrence (typically from page 1)

        Args:
            all_metadata: List of metadata dictionaries from each API call/page

        Returns:
            Merged metadata dictionary with aggregated values

        Example:
            >>> metadata1 = {"credits": 1, "searchParameters": {...}, "relatedSearches": ["a"]}
            >>> metadata2 = {"credits": 1, "searchParameters": {...}, "relatedSearches": ["b"]}
            >>> merged = self._merge_metadata([metadata1, metadata2])
            >>> merged["credits"]  # 2 (summed)
            >>> merged["relatedSearches"]  # ["a", "b"] (extended)
        """
        merged_metadata = {}

        if not all_metadata:
            return merged_metadata

        for metadata in all_metadata:
            for key, value in metadata.items():
                if key not in merged_metadata:
                    # First occurrence: just add it
                    merged_metadata[key] = value
                elif key == "credits" and isinstance(value, (int, float)):
                    # Sum credits across all API calls to track total usage
                    merged_metadata[key] += value
                elif isinstance(value, list) and isinstance(merged_metadata[key], list):
                    # Both are lists: extend (aggregate across queries/pages)
                    merged_metadata[key].extend(value)
                # For other non-list fields, keep the first occurrence

        if self.debug and merged_metadata:
            logger.debug(f"Merged metadata keys: {list(merged_metadata.keys())}")
            if "credits" in merged_metadata:
                logger.debug(f"Total credits used: {merged_metadata['credits']}")

        return merged_metadata

    async def _arun_single_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: str | None,
        max_results: int,
    ) -> list[SearchResultItem]:
        """
        Fetch search results for a single query from Serper API.

        Args:
            client: The httpx async client for making HTTP requests.
            query: The search query string.
            category: Optional category filter for the search.
            max_results: Maximum number of results to fetch for this query.

        Returns:
            List of SearchResultItem objects for this query.
        """
        # Fetch results with metadata (metadata not used in this method)
        results, _ = await self._fetch_serper_results_paginated(
            client,
            query,
            category,
            max_results,
        )

        return results

    async def _arun(
        self,
        params: SerperSearchToolInputSchema,
        max_results: int | None = None,
        **kwargs,  # noqa: ARG002
    ) -> SerperSearchToolOutputSchema:
        """
        Override base _arun to preserve Serper-specific metadata handling.

        Serper returns valuable metadata (credits used, related searches, etc.)
        that we want to preserve and merge across queries.

        Args:
            params: Input parameters with queries, category, and max_results
            max_results: Override for maximum results to return
            **kwargs: Additional keyword arguments

        Returns:
            SerperSearchToolOutputSchema with search results and metadata

        Raises:
            ValueError: If API key is missing
            Exception: If API requests fail
        """
        from akd.utils import reciprocal_rank_fusion

        max_results = max_results or params.max_results or self.max_results
        category = params.category or self.category

        # Log queries being sent to Serper
        if self.debug:
            logger.info(f"🔍 SERPER SEARCH QUERIES ({len(params.queries)} total):")
            for i, query in enumerate(params.queries, 1):
                logger.info(f"  {i}. '{query}'")
            logger.info(f"🎯 Target results per query: {max_results} (full RRF)")
            logger.info(f"📂 Category: {category}")
            logger.info(f"🔬 Search type: {self._get_search_endpoint(category)}")

        async with httpx.AsyncClient() as client:
            # Fetch results AND metadata for each query
            tasks = [
                self._fetch_serper_results_paginated(
                    client,
                    query,
                    category,
                    max_results,
                )
                for query in params.queries
            ]
            results_with_metadata = await asyncio.gather(*tasks)

        # Separate results and metadata
        all_results_per_query = []
        all_metadata = []
        for results, metadata in results_with_metadata:
            # Results are already SearchResultItem objects
            all_results_per_query.append(results)
            if metadata:
                all_metadata.append(metadata)

        # Apply RRF fusion
        fused_results = reciprocal_rank_fusion(
            *all_results_per_query,
            key="url",
            normalize=True,
        )

        # Trim to max_results
        final_results = fused_results[:max_results]

        if self.debug:
            logger.debug(f"Returning {len(final_results)} total results after RRF")

        # Merge metadata from all queries
        merged_metadata = self._merge_metadata(all_metadata)
        merged_metadata["total_pages_fetched"] = (
            max([r.extra.get("page", 1) for r in final_results]) if final_results else 0
        )

        return SerperSearchToolOutputSchema(
            results=final_results,
            category=params.category,
            extra=merged_metadata,
        )
