from __future__ import annotations

import asyncio
import os
from typing import List, Literal

import httpx
from loguru import logger
from pydantic import Field
from pydantic.networks import HttpUrl

from akd.structures import SearchResultItem

from ._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)


class SearxNGSearchToolInputSchema(SearchToolInputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content.
    """

    pass


class SearxNGSearchToolOutputSchema(SearchToolOutputSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content."""

    pass


class SearxNGSearchToolConfig(SearchToolConfig):
    base_url: HttpUrl = Field(
        default=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
    )
    max_results: int = Field(default=int(os.getenv("SEARXNG_MAX_RESULTS", "10")))
    engines: List[str] = Field(
        default_factory=lambda: os.getenv(
            "SEARXNG_ENGINES",
            "arxiv,google_scholar",
        ).split(","),
    )
    max_pages: int = Field(
        default=int(os.getenv("SEARXNG_MAX_PAGES", "25")),
        gt=0,
        le=100,
    )
    results_per_page: int = Field(
        default=int(os.getenv("SEARXNG_RESULTS_PER_PAGE", "10")),
        gt=0,
        le=100,
    )
    score_cutoff: float = Field(
        default=float(os.getenv("SEARXNG_SCORE_CUTOFF", "0.25")),
        ge=0.0,
        le=1.0,
    )
    strict: bool = Field(
        default=False,
        description="Whether to enforce strict search for filtering engines.",
    )

    safe_search: Literal[0, 1, 2] = Field(
        default=2,
        description="Safe search level: 0 (off), 1 (moderate), 2 (strict).",
    )
    debug: bool = Field(default=False, description="Whether to enable debug mode.")


class SearxNGSearchTool(SearchTool):
    """
    Tool for performing searches on SearxNG based on the provided queries and category.

    Attributes:
        input_schema (SearxNGSearchToolInputSchema): The schema for the input data.
        output_schema (SearxNGSearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        base_url (str): The base URL for the SearxNG instance to use.
    """

    input_schema = SearxNGSearchToolInputSchema
    output_schema = SearxNGSearchToolOutputSchema
    config_schema = SearxNGSearchToolConfig

    async def _fetch_search_results(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: str | None = None,
        page_num: int = 1,
    ) -> list[SearchResultItem]:
        """
        Fetches search results for a single query asynchronously.

        Args:
            client: The httpx async client to use for the request.
            query: The search query.
            category: The category of the search query.
            page_num: The page number to fetch.

        Returns:
            List of SearchResultItem objects.

        Raises:
            Exception: If the request to SearxNG fails.
        """
        query_params = {
            "q": query,
            "safesearch": str(self.safe_search),
            "format": "json",
            "language": "en",
            "engines": ",".join(self.engines),
            "pageno": page_num,
        }

        if category:
            query_params["categories"] = category

        if self.debug:
            logger.debug(
                f"Fetching SearxNG results for query '{query}'. Request params: {query_params}",
            )

        try:
            response = await client.get(
                f"{self.base_url}/search",
                params=query_params,
            )

            if response.status_code != 200:
                logger.error(
                    f"HTTP Error fetching SearxNG results for query '{query}': {response.status_code} {response.reason_phrase}",
                )
                if self.debug:
                    logger.debug(f"Request URL: {response.url}")
                    logger.debug(f"Request Params: {query_params}")
                raise Exception(
                    f"Failed to fetch search results for query '{query}': {response.status_code} {response.reason_phrase}",
                )

            data = response.json()
            raw_results = data.get("results", [])

            # Convert to SearchResultItem objects
            def url_or_none(v: str | None) -> str | None:
                """Return None for empty or whitespace-only strings so AnyUrl accepts it."""
                if v is None or (isinstance(v, str) and not v.strip()):
                    return None
                return v

            search_results = []
            for result in raw_results:
                # Handle DOI normalization
                doi = result.get("doi")
                if doi:
                    if isinstance(doi, list):
                        doi = doi[0] if doi else None
                    elif not isinstance(doi, str):
                        doi = str(doi)
                    result["doi"] = doi

                raw_url = url_or_none(result.pop("url", None))
                if raw_url is None:
                    continue  # skip results with no valid URL (required field)
                search_results.append(
                    SearchResultItem(
                        url=raw_url,
                        title=result.pop("title", "Untitled") or "",
                        content=result.pop("content", "") or "",
                        query=query,
                        pdf_url=url_or_none(result.pop("pdf_url", None)),
                        category=result.pop("category", None),
                        doi=result.pop("doi", None),
                        published_date=result.pop("publishedDate", None),
                        engine=result.pop("engine", None),
                        tags=result.pop("tags", None),
                        authors=result.pop("authors", None),
                        score=result.pop("score", None),
                        extra=result,  # Everything else goes in extra
                    ),
                )

            return search_results
        except Exception as e:
            logger.error(f"Failed to fetch SearxNG results for query '{query}': {e}")
            raise

    def _process_results(
        self,
        results: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """
        Process and filter SearxNG search results.

        Applies score cutoff filtering, strict engine filtering,
        deduplication by URL, and sorting by score.

        Args:
            results: List of search result items to process.

        Returns:
            Processed and filtered list of search result items.
        """
        n_orig = len(results)

        # Filter by score cutoff
        filtered_results = [r for r in results if (r.score or 0) >= self.score_cutoff]

        # Apply strict engine filtering if enabled
        if self.strict and self.engines:
            filtered_results = [
                r
                for r in filtered_results
                if any(self.engine_names_match(engine, r.engine or "") for engine in self.engines)
            ]
            if self.debug:
                logger.debug(
                    f"Filtered {n_orig - len(filtered_results)} results based on strict engine filtering.",
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
            # Skip results missing required fields
            if not result.url or not result.title:
                continue
            url_str = str(result.url)
            if url_str not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url_str)

        return unique_results

    async def _fetch_search_results_paginated(
        self,
        client: httpx.AsyncClient,
        query: str,
        category: str | None,
        target_results: int,
    ) -> list[SearchResultItem]:
        """
        Fetches search results for a single query across multiple pages
        to reach the target number of results.

        Args:
            client: The httpx async client to use for the request.
            query: The search query.
            category: The category of the search query.
            target_results: The target number of results to fetch.

        Returns:
            List of SearchResultItem objects.
        """
        all_results = []
        current_page = 1

        while len(all_results) < target_results and current_page <= self.max_pages:
            try:
                if self.debug:
                    logger.debug(f"Fetching page {current_page} for query: {query}")

                results = await self._fetch_search_results(
                    client,
                    query,
                    category,
                    current_page,
                )
                if self.debug:
                    logger.debug(
                        f"Fetched {len(results)} results for page {current_page}",
                    )

                # Add and process so that the final list will be better
                all_results.extend(results)
                all_results = self._process_results(all_results)
                current_page += 1

                # Add a short delay to avoid hammering the SearxNG instance
                await asyncio.sleep(0.1)

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

    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> SearxNGSearchToolOutputSchema:
        """
        Fetch search results for a single query from SearxNG.

        This method creates its own HTTP client and uses the existing
        pagination logic to fetch results. Each query execution is independent.

        Args:
            query: The search query string.
            max_results: Maximum number of results to fetch for this query.
            **kwargs: Additional parameters including:
                - category (str | None): Optional category filter for the search

        Returns:
            SearxNGSearchToolOutputSchema with results and metadata for this query.
        """
        category = kwargs.get("category")

        # Create client per query (search I/O dominates client creation overhead)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            results = await self._fetch_search_results_paginated(
                client,
                query,
                category,
                max_results,
            )

        results = self._process_results(results)
        return self.output_schema(
            results=results,
        )

    @staticmethod
    def normalize_engine_name(name: str) -> str:
        """Convert engine name to standardized format for comparison."""
        return name.lower().replace(" ", "_")

    @staticmethod
    def engine_names_match(configured_name: str, result_engine: str) -> bool:
        """Check if configured engine name matches result engine name."""
        return SearxNGSearchTool.normalize_engine_name(
            configured_name,
        ) == SearxNGSearchTool.normalize_engine_name(result_engine)
