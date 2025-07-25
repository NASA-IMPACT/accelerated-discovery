from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import aiohttp
from loguru import logger
from pydantic.networks import HttpUrl

from akd.structures import SearchResultItem
from .base_search import SearchTool, SearchToolConfig, SearchToolInputSchema, SearchToolOutputSchema


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
    base_url: HttpUrl = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    max_results: int = os.getenv("SEARXNG_MAX_RESULTS", 10)
    engines: List[str] = os.getenv(
        "SEARXNG_ENGINES",
        "google,arxiv,google_scholar",
    ).split(",")
    max_pages: int = int(os.getenv("SEARXNG_MAX_PAGES", 25))
    results_per_page: int = int(os.getenv("SEARXNG_RESULTS_PER_PAGE", 10))
    score_cutoff: float = float(os.getenv("SEARXNG_SCORE_CUTOFF", 0.25))
    debug: bool = False


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

    def __init__(
        self,
        config: SearxNGSearchToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initializes the SearxNGTool.

        Args:
            config (SearxNGSearchToolConfig):
                Configuration for the tool, including
                    - base URL
                    - max results,
                    - engines,
                    - max pages,
                    - results per page,
                    - score cutoff,
                    - and optional title and description overrides.
        """
        config = config or SearxNGSearchToolConfig()
        super().__init__(config, debug)

    @classmethod
    def from_params(
        cls,
        base_url: Optional[HttpUrl] = None,
        max_results: int = 10,
        engines: Optional[List[str]] = None,
        max_pages: int = 5,
        results_per_page: int = 10,
        score_cutoff: float = 0.25,
        debug: bool = False,
    ) -> SearxNGSearchTool:
        base_url = base_url or os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
        engines = engines or os.getenv(
            "SEARXNG_ENGINES",
            "google,arxiv,google_scholar",
        ).split(",")
        config = SearxNGSearchToolConfig(
            base_url=base_url,
            max_results=max_results,
            engines=engines,
            max_pages=max_pages,
            results_per_page=results_per_page,
            score_cutoff=score_cutoff,
            debug=debug,
        )
        return cls(config, debug)

    async def _fetch_search_results(
        self,
        session: aiohttp.ClientSession,
        query: str,
        category: Optional[str] = None,
        page_num: int = 1,
    ) -> List[dict]:
        """
        Fetches search results for a single query asynchronously.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            query (str): The search query.
            category (Optional[str]): The category of the search query.
            page_num (int): The page number to fetch.

        Returns:
            List[dict]: A list of search result dictionaries.

        Raises:
            Exception: If the request to SearxNG fails.
        """
        query_params = {
            "q": query,
            "safesearch": "0",
            "format": "json",
            "language": "en",
            "engines": ",".join(self.engines),
            "pageno": page_num,
        }

        if category:
            query_params["categories"] = category

        async with session.get(
            f"{self.base_url}/search",
            params=query_params,
        ) as response:
            if response.status != 200:
                raise Exception(
                    f"Failed to fetch search results for query '{query}': {response.status} {response.reason}",
                )
            data = await response.json()
            results = data.get("results", [])

            # Add the query to each result
            for result in results:
                result["query"] = query

            return results

    async def _process_results(
        self,
        results: List[dict],
    ) -> List[dict]:
        results = filter(lambda r: r.get("score", 0) >= self.score_cutoff, results)
        sorted_results = sorted(
            results,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in sorted_results:
            if "content" not in result or "title" not in result or "url" not in result:
                continue
            if result["url"] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result["url"])
            if "doi" in result:
                if isinstance(result["doi"], list):
                    result["doi"] = result["doi"][0]
                elif not isinstance(result["doi"], str):
                    result["doi"] = str(result["doi"])
        return unique_results

    async def _fetch_search_results_paginated(
        self,
        session: aiohttp.ClientSession,
        query: str,
        category: Optional[str],
        target_results: int,
    ) -> List[dict]:
        """
        Fetches search results for a single query across multiple pages
        to reach the target number of results.

        Args:
            session (aiohttp.ClientSession):
                The aiohttp session to use for the request.
            query (str):
                The search query.
            category (Optional[str]):
                The category of the search query.
            target_results (int): The target number of results to fetch.

        Returns:
            List[dict]: A list of search result dictionaries.
        """
        all_results = []
        current_page = 1

        while len(all_results) < target_results and current_page <= self.max_pages:
            try:
                if self.debug:
                    logger.debug(f"Fetching page {current_page} for query: {query}")

                results = await self._fetch_search_results(
                    session,
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
                all_results = await self._process_results(all_results)
                current_page += 1

                # Add a short delay to avoid hammering the SearxNG instance
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Error fetching page {current_page} for query '{query}': {str(e)}",
                )
                break

        if self.debug and current_page > 1:
            logger.debug(
                f"Fetched {len(all_results)} results across {current_page - 1} pages for query: {query}",
            )

        return all_results

    async def _arun(
        self,
        params: SearxNGSearchToolInputSchema,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> SearxNGSearchToolOutputSchema:
        """
        Runs the SearxNGTool asynchronously with the given parameters.

        Args:
            params (SearxNGSearchToolInputSchema):
                The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]):
                The maximum number of search results to return.

        Returns:
            SearxNGSearchToolOutputSchema:
                The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearxNG fails.
        """
        max_results = max_results or params.max_results or self.max_results
        multiplier = 1.5
        target_results_per_query = min(
            int((max_results * multiplier) / len(params.queries)),
            self.max_pages * self.results_per_page,  # Don't exceed max possible results
        )
        # Log all queries being sent to SearxNG
        if self.debug:
            logger.info(f"🔍 SearxNG SEARCH QUERIES ({len(params.queries)} total):")
            for i, query in enumerate(params.queries, 1):
                logger.info(f"  {i}. '{query}'")
            logger.info(f"🎯 Target results per query: {target_results_per_query}")
            logger.info(f"📂 Category: {params.category}")
            logger.info(f"🔧 Engines: {self.engines}")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_search_results_paginated(
                    session,
                    query,
                    params.category,
                    target_results_per_query,
                )
                for query in params.queries
            ]
            results = await asyncio.gather(*tasks)

        # Process final time
        # No need to process title here
        # because already done during pagination
        filtered_results = await self._process_results(
            [item for sublist in results for item in sublist],
        )
        filtered_results = filtered_results[:max_results]

        if self.debug:
            logger.debug(filtered_results)

        results = [
            SearchResultItem(
                url=result.pop("url", None),
                pdf_url=result.pop("pdf_url", None),
                title=result.pop("title", None) or "Untitled",  # Ensure title is never None
                content=result.pop("content", None),
                query=result.pop("query", None) or "Unknown query",  # Ensure query is never None
                category=result.pop("category", None),
                doi=result.pop("doi", None),
                published_date=result.pop("publishedDate", None),
                engine=result.pop("engine", None),
                tags=result.pop("tags", None),
                extra=result,  # Remaining keys in result
            )
            for result in filtered_results
        ]
        return SearxNGSearchToolOutputSchema(
            results=results,
            category=params.category,
        )