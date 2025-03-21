from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import aiohttp
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from loguru import logger
from pydantic.fields import Field
from pydantic.networks import HttpUrl

from ..structures import SearchResultItem


class SearchToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content.
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "science", "technology"]] = Field(
        "science",
        description="Category of the search queries.",
    )


class SearchToolOutputSchema(BaseIOSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content."""

    results: List[SearchResultItem] = Field(
        ...,
        description="List of search result items",
    )
    category: Optional[str] = Field(
        None,
        description="The category of the search results",
    )


class SearchTool(BaseTool):
    """
    Tool for performing searches on SearxNG based on the provided queries and category.

    Attributes:
        input_schema (SearchToolInputSchema): The schema for the input data.
        output_schema (SearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        base_url (str): The base URL for the SearxNG instance to use.
    """

    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema


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


class SearxNGSearchToolConfig(BaseToolConfig):
    base_url: HttpUrl = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    max_results: int = os.getenv("SEARXNG_MAX_RESULTS", 10)
    engines: List[str] = os.getenv(
        "SEARXNG_ENGINES",
        "google,arxiv,google_scholar",
    ).split(",")
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

    def __init__(self, config: Optional[SearxNGSearchToolConfig] = None):
        """
        Initializes the SearxNGTool.

        Args:
            config (SearxNGSearchToolConfig):
                Configuration for the tool, including
                    - base URL
                    - max results,
                    - and optional title and description overrides.
        """
        config = config or SearxNGSearchToolConfig()
        super().__init__(config)
        self.base_url = config.base_url
        self.max_results = config.max_results
        self.engines = config.engines
        self.debug = config.debug

    @classmethod
    def from_params(
        cls,
        base_url: Optional[HttpUrl] = None,
        max_results: int = 10,
        engines: Optional[List[str]] = None,
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
            debug=debug,
        )
        return cls(config)

    async def _fetch_search_results(
        self,
        session: aiohttp.ClientSession,
        query: str,
        category: Optional[str],
    ) -> List[dict]:
        """
        Fetches search results for a single query asynchronously.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            query (str): The search query.
            category (Optional[str]): The category of the search query.

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

    async def run_async(
        self,
        params: SearxNGSearchToolInputSchema,
        max_results: Optional[int] = None,
    ) -> SearxNGSearchToolOutputSchema:
        """
        Runs the SearxNGTool asynchronously with the given parameters.

        Args:
            params (SearxNGSearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]): The maximum number of search results to return.

        Returns:
            SearxNGSearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearxNG fails.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_search_results(session, query, params.category)
                for query in params.queries
            ]
            results = await asyncio.gather(*tasks)

        all_results = [item for sublist in results for item in sublist]

        # Sort the combined results by score in descending order
        sorted_results = sorted(
            all_results,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in sorted_results:
            if (
                "content" not in result
                or "title" not in result
                or "url" not in result
                or "query" not in result
            ):
                continue
            if result["url"] not in seen_urls:
                unique_results.append(result)
                if "metadata" in result:
                    result[
                        "title"
                    ] = f"{result['title']} - (Published {result['metadata']})"
                if "publishedDate" in result and result["publishedDate"]:
                    result[
                        "title"
                    ] = f"{result['title']} - (Published {result['publishedDate']})"
                seen_urls.add(result["url"])
            if "doi" in result and isinstance(result["doi"], list):
                result["doi"] = result["doi"][0]

        # Filter results to include only those with the correct category if it is set
        filtered_results = (
            list(filter(lambda r: r.get("category") == params.category, unique_results))
            if params.category
            else unique_results
        )

        filtered_results = filtered_results[: max_results or self.max_results]

        if self.debug:
            logger.debug(filtered_results)
            # print(filtered_results)

        results = [
            SearchResultItem(
                url=result.pop("url", None),
                pdf_url=result.pop("pdf_url", None),
                title=result.pop("title", None),
                content=result.pop("content", None),
                query=result.pop("query", None),
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

    def run(
        self,
        params: SearxNGSearchToolInputSchema,
        max_results: Optional[int] = None,
    ) -> SearxNGSearchToolOutputSchema:
        """
        Runs the SearxNGTool synchronously with the given parameters.

        This method creates an event loop in a separate thread to run the asynchronous operations.

        Args:
            params (SearxNGSearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]): The maximum number of search results to return.

        Returns:
            SearxNGSearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearxNG fails.
        """
        with ThreadPoolExecutor() as executor:
            return executor.submit(
                asyncio.run,
                self.run_async(params, max_results),
            ).result()
