from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin

import aiohttp
from loguru import logger
from pydantic import BaseModel, SecretStr, field_validator
from pydantic.fields import Field
from pydantic.networks import HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.agents.query import (
    FollowUpQueryAgent,
    FollowUpQueryAgentInputSchema,
    FollowUpQueryAgentOutputSchema,
    QueryAgent,
    QueryAgentInputSchema,
    QueryAgentOutputSchema,
)
from akd.agents.relevancy import (
    ContentDepthLabel,
    EvidenceQualityLabel,
    MethodologicalRelevanceLabel,
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
    MultiRubricRelevancyOutputSchema,
    RecencyRelevanceLabel,
    ScopeRelevanceLabel,
    TopicAlignmentLabel,
)
from akd.structures import PaperDataItem, SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig


class QueryFocusStrategy(str, Enum):
    """Query adaptation strategies based on rubric analysis."""

    REFINE_TOPIC_SPECIFICITY = "refine_topic_specificity"
    SEARCH_COMPREHENSIVE_REVIEWS = "search_comprehensive_reviews"
    TARGET_PEER_REVIEWED_SOURCES = "target_peer_reviewed_sources"
    SEARCH_METHODOLOGICAL_PAPERS = "search_methodological_papers"
    ADD_RECENT_YEAR_FILTERS = "add_recent_year_filters"
    ADJUST_QUERY_SCOPE = "adjust_query_scope"


class SearchToolInputSchema(InputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content.
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "science", "technology"]] = Field(
        "science",
        description="Category of the search queries.",
    )
    max_results: int = Field(
        10,
        description="Maximum number of search results to return.",
    )


class SearchToolOutputSchema(OutputSchema):
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


class SearchTool(BaseTool[SearchToolInputSchema, SearchToolOutputSchema]):
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


class SemanticScholarSearchToolInputSchema(SearchToolInputSchema):
    """
    Schema for input to a tool for searching for information,
    news, references, and other content using Semantic Scholar
    """

    pass


class SemanticScholarSearchToolOutputSchema(SearchToolOutputSchema):
    """Schema for output of a tool for searching for information,
    news, references, and other content using Semantic Scholar."""

    pass


class SemanticScholarSearchToolConfig(BaseToolConfig):
    """Configuration for the Semantic Scholar Search Tool."""

    api_key: Optional[str] = Field(
        default=(os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")),
    )
    base_url: HttpUrl = Field(default="https://api.semanticscholar.org")
    max_results: int = Field(default=int(os.getenv("SEMANTIC_SCHOLAR_MAX_RESULTS", 10)))
    fields: List[str] = Field(
        default_factory=lambda: [
            "paperId",
            "externalIds",
            "url",
            "title",
            "abstract",
            "year",
            "authors.name",
            "isOpenAccess",
            "openAccessPdf",
        ],
    )
    # Semantic Scholar API limits per request
    results_per_page: int = Field(
        default=int(os.getenv("SEMANTIC_SCHOLAR_RESULTS_PER_PAGE", 10)),
        le=100,
        gt=0,
    )
    # Maximum number of *pages* to fetch per query to avoid excessive requests
    max_pages_per_query: int = Field(
        default=int(
            os.getenv(
                "SEMANTIC_SCHOLAR_MAX_PAGES",
                10,
            ),
        ),
    )
    debug: bool = False

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v):
        if not v:
            logger.warning(
                "Semantic Scholar API key not provided. Rate limits may apply.",
            )
            return None
        if isinstance(v, SecretStr):
            return v.get_secret_value()
        return v


class SemanticScholarSearchTool(
    BaseTool[
        SemanticScholarSearchToolInputSchema,
        SemanticScholarSearchToolOutputSchema,
    ],
):
    """
    Tool for performing searches on Semantic Scholar based on provided queries.
    """

    input_schema = SemanticScholarSearchToolInputSchema
    output_schema = SemanticScholarSearchToolOutputSchema

    config_schema = SemanticScholarSearchToolConfig

    @classmethod
    def from_params(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[HttpUrl] = "https://api.semanticscholar.org",
        max_results: int = 10,
        fields: Optional[List[str]] = None,
        results_per_page: int = 100,
        max_pages_per_query: int = 5,
        debug: bool = False,
    ) -> SemanticScholarSearchTool:
        """Creates an instance from specific parameters."""
        config_data = {
            "api_key": api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "base_url": base_url or "https://api.semanticscholar.org",
            "max_results": max_results,
            "results_per_page": results_per_page,
            "max_pages_per_query": max_pages_per_query,
            "debug": debug,
        }
        if fields:
            config_data["fields"] = fields

        config = SemanticScholarSearchToolConfig(**config_data)
        return cls(config, debug)

    async def _fetch_search_page(
        self,
        session: aiohttp.ClientSession,
        query: str,
        offset: int,
        limit: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches a single page of search results from Semantic Scholar.

        Args:
            session: The aiohttp session.
            query: The search query.
            offset: The starting offset for results.
            limit: The number of results to fetch for this page.

        Returns:
            The JSON response dictionary from the API or None if an error occurs.
        """
        search_url = urljoin(self.config.base_url, "graph/v1/paper/search")
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": ",".join(self.config.fields),
        }
        headers = {}
        if self.config.api_key:
            api_key_value = self.config.api_key
            if api_key_value:
                headers["x-api-key"] = api_key_value

        if self.debug:
            logger.debug(
                f"Fetching Semantic Scholar: query='{query}', offset={offset}, limit={limit}",
            )
            if headers:
                logger.debug("Using API Key.")

        try:
            async with session.get(
                search_url,
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()  # Raise exception for 4xx or 5xx errors
                data = await response.json()
                if self.debug:
                    logger.debug(data)
                    logger.debug(f"API Response Status: {response.status}")
                    # Avoid logging full data if it's too large or sensitive
                    logger.debug(
                        f"Received {len(data.get('data', []))} items. Total: {data.get('total')}, Offset: {data.get('offset')}, Next: {data.get('next')}",
                    )
                return data
        except aiohttp.ClientResponseError as e:
            logger.error(
                f"HTTP Error fetching Semantic Scholar for query '{query}': {e.status} {e.message}",
            )
            # Log request details that caused the error
            logger.error(f"Request URL: {response.url}")
            logger.error(f"Request Params: {params}")
            logger.error(f"Response Headers: {response.headers}")
            try:
                error_body = await response.text()
                logger.error(
                    f"Response Body: {error_body[:500]}",
                )  # Log part of the body
            except Exception as read_err:
                logger.error(f"Could not read error response body: {read_err}")

        except Exception as e:
            logger.error(
                f"Failed to fetch Semantic Scholar results for query '{query}': {e}",
            )

        return None

    def _parse_paper(
        self,
        item: Dict[str, Any],
        doi: Optional[str],
    ) -> Optional[PaperDataItem]:
        """Parses a single paper from the Semantic Scholar API response."""
        if not item or not item.get("paperId"):
            return None
        try:
            paper_item = PaperDataItem(
                paper_id=item.get("paperId"),
                corpus_id=item.get("corpusId"),
                external_ids=item.get("externalIds"),
                url=item.get("url"),
                title=item.get("title"),
                abstract=item.get("abstract"),
                venue=item.get("venue"),
                publication_venue=item.get("publicationVenue"),
                year=item.get("year"),
                reference_count=item.get("referenceCount"),
                citation_count=item.get("citationCount"),
                influential_citation_count=item.get("influentialCitationCount"),
                is_open_access=item.get("isOpenAccess"),
                open_access_pdf=item.get("openAccessPdf"),
                fields_of_study=item.get("fieldsOfStudy"),
                s2_fields_of_study=item.get("s2FieldsOfStudy"),
                publication_types=item.get("publicationTypes"),
                publication_date=item.get("publicationDate"),
                journal=item.get("journal"),
                citation_styles=item.get("citationStyles"),
                authors=item.get("authors"),
                citations=item.get("citations"),
                references=item.get("references"),
                embedding=item.get("embedding"),
                tldr=item.get("tldr"),
                doi=doi or None,
            )
            if self.debug:
                logger.debug(
                    f"Processed paper with DOI {doi}",
                )
            return paper_item
        except Exception as e:
            logger.debug(
                f"Could not parse response to paper object for doi {doi}: {str(e)}",
            )

    async def _fetch_paper_by_doi(
        self,
        session: aiohttp.ClientSession,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches a single page of search results from Semantic Scholar.

        Args:
            session: The aiohttp session.
            query: The search query.
            offset: The starting offset for results.
            limit: The number of results to fetch for this page.

        Returns:
            The JSON response dictionary from the API or None if an error occurs.
        """
        search_url = urljoin(self.config.base_url, "graph/v1/paper/search")
        params = {
            "fields": ",".join(self.config.fields),
        }
        headers = {}
        if self.config.api_key:
            api_key_value = self.config.api_key
            if api_key_value:
                headers["x-api-key"] = api_key_value

        if self.debug:
            logger.debug(
                f"Fetching paper details via Semantic Scholar: query='{query}'",
            )
            if headers:
                logger.debug("Using API Key.")

        try:
            async with session.get(
                search_url,
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()  # Raise exception for 4xx or 5xx errors
                data = await response.json()
                if self.debug:
                    logger.debug(data)
                    logger.debug(f"API Response Status: {response.status}")
                    # Avoid logging full data if it's too large or sensitive
                    logger.debug(
                        f"Received {len(data.get('data', []))} items. Total: {data.get('total')}, Offset: {data.get('offset')}, Next: {data.get('next')}",
                    )
                return [self._parse_paper(item=data, doi=query)]

        except aiohttp.ClientResponseError as e:
            logger.error(
                f"HTTP Error fetching Semantic Scholar for query '{query}': {e.status} {e.message}",
            )
            # Log request details that caused the error
            logger.error(f"Request URL: {response.url}")
            logger.error(f"Request Params: {params}")
            logger.error(f"Response Headers: {response.headers}")
            try:
                error_body = await response.text()
                logger.error(
                    f"Response Body: {error_body[:500]}",
                )  # Log part of the body
            except Exception as read_err:
                logger.error(f"Could not read error response body: {read_err}")

        except Exception as e:
            logger.error(
                f"Failed to fetch Semantic Scholar results for query '{query}': {e}",
            )

        return []

    def _parse_result(
        self,
        item: Dict[str, Any],
        query: str,
        category: Optional[str],
    ) -> Optional[SearchResultItem]:
        """Parses a single item from the Semantic Scholar API response."""
        if (
            not item
            or not item.get("paperId")
            or not item.get("title")
            or not item.get("url")
        ):
            return None  # Skip incomplete results

        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")  # All papers do not have a DOI

        # Extract author names if requested and available
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if author.get("name")
        ]

        # Basic PDF URL check (often requires specific field request like 'openAccessPdf')
        pdf_url = None
        if (
            "openAccessPdf" in item
            and item["openAccessPdf"]
            and isinstance(item["openAccessPdf"], dict)
        ):
            pdf_url = item["openAccessPdf"].get("url") or None

        return SearchResultItem(
            url=item.pop("url"),
            pdf_url=pdf_url,
            title=item.pop("title"),
            content=item.pop("abstract"),  # Map abstract to content
            query=query,
            category=category,
            doi=doi,
            published_date=str(item.pop("year")) if item.get("year") else None,
            engine="semanticscholar",
            tags=authors if authors else None,
            extra=item,
        )

    async def _fetch_search_results_paginated(
        self,
        session: aiohttp.ClientSession,
        query: str,
        category: Optional[str],
        target_results_for_query: int,  # How many results to aim for *this specific query* before deduplication
    ) -> List[SearchResultItem]:
        """
        Fetches and paginates search results for a single query up to a target number
        or API limits.
        """
        all_parsed_results: List[SearchResultItem] = []
        seen_paper_ids = set()
        current_offset = 0
        pages_fetched = 0
        total_available = None  # Keep track of total results reported by API

        while pages_fetched < self.config.max_pages_per_query:
            # Check if we might have enough *raw* results before fetching more
            # This is an estimate; final count depends on parsing and deduplication
            if len(all_parsed_results) >= target_results_for_query:
                if self.debug:
                    logger.debug(
                        f"Query '{query}': Reached estimated target {target_results_for_query}, "
                        f"stopping pagination.",
                    )
                break

            # Check if we've potentially exhausted all results based on the last API response
            if total_available is not None and current_offset >= total_available:
                if self.debug:
                    logger.debug(
                        f"Query '{query}': Current offset {current_offset} >= total "
                        f"available {total_available}, stopping pagination.",
                    )
                break

            limit = min(
                self.config.results_per_page,
                target_results_for_query - len(all_parsed_results),
            )  # Request fewer if close to target
            limit = max(limit, 1)  # Ensure limit is at least 1

            data = await self._fetch_search_page(
                session,
                query,
                offset=current_offset,
                limit=limit,
            )
            pages_fetched += 1

            if data is None:  # Error occurred during fetch
                break

            raw_results = data.get("data", [])
            if not raw_results:  # No more results on this page or ever
                if self.debug:
                    logger.debug(
                        f"Query '{query}': No more results returned at offset {current_offset}.",
                    )
                break

            # Update total available if not yet set or if it changed (shouldn't normally)
            if total_available is None:
                total_available = data.get("total")
                if self.debug:
                    logger.debug(
                        f"Query '{query}': API reports total results: {total_available}",
                    )

            # Parse and add unique results for this page
            page_added_count = 0
            for item in raw_results:
                parsed = self._parse_result(item, query, category)
                if parsed and parsed.extra.get("paperId") not in seen_paper_ids:
                    all_parsed_results.append(parsed)
                    seen_paper_ids.add(parsed.extra["paperId"])
                    page_added_count += 1

            if self.debug:
                logger.debug(
                    f"Query '{query}', Page {pages_fetched}: "
                    f"Fetched {len(raw_results)}, Added {page_added_count} unique results. "
                    f"Total unique now: {len(all_parsed_results)}.",
                )

            # Prepare for next page
            next_offset = data.get(
                "next",
            )  # Semantic Scholar provides the next offset directly
            if next_offset is not None and next_offset > current_offset:
                current_offset = next_offset
            else:
                # If 'next' is not provided or doesn't advance, stop pagination
                if self.debug:
                    logger.debug(
                        f"Query '{query}': No valid 'next' offset provided ({next_offset}) "
                        f"or pagination limit reached. Stopping.",
                    )
                break

            # Optional: Add a small delay between pages
            await asyncio.sleep(0.1)

        if self.debug:
            logger.debug(
                f"Finished pagination for query '{query}'. "
                f"Fetched {pages_fetched} pages, gathered {len(all_parsed_results)} unique results.",
            )

        return all_parsed_results

    async def _process_final_results(
        self,
        all_results: List[SearchResultItem],
        max_total_results: int,
    ) -> List[SearchResultItem]:
        """Deduplicates across queries and trims to the final max_results limit."""
        # Results are already SearchResultItem objects from pagination parsing
        # Deduplicate based on paperId stored in 'extra' across all queries
        seen_paper_ids = set()
        unique_final_results = []
        for result in all_results:
            paper_id = result.extra.get("paperId")
            # Also check URL as a fallback, though paperId is more reliable
            unique_key = paper_id or result.url
            if unique_key and unique_key not in seen_paper_ids:
                unique_final_results.append(result)
                seen_paper_ids.add(unique_key)

        # No explicit sorting by score needed as Semantic Scholar ranks by relevance implicitly.
        # Could add sorting by year here if desired:
        # unique_final_results.sort(key=lambda r: r.published_date or '0', reverse=True)

        # Trim to the overall max_results limit
        final_results = unique_final_results[:max_total_results]

        if self.debug:
            logger.debug(
                f"Processed final results: {len(all_results)} raw -> "
                f"{len(unique_final_results)} unique -> {len(final_results)} trimmed.",
            )

        return final_results

    async def doi_to_paper(
        self,
        params: SemanticScholarSearchToolInputSchema,
        **kwargs,
    ) -> list[PaperDataItem]:
        """
        Fetches a paper based on it's DOI.

        Args:
            params: Input parameters including queries and category.

        Returns:
            List of PaperDataItem objects.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_paper_by_doi(
                    session,
                    query,
                )
                for query in params.queries
            ]
            results_per_query = await asyncio.gather(*tasks)
        results = [item for sublist in results_per_query for item in sublist]
        return results

    async def _arun(
        self,
        params: SemanticScholarSearchToolInputSchema,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> SemanticScholarSearchToolOutputSchema:
        """
        Runs the SemanticScholarSearchTool asynchronously.

        Args:
            params: Input parameters including queries and category.
            max_results: Override for the maximum number of final results.

        Returns:
            Output schema containing the list of search results.
        """
        # Determine the final max_results limit
        final_max_results = max_results or params.max_results or self.config.max_results

        # Calculate a *target* number of results per query to aim for during pagination.
        # Aim slightly higher than needed initially to account for deduplication across queries.
        # Ensure it doesn't request impossible amounts per query based on page limits.
        target_per_query = (
            final_max_results  # Start by aiming for the final count per query
        )
        if len(params.queries) > 1:
            # Fetch slightly more if multiple queries to allow for cross-query deduplication buffer
            target_per_query = int(final_max_results / len(params.queries) * 1.2) + 2

        max_possible_per_query = (
            self.config.max_pages_per_query * self.config.results_per_page
        )
        target_results_per_query = min(target_per_query, max_possible_per_query)
        target_results_per_query = max(target_results_per_query, 1)  # Ensure at least 1

        if self.debug:
            logger.debug(
                "Running Semantic Scholar Search: ",
                f"final_max_results={final_max_results}, "
                f"target_per_query={target_results_per_query}, "
                f"num_queries={len(params.queries)}",
            )

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
            results_per_query = await asyncio.gather(*tasks)

        all_raw_results = [item for sublist in results_per_query for item in sublist]

        # Final processing: deduplicate across queries and trim to max_results
        final_results = await self._process_final_results(
            all_raw_results,
            final_max_results,
        )

        return SemanticScholarSearchToolOutputSchema(
            results=final_results,
            category=params.category,  # Pass through the requested category
        )


class SimpleAgenticLitSearchToolConfig(BaseToolConfig):
    """
    Configuration for the SimpleAgenticLitSearchTool.
    This tool uses multi-rubric analysis and agentic decision-making
    to perform intelligent iterative literature searches.
    """

    min_positive_rubrics: int = Field(
        default=3,
        description="Minimum number of positive rubrics needed to stop searching (out of 6).",
    )
    max_iteration: int = Field(
        default=5,
        description="Maximum number of iterations to perform.",
    )
    max_results_per_iteration: int = Field(
        default=10,
        description="Maximum number of results to return per iteration.",
    )
    use_followup_after_iteration: int = Field(
        default=1,
        description="Use follow-up query agent after this many iterations.",
    )
    rubric_improvement_threshold: int = Field(
        default=2,
        description="Stop if no rubric improvement for this many iterations.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )


class SimpleAgenticLitSearchTool(SearchTool):
    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema

    config_schema = SimpleAgenticLitSearchToolConfig

    class _RubricAnalysis(BaseModel):
        """Analysis of multi-rubric assessment for agentic decision making."""

        topic_alignment_positive: bool = Field(default=False)
        content_depth_positive: bool = Field(default=False)
        recency_relevance_positive: bool = Field(default=False)
        methodological_relevance_positive: bool = Field(default=False)
        evidence_quality_positive: bool = Field(default=False)
        scope_relevance_positive: bool = Field(default=False)

        positive_rubric_count: int = Field(default=0)
        weak_rubrics: List[str] = Field(default_factory=list)
        strong_rubrics: List[str] = Field(default_factory=list)

        overall_assessment: str = Field(default="")
        reasoning_steps: List[str] = Field(default_factory=list)

    class _StoppingCriteria(BaseModel):
        stop_now: bool = Field(default=False)
        reasoning_trace: str = Field(default="")
        rubric_analysis: Optional["SimpleAgenticLitSearchTool._RubricAnalysis"] = Field(
            default=None,
        )
        recommended_query_focus: List[str] = Field(default_factory=list)

    def __init__(
        self,
        config: SimpleAgenticLitSearchToolConfig | None = None,
        search_tool: SearchTool | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        debug: bool = False,
    ) -> None:
        config = config or SimpleAgenticLitSearchToolConfig(debug=debug)
        super().__init__(config=config, debug=debug)
        self.search_tool = search_tool or SearxNGSearchTool()
        self.relevancy_agent = relevancy_agent or MultiRubricRelevancyAgent()
        self.query_agent = query_agent or QueryAgent()
        self.followup_query_agent = followup_query_agent or FollowUpQueryAgent()

        # Track rubric patterns for agentic learning
        self.rubric_history = []

    def _deduplicate_results(
        self,
        new_results: List[SearchResultItem],
        existing_results: List[SearchResultItem],
    ) -> List[SearchResultItem]:
        """Remove duplicate results based on URL."""
        existing_urls = {r.url for r in existing_results}
        return [r for r in new_results if r.url not in existing_urls]

    def _accumulate_content(self, results: List[SearchResultItem]) -> str:
        content = ""
        for result in results:
            content += f"\nTitle: {result.title}\nContent: {result.content}\n"
        return content.strip()

    def _analyze_rubrics(
        self,
        rubric_output: MultiRubricRelevancyOutputSchema,
    ) -> "_RubricAnalysis":
        """Analyze multi-rubric output to determine positive/negative assessments."""
        analysis = self._RubricAnalysis()

        # Determine positive assessments using enum values directly
        analysis.topic_alignment_positive = (
            rubric_output.topic_alignment == TopicAlignmentLabel.ALIGNED
        )
        analysis.content_depth_positive = (
            rubric_output.content_depth == ContentDepthLabel.COMPREHENSIVE
        )
        analysis.recency_relevance_positive = (
            rubric_output.recency_relevance == RecencyRelevanceLabel.CURRENT
        )
        analysis.methodological_relevance_positive = (
            rubric_output.methodological_relevance
            == MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND
        )
        analysis.evidence_quality_positive = (
            rubric_output.evidence_quality == EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE
        )
        analysis.scope_relevance_positive = (
            rubric_output.scope_relevance == ScopeRelevanceLabel.IN_SCOPE
        )

        # Count positive rubrics
        positive_flags = [
            analysis.topic_alignment_positive,
            analysis.content_depth_positive,
            analysis.recency_relevance_positive,
            analysis.methodological_relevance_positive,
            analysis.evidence_quality_positive,
            analysis.scope_relevance_positive,
        ]
        analysis.positive_rubric_count = sum(positive_flags)

        # Identify weak and strong rubrics
        rubric_mapping = {
            "topic_alignment": analysis.topic_alignment_positive,
            "content_depth": analysis.content_depth_positive,
            "recency_relevance": analysis.recency_relevance_positive,
            "methodological_relevance": analysis.methodological_relevance_positive,
            "evidence_quality": analysis.evidence_quality_positive,
            "scope_relevance": analysis.scope_relevance_positive,
        }

        for rubric_name, is_positive in rubric_mapping.items():
            if is_positive:
                analysis.strong_rubrics.append(rubric_name)
            else:
                analysis.weak_rubrics.append(rubric_name)

        # Overall assessment
        if analysis.positive_rubric_count >= 4:
            analysis.overall_assessment = "strong"
        elif analysis.positive_rubric_count >= 2:
            analysis.overall_assessment = "moderate"
        else:
            analysis.overall_assessment = "weak"

        # Copy reasoning steps
        analysis.reasoning_steps = rubric_output.reasoning_steps

        return analysis

    def _make_agentic_stopping_decision(
        self,
        rubric_analysis: "_RubricAnalysis",
        iteration: int,
        current_result_count: int,
        desired_max_results: int,
        query: str,
    ) -> tuple[bool, str]:
        """Dynamic agentic stopping decision balancing quality, quantity, and adaptivity."""

        # Calculate progress metrics for dynamic decision making
        result_progress = (
            current_result_count / desired_max_results if desired_max_results > 0 else 0
        )
        quality_score = rubric_analysis.positive_rubric_count / 6

        # Dynamic minimum results threshold (adaptive based on quality)
        min_results_threshold = max(
            desired_max_results * 0.3,  # At least 30% of requested
            min(5, desired_max_results),  # But at least 5 or total requested if less
        )

        # Never stop if we have too few results
        if current_result_count < min_results_threshold:
            return False, (
                f"CONTINUE: Need more results ({current_result_count}/{min_results_threshold:.0f} minimum)"
            )

        # Dynamic quality threshold - lower if we have many results, higher if few
        quality_threshold = self.min_positive_rubrics
        if result_progress > 0.8:  # If we have most requested results
            quality_threshold = max(
                2,
                self.min_positive_rubrics - 1,
            )  # Lower quality bar
        elif result_progress < 0.5:  # If we have few results
            quality_threshold = min(
                5,
                self.min_positive_rubrics + 1,
            )  # Higher quality bar

        # Adaptive stopping based on quality + quantity balance
        if rubric_analysis.positive_rubric_count >= quality_threshold:
            # Good quality achieved
            if current_result_count >= desired_max_results:
                return (
                    True,
                    f"STOP: Target reached ({current_result_count}/{desired_max_results}) + quality good ({rubric_analysis.positive_rubric_count}/6)",
                )
            elif (
                result_progress >= 0.7 and quality_score >= 0.67
            ):  # 70% results + 67% quality
                return (
                    True,
                    f"STOP: Sufficient results ({current_result_count}/{desired_max_results}) + excellent quality ({rubric_analysis.positive_rubric_count}/6)",
                )

        # Force stop if we've exceeded target significantly (search overflow protection)
        if current_result_count >= desired_max_results * 1.5:
            return (
                True,
                f"STOP: Exceeded target ({current_result_count}/{desired_max_results}) - preventing overflow",
            )

        # Critical rubrics check - but balanced with results progress
        critical_rubrics = {"topic_alignment", "evidence_quality"}
        weak_critical = critical_rubrics.intersection(set(rubric_analysis.weak_rubrics))

        if (
            weak_critical and result_progress < 0.8
        ):  # Only block if we don't have most results
            return False, (
                f"CONTINUE: Critical rubrics weak ({weak_critical}) + need more results ({current_result_count}/{desired_max_results})"
            )

        # Adaptive stagnation check - more lenient if we need more results
        if len(self.rubric_history) >= self.rubric_improvement_threshold:
            recent_scores = [
                h["analysis"].positive_rubric_count
                for h in self.rubric_history[-self.rubric_improvement_threshold :]
            ]
            if all(
                score <= rubric_analysis.positive_rubric_count
                for score in recent_scores
            ):
                # Only stop for stagnation if we have reasonable results or excellent quality
                if result_progress >= 0.6 or quality_score >= 0.8:
                    return True, (
                        f"STOP: No improvement in {self.rubric_improvement_threshold} iterations + {current_result_count}/{desired_max_results} results"
                    )

        # Continue searching - provide specific guidance
        if result_progress < 0.5:
            return (
                False,
                f"CONTINUE: Need more results ({current_result_count}/{desired_max_results}) + improving quality ({rubric_analysis.positive_rubric_count}/6)",
            )
        else:
            return (
                False,
                f"CONTINUE: Refining quality ({rubric_analysis.positive_rubric_count}/6) with {current_result_count}/{desired_max_results} results",
            )

    def _generate_query_focus_recommendations(
        self,
        rubric_analysis: "_RubricAnalysis",
    ) -> List[str]:
        """Generate query focus recommendations based on weak rubrics."""
        # Mapping from weak rubrics to query focus strategies
        rubric_to_strategy = {
            "topic_alignment": QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY,
            "content_depth": QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS,
            "evidence_quality": QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES,
            "methodological_relevance": QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS,
            "recency_relevance": QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS,
            "scope_relevance": QueryFocusStrategy.ADJUST_QUERY_SCOPE,
        }

        recommendations = []
        for weak_rubric in rubric_analysis.weak_rubrics:
            if strategy := rubric_to_strategy.get(weak_rubric):
                recommendations.append(strategy.value)

        return recommendations

    async def _should_stop(
        self,
        iteration: int,
        query: str,
        all_results: list,
        current_results: list,
        max_results: int,
    ) -> "SimpleAgenticLitSearchTool._StoppingCriteria":
        criteria = self._StoppingCriteria(
            stop_now=False,
            reasoning_trace=f"Iteration {iteration}/{self.max_iteration}",
        )

        # Basic stopping conditions
        if iteration >= self.max_iteration:
            criteria.stop_now = True
            criteria.reasoning_trace = (
                f"Max iterations reached ({iteration}/{self.max_iteration})"
            )
            return criteria

        if iteration > 0 and not current_results:
            criteria.stop_now = True
            criteria.reasoning_trace = f"No new results found in iteration {iteration}"
            return criteria

        # Multi-rubric analysis for agentic decision making
        if (context := self._accumulate_content(current_results)) and iteration > 0:
            try:
                # Get multi-rubric assessment
                rubric_result = await self.relevancy_agent.arun(
                    MultiRubricRelevancyInputSchema(content=context, query=query),
                )

                if self.debug:
                    logger.debug(
                        f"Multi-rubric assessment for iteration {iteration}: {rubric_result}",
                    )

                # Analyze rubrics for decision making
                rubric_analysis = self._analyze_rubrics(rubric_result)
                criteria.rubric_analysis = rubric_analysis

                # Store rubric history for meta-learning (moved to _learn_from_iteration)

                # Agentic stopping decision based on multi-rubric analysis
                stop_decision, reasoning = self._make_agentic_stopping_decision(
                    rubric_analysis,
                    iteration,
                    len(all_results),
                    max_results,
                    query,
                )

                criteria.stop_now = stop_decision
                criteria.reasoning_trace = reasoning

                # Generate query focus recommendations for next iteration
                if not stop_decision:
                    criteria.recommended_query_focus = (
                        self._generate_query_focus_recommendations(
                            rubric_analysis,
                        )
                    )

            except Exception as e:
                logger.warning(f"Error in multi-rubric analysis: {str(e)}")
                # Fallback to continue searching if analysis fails
                criteria.stop_now = False
                criteria.reasoning_trace = (
                    "Multi-rubric analysis failed, continuing search"
                )

        return criteria

    async def _generate_queries(
        self,
        queries: List[str],
        iteration: int,
        num_queries: int = 3,
        results: Optional[List[SearchResultItem]] = None,
        accumulated_content: str = "",
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate queries using either initial query agent or follow-up query agent
        based on the iteration number, with adaptive focus based on rubric analysis.
        """
        if iteration <= self.use_followup_after_iteration:
            # Use initial query generation for early iterations
            return await self._generate_initial_queries(
                queries=queries,
                iteration=iteration,
                num_queries=num_queries,
                results=results,
                rubric_focus=rubric_focus,
            )
        else:
            # Use follow-up query generation for later iterations
            if accumulated_content:
                logger.debug(
                    f"Switching to follow-up query generation at iteration {iteration}",
                )
                return await self._generate_followup_queries(
                    original_queries=queries,
                    accumulated_content=accumulated_content,
                    num_queries=num_queries,
                    rubric_focus=rubric_focus,
                )
            else:
                # Fallback to initial query generation if no content available
                return await self._generate_initial_queries(
                    queries=queries,
                    iteration=iteration,
                    num_queries=num_queries,
                    results=results,
                    rubric_focus=rubric_focus,
                )

    async def _generate_followup_queries(
        self,
        original_queries: List[str],
        accumulated_content: str,
        num_queries: int = 3,
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate follow-up queries using the followup_query_agent with rubric-based adaptations."""
        try:
            # Enhance content with rubric-based guidance
            enhanced_content = accumulated_content
            if rubric_focus:
                focus_guidance = self._create_rubric_focus_guidance(rubric_focus)
                enhanced_content += f"\n\nFOCUS AREAS NEEDED: {focus_guidance}"

            followup_input = FollowUpQueryAgentInputSchema(
                original_queries=original_queries,
                content=enhanced_content,
                num_queries=num_queries,
            )

            followup_result: FollowUpQueryAgentOutputSchema = (
                await self.followup_query_agent.arun(
                    followup_input,
                )
            )

            if self.debug:
                logger.debug(
                    f"Follow-up reasoning: {followup_result.reasoning}",
                )
                logger.debug(
                    f"Identified gaps: {followup_result.original_query_gaps}",
                )
                if rubric_focus:
                    logger.debug(f"Applied rubric focus: {rubric_focus}")

            # Apply rubric-based query modifications
            if rubric_focus:
                adapted_queries = self._adapt_queries_for_rubrics(
                    followup_result.followup_queries,
                    rubric_focus,
                )
                return adapted_queries

            return followup_result.followup_queries

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning(
                f"Error generating follow-up queries. Error => {str(e)}",
            )
            return original_queries  # Fallback to original queries
        return original_queries

    async def _generate_initial_queries(
        self,
        queries: List[str],
        iteration: int,
        num_queries: int = 3,
        results: Optional[List[SearchResultItem]] = None,
        rubric_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate initial queries using the query_agent with rubric-based adaptations."""
        context = ""
        if results:
            titles = [r.title for r in results]
            context = f"Previous searches found: {', '.join(titles)}"

        query_instruction = f"""
        Iteration {iteration} queries : {queries}
        Context/results so far: {context}
        """.strip()

        # Add rubric-based focus guidance
        if rubric_focus:
            focus_guidance = self._create_rubric_focus_guidance(rubric_focus)
            query_instruction += f"\n\nFOCUS AREAS NEEDED: {focus_guidance}"

        res = QueryAgentOutputSchema(queries=queries)
        try:
            res = await self.query_agent.arun(
                QueryAgentInputSchema(
                    num_queries=num_queries,
                    query=query_instruction,
                ),
            )

            # Apply rubric-based query modifications
            if rubric_focus:
                adapted_queries = self._adapt_queries_for_rubrics(
                    res.queries,
                    rubric_focus,
                )
                res.queries = adapted_queries

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning(
                f"Error generating initial queries. Error => {str(e)}",
            )
        return res.queries

    def _create_rubric_focus_guidance(self, rubric_focus: List[str]) -> str:
        """Create human-readable guidance for rubric focus areas."""
        guidance_map = {
            QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value: "Make queries more specific to the target topic and domain",
            QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value: "Look for systematic reviews, meta-analyses, and comprehensive surveys",
            QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value: "Focus on high-impact peer-reviewed journals and quality publications",
            QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value: "Find papers with strong methodological approaches and validation",
            QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value: "Prioritize recent publications (last 2-3 years)",
            QueryFocusStrategy.ADJUST_QUERY_SCOPE.value: "Refine the scope to match the research question boundaries",
        }

        guidance_list = [guidance_map.get(focus, focus) for focus in rubric_focus]
        return "; ".join(guidance_list)

    def _prioritize_rubric_focus(self, rubric_focus: List[str]) -> List[str]:
        """Prioritize rubric focus areas to avoid query over-complexity."""
        if not rubric_focus:
            return []

        # Priority order for rubric fixes (most critical first)
        priority_order = [
            QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value,  # Most important - improves relevance
            QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value,  # High impact - improves quality
            QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value,  # Good for depth
            QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value,  # Simple but effective
            QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value,  # Specific improvement
            QueryFocusStrategy.ADJUST_QUERY_SCOPE.value,  # General fallback
        ]

        # Return top 2-3 priorities to avoid complexity
        prioritized = []
        for priority in priority_order:
            if priority in rubric_focus:
                prioritized.append(priority)
                if len(prioritized) >= 2:  # Limit to 2 adaptations per query
                    break

        return prioritized

    def _adapt_queries_for_rubrics(
        self,
        queries: List[str],
        rubric_focus: List[str],
    ) -> List[str]:
        """Apply rubric-specific adaptations to queries with smart prioritization."""
        # Prioritize focus areas to avoid over-complexity
        prioritized_focus = self._prioritize_rubric_focus(rubric_focus)

        if self.debug:
            logger.debug(
                f"Prioritized rubric focus (from {len(rubric_focus)} to {len(prioritized_focus)}): {prioritized_focus}",
            )

        adapted_queries = []
        for i, query in enumerate(queries):
            # Apply different adaptations to different queries for variety
            focus_for_this_query = (
                prioritized_focus[i % len(prioritized_focus)]
                if prioritized_focus
                else None
            )

            if focus_for_this_query:
                adapted_query = self._apply_single_focus_adaptation(
                    query,
                    focus_for_this_query,
                )
                adapted_queries.append(adapted_query)
            else:
                adapted_queries.append(query)  # Keep original if no focus

        # Always include at least one original query as fallback
        if queries[0] not in adapted_queries:
            adapted_queries.append(queries[0])

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in adapted_queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)

        return unique_queries

    def _apply_single_focus_adaptation(self, query: str, focus: str) -> str:
        """Apply a single, clean adaptation based on rubric focus."""
        if focus == QueryFocusStrategy.REFINE_TOPIC_SPECIFICITY.value:
            # Make query more specific without complex syntax
            return f"{query} specific detailed"

        elif focus == QueryFocusStrategy.SEARCH_COMPREHENSIVE_REVIEWS.value:
            return f"{query} review OR survey OR meta-analysis"

        elif focus == QueryFocusStrategy.TARGET_PEER_REVIEWED_SOURCES.value:
            return f"{query} journal peer-reviewed"

        elif focus == QueryFocusStrategy.SEARCH_METHODOLOGICAL_PAPERS.value:
            return f"{query} methodology approach"

        elif focus == QueryFocusStrategy.ADD_RECENT_YEAR_FILTERS.value:
            return f"{query} 2020..2025"

        elif focus == QueryFocusStrategy.ADJUST_QUERY_SCOPE.value:
            return f'"{query}" specific'

        return query

    def _learn_from_iteration(
        self,
        iteration: int,
        queries: List[str],
        rubric_analysis: "_RubricAnalysis",
    ):
        """Simple learning: just track rubric history for trend analysis."""
        self.rubric_history.append(
            {
                "iteration": iteration,
                "analysis": rubric_analysis,
                "query": " AND ".join(queries),
            },
        )

    def _calculate_dynamic_batch_size(self, iteration: int, previous_criteria) -> int:
        """Calculate adaptive batch size based on rubric performance."""
        base_size = self.max_results_per_iteration

        # First iteration uses base size
        if iteration <= 1:
            return base_size

        # If we have rubric analysis from previous iteration
        if (
            hasattr(previous_criteria, "rubric_analysis")
            and previous_criteria.rubric_analysis
        ):
            rubric_count = previous_criteria.rubric_analysis.positive_rubric_count

            # If doing very poorly (0-1 positive rubrics), search more aggressively
            if rubric_count <= 1:
                return min(base_size * 2, 20)  # Double the search, cap at 20

            # If doing poorly (2 positive rubrics), search slightly more
            elif rubric_count == 2:
                return int(base_size * 1.5)

            # If doing well (4+ positive rubrics), can be more conservative
            elif rubric_count >= 4:
                return max(base_size // 2, 5)  # Half the search, minimum 5

        return base_size

    async def _arun(
        self,
        params: SearchToolInputSchema,
        **kwargs: Any,
    ) -> SearchToolOutputSchema:
        desired_max_results = params.max_results

        iteration = 0
        all_results = []
        current_results = []
        content_so_far = ""

        while not (
            criteria := await self._should_stop(
                iteration=iteration,
                all_results=all_results,
                current_results=current_results,
                max_results=desired_max_results,
                query=" AND ".join(params.queries),
            )
        ).stop_now:
            logger.debug(f"Stopping Criteria :: {criteria}")
            iteration += 1

            remaining_needed = desired_max_results - len(all_results)
            # Dynamic batch sizing: adjust based on previous rubric performance
            dynamic_batch_size = self._calculate_dynamic_batch_size(iteration, criteria)
            search_limit = min(remaining_needed, dynamic_batch_size)

            current_queries = params.queries
            if iteration > 0:
                # Get rubric focus from previous iteration's stopping criteria
                rubric_focus = None
                if (
                    hasattr(criteria, "recommended_query_focus")
                    and criteria.recommended_query_focus
                ):
                    rubric_focus = criteria.recommended_query_focus

                logger.debug(f"Rubric focus for iteration {iteration}: {rubric_focus}")

                current_queries = await self._generate_queries(
                    iteration=iteration,
                    num_queries=3,
                    queries=params.queries,
                    results=all_results,
                    accumulated_content=content_so_far,  # Pass accumulated content
                    rubric_focus=rubric_focus,  # Pass rubric focus for adaptive queries
                )

            logger.debug(
                f"Generated queries (iteration {iteration}): {current_queries}",
            )
            search_input = SearchToolInputSchema(
                queries=current_queries,
                max_results=search_limit,
                category=params.category,
            )
            search_result = await self.search_tool.arun(
                self.search_tool.input_schema(**search_input.model_dump()),
            )

            current_results = self._deduplicate_results(
                new_results=search_result.results,
                existing_results=all_results,
            )

            # Fallback mechanism: If adapted queries return no results, try original queries
            if (
                not current_results
                and iteration > 1
                and current_queries != params.queries
            ):
                if self.debug:
                    logger.debug(
                        f"No results from adapted queries, falling back to original queries: {params.queries}",
                    )

                fallback_input = SearchToolInputSchema(
                    queries=params.queries,  # Use original queries
                    max_results=search_limit,
                    category=params.category,
                )

                fallback_result = await self.search_tool.arun(
                    self.search_tool.input_schema(**fallback_input.model_dump()),
                )

                current_results = self._deduplicate_results(
                    new_results=fallback_result.results,
                    existing_results=all_results,
                )

                if current_results and self.debug:
                    logger.debug(
                        f"Fallback successful: found {len(current_results)} results",
                    )

            all_results.extend(current_results)

            # Update accumulated content after each iteration
            new_content = self._accumulate_content(current_results)
            if new_content:
                content_so_far += "\n" + new_content if content_so_far else new_content

            # Learn from this iteration if we have rubric analysis
            if hasattr(criteria, "rubric_analysis") and criteria.rubric_analysis:
                self._learn_from_iteration(
                    iteration,
                    current_queries,
                    criteria.rubric_analysis,
                )

            if self.debug:
                logger.debug(
                    f"Content accumulated so far (chars): {len(content_so_far)}",
                )
                logger.debug(
                    f"Current iteration results: {len(current_results)}",
                )

        logger.debug(f"Final Stopping Criteria :: {criteria}")
        return SearchToolOutputSchema(results=all_results, category=params.category)
