from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin

import aiohttp
from loguru import logger
from pydantic import SecretStr, field_validator
from pydantic.fields import Field
from pydantic.networks import HttpUrl

from akd.structures import PaperDataItem, SearchResultItem
from akd.tools._base import BaseTool
from akd.utils import RateLimiter

from ._base import SearchToolConfig, SearchToolInputSchema, SearchToolOutputSchema


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


class SemanticScholarSearchToolConfig(SearchToolConfig):
    """Configuration for the Semantic Scholar Search Tool."""

    api_key: Optional[str] = Field(
        default=(os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")),
    )
    base_url: HttpUrl = Field(default="https://api.semanticscholar.org")
    max_results: int = Field(
        default=int(os.getenv("SEMANTIC_SCHOLAR_MAX_RESULTS", "10")),
    )
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
        default=int(os.getenv("SEMANTIC_SCHOLAR_RESULTS_PER_PAGE", "10")),
        le=100,
        gt=0,
    )
    # Maximum number of *pages* to fetch per query to avoid excessive requests
    max_pages_per_query: int = Field(
        default=int(
            os.getenv(
                "SEMANTIC_SCHOLAR_MAX_PAGES",
                "10",
            ),
        ),
        gt=0,
        le=50,
    )
    external_id: Optional[
        Literal["DOI", "ARXIV", "PMID", "ACL", "MAG", "CorpusId", "PMCID", "URL"]
    ] = Field(default="DOI")
    debug: bool = False

    # Rate limiting configuration
    requests_per_second: float = Field(
        default=float(os.getenv("SEMANTIC_SCHOLAR_REQUESTS_PER_SECOND", "1.0")),
        description="Maximum requests per second to respect API rate limits",
        gt=0,
        le=10,
    )

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

    def _post_init(self) -> None:
        super()._post_init()
        if self.debug:
            logger.debug(
                f"Setting rate limit to :: {self.config.requests_per_second} Requests/Second",
            )
        self.rate_limiter = RateLimiter(
            max_calls_per_second=self.config.requests_per_second,
        )

    @classmethod
    def from_params(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[HttpUrl] = "https://api.semanticscholar.org",
        max_results: int = 10,
        fields: Optional[List[str]] = None,
        results_per_page: int = 100,
        max_pages_per_query: int = 5,
        external_id: str = "DOI",
        debug: bool = False,
        requests_per_second: float = 1.0,
    ) -> SemanticScholarSearchTool:
        """Creates an instance from specific parameters."""
        config_data = {
            "api_key": api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "base_url": base_url or "https://api.semanticscholar.org",
            "max_results": max_results,
            "results_per_page": results_per_page,
            "max_pages_per_query": max_pages_per_query,
            "external_id": external_id,
            "debug": debug,
            "requests_per_second": requests_per_second,
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
        search_url = urljoin(str(self.config.base_url), "graph/v1/paper/search")
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
            # Apply rate limiting before making the request
            await self.rate_limiter.acquire()

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
        external_id: Optional[str],
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
                external_id=external_id or None,
            )
            if self.debug:
                logger.debug(
                    f"Processed paper with External ID {external_id}",
                )
            return paper_item
        except Exception as e:
            logger.debug(
                f"Could not parse response to paper object for External ID {external_id}: {str(e)}",
            )

    async def _fetch_paper_by_external_id(
        self,
        session: aiohttp.ClientSession,
        query: str,
        external_id: str = "DOI",
    ) -> list[PaperDataItem]:
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
        search_url = urljoin(
            str(self.config.base_url),
            f"graph/v1/paper/{external_id}:{query}",
        )
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
            # Apply rate limiting before making the request
            await self.rate_limiter.acquire()

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
                return [self._parse_paper(item=data, external_id=query)]

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
            title=item.pop("title") or "Untitled",  # Ensure title is never None
            content=item.pop("abstract"),  # Map abstract to content
            query=query or "Unknown query",  # Ensure query is never None
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

            # Rate limiting is handled by the rate_limiter in _fetch_search_page

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

    async def fetch_paper_by_external_id(
        self,
        params: SemanticScholarSearchToolInputSchema,
        **kwargs,
    ) -> list[PaperDataItem]:
        """
        Fetches a paper from Semantic Scholar using an external ID.

        Args:
            params: Input parameters including queries and category.

        Returns:
            List of PaperDataItem objects.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_paper_by_external_id(
                    session,
                    query,
                    kwargs.get("external_id", self.config.external_id),
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
