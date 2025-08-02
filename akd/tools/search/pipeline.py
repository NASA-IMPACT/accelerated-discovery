from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from pydantic import Field

from akd.structures import SearchResultItem
from akd.tools.scrapers._base import ScraperToolBase
from akd.tools.scrapers.composite import CompositeScraper, ResearchArticleResolver
from akd.tools.scrapers.resolvers import BaseArticleResolver

from ._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)


class FullTextSearchPipelineConfig(SearchToolConfig):
    """Configuration for the FullTextSearchPipeline."""

    # Pipeline behavior configuration
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process search results in parallel for scraping",
    )
    max_concurrent_scrapes: int = Field(
        default=5,
        description="Maximum number of concurrent scraping operations",
    )
    include_original_content: bool = Field(
        default=True,
        description="Whether to preserve original search result content alongside scraped content",
    )
    scraping_timeout: int = Field(
        default=30,
        description="Timeout in seconds for individual scraping operations",
    )

    # Content processing configuration
    max_content_length: int = Field(
        default=100_000,
        description="Maximum length of scraped content to include",
    )
    content_truncation_strategy: str = Field(
        default="end",
        description="How to truncate content if it exceeds max_length: 'end', 'middle', 'start'",
    )

    # Error handling configuration
    fail_on_scraping_errors: bool = Field(
        default=False,
        description="Whether to fail the entire pipeline if scraping fails for any result",
    )
    min_successful_scrapes: Optional[int] = Field(
        default=None,
        description="Minimum number of successful scrapes required (None = no minimum)",
    )


class FullTextSearchPipeline(SearchTool):
    """
    Full-text search pipeline that combines search, URL resolution, and content scraping.

    This pipeline:
    1. Uses an underlying search tool to fetch search results
    2. Resolves open access URLs for academic papers
    3. Scrapes full text content from the resolved URLs
    4. Enhances SearchResultItem objects with scraped content

    Attributes:
        search_tool: The underlying search tool to use for initial search
        resolver: Article resolver for finding open access URLs
        scraper: Content scraper for extracting full text
    """

    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema
    config_schema = FullTextSearchPipelineConfig

    def __init__(
        self,
        search_tool: SearchTool,
        resolver: Optional[BaseArticleResolver] = None,
        scraper: Optional[ScraperToolBase] = None,
        config: Optional[FullTextSearchPipelineConfig] = None,
        debug: bool = False,
    ):
        """
        Initialize the FullTextSearchPipeline.

        Args:
            search_tool: The underlying search tool (e.g., SearxNG, Semantic Scholar)
            resolver: Article resolver for open access URLs (defaults to ResearchArticleResolver)
            scraper: Content scraper (defaults to CompositeScraper)
            config: Pipeline configuration
            debug: Enable debug logging
        """
        config = config or FullTextSearchPipelineConfig()
        super().__init__(config, debug)

        self.search_tool = search_tool
        self.resolver = resolver or ResearchArticleResolver(debug=debug)
        self.scraper = scraper or CompositeScraper(debug=debug)

        if debug:
            logger.debug("Initialized FullTextSearchPipeline with:")
            logger.debug(f"  - Search tool: {search_tool.__class__.__name__}")
            logger.debug(f"  - Resolver: {self.resolver.__class__.__name__}")
            logger.debug(f"  - Scraper: {self.scraper.__class__.__name__}")

    async def _resolve_open_access_url(self, result: SearchResultItem) -> Optional[str]:
        """
        Resolve the open access URL for a search result.

        Args:
            result: Search result item to resolve

        Returns:
            Resolved open access URL or None if resolution fails
        """
        try:
            # Try to resolve from the main URL first
            resolver_output = await self.resolver.arun(
                self.resolver.input_schema(url=result.url),
            )

            if resolver_output.url and str(resolver_output.url) != str(result.url):
                if self.debug:
                    logger.debug(f"Resolved {result.url} -> {resolver_output.url}")
                return str(resolver_output.url)

            # If no resolution found and there's a PDF URL, try that
            if result.pdf_url:
                return str(result.pdf_url)

            # Fall back to original URL
            return str(result.url)

        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to resolve URL {result.url}: {e}")
            # Fall back to pdf_url or original url
            return str(result.pdf_url) if result.pdf_url else str(result.url)

    async def _scrape_content(
        self,
        url: str,
        result: SearchResultItem,
    ) -> Optional[str]:
        """
        Scrape full text content from a URL.

        Args:
            url: URL to scrape
            result: Original search result for context

        Returns:
            Scraped content or None if scraping fails
        """
        try:
            scraper_output = await asyncio.wait_for(
                self.scraper.arun(self.scraper.input_schema(url=url)),
                timeout=self.scraping_timeout,
            )

            if scraper_output.content and scraper_output.content.strip():
                content = scraper_output.content.strip()

                # Apply content length limits
                if len(content) > self.max_content_length:
                    content = self._truncate_content(content)

                if self.debug:
                    logger.debug(
                        f"Successfully scraped {len(content)} characters from {url}",
                    )

                return content
            else:
                if self.debug:
                    logger.warning(f"No content scraped from {url}")
                return None

        except asyncio.TimeoutError:
            if self.debug:
                logger.warning(f"Scraping timeout for {url}")
            return None
        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def _truncate_content(self, content: str) -> str:
        """Truncate content according to the configured strategy."""
        if len(content) <= self.max_content_length:
            return content

        if self.content_truncation_strategy == "start":
            return "..." + content[-(self.max_content_length - 3) :]
        elif self.content_truncation_strategy == "middle":
            half = (self.max_content_length - 5) // 2
            return content[:half] + " ... " + content[-half:]
        else:  # "end" (default)
            return content[: self.max_content_length - 3] + "..."

    async def _process_single_result(
        self,
        result: SearchResultItem,
    ) -> SearchResultItem:
        """
        Process a single search result through the full pipeline.

        Args:
            result: Original search result

        Returns:
            Enhanced search result with scraped content
        """
        try:
            # Resolve open access URL
            resolved_url = await self._resolve_open_access_url(result)

            if not resolved_url:
                if self.debug:
                    logger.warning(f"No URL to scrape for result: {result.title}")
                return result

            # Scrape content
            scraped_content = await self._scrape_content(resolved_url, result)

            if scraped_content:
                # Create enhanced result
                enhanced_result = result.model_copy()

                if self.include_original_content and result.content:
                    # Combine original and scraped content
                    enhanced_result.content = (
                        f"{result.content}\n\n--- FULL TEXT ---\n\n{scraped_content}"
                    )
                else:
                    # Replace with scraped content
                    enhanced_result.content = scraped_content

                # Update PDF URL if we resolved it
                if resolved_url != str(result.url):
                    enhanced_result.pdf_url = resolved_url

                # Add metadata about scraping
                if not enhanced_result.extra:
                    enhanced_result.extra = {}
                enhanced_result.extra.update(
                    {
                        "full_text_scraped": True,
                        "scraped_url": resolved_url,
                        "scraper_used": self.scraper.__class__.__name__,
                        "resolver_used": self.resolver.__class__.__name__,
                    },
                )

                return enhanced_result
            else:
                # Scraping failed, return original with metadata
                enhanced_result = result.model_copy()
                if not enhanced_result.extra:
                    enhanced_result.extra = {}
                enhanced_result.extra["full_text_scraped"] = False
                enhanced_result.extra["scraping_attempted_url"] = resolved_url

                return enhanced_result

        except Exception as e:
            if self.debug:
                logger.error(f"Error processing result {result.title}: {e}")

            # Return original result with error metadata
            enhanced_result = result.model_copy()
            if not enhanced_result.extra:
                enhanced_result.extra = {}
            enhanced_result.extra["full_text_scraped"] = False
            enhanced_result.extra["scraping_error"] = str(e)

            return enhanced_result

    async def _arun(
        self,
        params: SearchToolInputSchema,
        **kwargs,
    ) -> SearchToolOutputSchema:
        """
        Run the full-text search pipeline.

        Args:
            params: Search parameters
            **kwargs: Additional parameters

        Returns:
            Search results enhanced with full text content

        Raises:
            Exception: If the pipeline fails and fail_on_scraping_errors is True
        """
        if self.debug:
            logger.info(
                f"🔄 Starting FullTextSearchPipeline for {len(params.queries)} queries",
            )

        # Step 1: Get initial search results
        params_search = self.search_tool.input_schema(**params.model_dump())
        search_results = await self.search_tool.arun(params_search, **kwargs)

        if not search_results.results:
            if self.debug:
                logger.info("No search results to process")
            return search_results

        if self.debug:
            logger.info(f"📄 Processing {len(search_results.results)} search results")

        # Step 2: Process results through the pipeline
        if self.parallel_processing:
            # Process in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent_scrapes)

            async def process_with_semaphore(result):
                async with semaphore:
                    return await self._process_single_result(result)

            enhanced_results = await asyncio.gather(
                *[process_with_semaphore(result) for result in search_results.results],
                return_exceptions=not self.fail_on_scraping_errors,
            )

            # Filter out exceptions if we're not failing on errors
            if not self.fail_on_scraping_errors:
                enhanced_results = [
                    result
                    for result in enhanced_results
                    if not isinstance(result, Exception)
                ]
        else:
            # Process sequentially
            enhanced_results = []
            for result in search_results.results:
                try:
                    enhanced_result = await self._process_single_result(result)
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    if self.fail_on_scraping_errors:
                        raise
                    if self.debug:
                        logger.error(f"Failed to process result {result.title}: {e}")

        # Step 3: Validate results
        successful_scrapes = sum(
            1
            for result in enhanced_results
            if result.extra and result.extra.get("full_text_scraped", False)
        )

        if self.debug:
            logger.info(
                f"✅ Successfully scraped {successful_scrapes}/{len(enhanced_results)} results",
            )

        if (
            self.min_successful_scrapes is not None
            and successful_scrapes < self.min_successful_scrapes
        ):
            raise Exception(
                f"Only {successful_scrapes} successful scrapes, minimum required: {self.min_successful_scrapes}",
            )

        # Return enhanced results
        return SearchToolOutputSchema(
            results=enhanced_results,
            category=search_results.category,
        )
