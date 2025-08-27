from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from pydantic import Field

from akd.structures import SearchResultItem
from akd.tools.resolvers import (
    ResearchArticleResolver,
    CrossRefDoiResolver,
    ArxivResolver,
    ADSResolver, 
    DOIResolver,
    PDFUrlResolver, 
    BaseArticleResolver,
)
from akd.tools.resolvers._base import ResolverOutputSchema
from akd.tools.resolvers.unpaywall import UnpaywallResolver
from akd.tools.scrapers._base import ScraperToolBase
from akd.tools.scrapers.composite import CompositeScraper
from akd.tools.scrapers.omni import DoclingScraper
from akd.tools.scrapers.pdf_scrapers import SimplePDFScraper
from akd.tools.scrapers.web_scrapers import Crawl4AIWebScraper, SimpleWebScraper

from ._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)


class SearchPipelineConfig(SearchToolConfig):
    """Configuration for the SearchPipeline."""

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

    enable_scraping: bool = Field(
        default=True,
        description="Whether to enable the scraping step in the pipeline",
    ) 

    scraping_timeout: int = Field(
        default=30,
        description="Timeout in seconds for individual scraping operations",
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


class SearchPipeline(SearchTool):
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
    config_schema = SearchPipelineConfig

    def _default_research_article_resolver(self, debug: bool = False) -> ResearchArticleResolver:
        return ResearchArticleResolver(
        PDFUrlResolver(debug=debug),
        ArxivResolver(debug=debug),
        ADSResolver(debug=debug),
        DOIResolver(debug=debug),
        CrossRefDoiResolver(debug=debug),
        UnpaywallResolver(debug=debug),
        debug=debug,
        )
    

    def default_scraper(self, debug: bool = False) -> ScraperToolBase:
        return CompositeScraper(
            DoclingScraper(debug=debug),
            Crawl4AIWebScraper(debug=debug),
            SimpleWebScraper(debug=debug),
            SimplePDFScraper(debug=debug),
        )


    def __init__(
        self,
        search_tool: SearchTool,
        resolver: Optional[BaseArticleResolver] = None,
        scraper: Optional[ScraperToolBase] = None,
        config: Optional[SearchPipelineConfig] = None,
        debug: bool = False,
    ):
        """
        Initialize the SearchPipeline.

        Args:
            search_tool: The underlying search tool (e.g., SearxNG, Semantic Scholar)
            resolver: Article resolver for open access URLs (defaults to ResearchArticleResolver)
            scraper: Content scraper (defaults to CompositeScraper)
            config: Pipeline configuration
            debug: Enable debug logging
        """
        config = config or SearchPipelineConfig()
        super().__init__(config, debug)

        self.search_tool = search_tool
        self.resolver = resolver or self._default_research_article_resolver(debug=debug)
        self.scraper = scraper or self.default_scraper(debug=debug)

        if debug:
            logger.debug("Initialized SearchPipeline with:")
            logger.debug(f"  - Search tool: {search_tool.__class__.__name__}")
            logger.debug(f"  - Resolver: {self.resolver.__class__.__name__}")
            logger.debug(f"  - Scraper: {self.scraper.__class__.__name__}")

    async def _resolve_essential_metadata(self, result: SearchResultItem) -> Optional[ResolverOutputSchema]:
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
                self.resolver.input_schema(**result.model_dump()),
            )

            return resolver_output

        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to resolve URL {result.url}: {e}")
            # Fall back to pdf_url or original url
            return ResolverOutputSchema(**result.model_dump())

    async def _scrape_content(
        self,
        url: str,
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
            # result.resolved_url if it exists, otherwise use original URL
            scraper_output = await asyncio.wait_for(
                self.scraper.arun(self.scraper.input_schema(url=url)),
                timeout=self.scraping_timeout,
            )

            if scraper_output.content and scraper_output.content.strip():
                content = scraper_output.content.strip()

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

    async def _process_single_result(
        self,
        result: SearchResultItem,
    ) -> SearchResultItem:
        """
        Process a single search result through the full pipeline.

        Args:
            result: Original search result

        Returns:
            Enhanced search result with optional scraped content
        """
        try:
            # Step 1: URL resolution (always performed)
            resolved_result = await self._resolve_essential_metadata(result)

            scraped_content = None
            if self.enable_scraping:
                scraping_url = resolved_result.resolved_url if hasattr(resolved_result, 'resolved_url') and resolved_result.resolved_url else resolved_result.url
                scraped_content = await self._scrape_content(scraping_url)
            else:
                if self.debug:
                    logger.debug(f"Skipping scraping for {result.title}")

            # Step 3: Create enhanced result
            search_item_data = resolved_result.model_dump(
                include=set(SearchResultItem.model_fields.keys())
            )
            enhanced_result = SearchResultItem(**search_item_data)

            # Handle content based on what was performed
            if scraped_content:
                if self.include_original_content and result.content:
                    # Combine original and scraped content
                    enhanced_result.content = (
                        f"{result.content}\n\n--- FULL TEXT ---\n\n{scraped_content}"
                    )
                else:
                    # Replace with scraped content
                    enhanced_result.content = scraped_content

            # Add metadata about what was performed
            if not enhanced_result.extra:
                enhanced_result.extra = {}
                
            enhanced_result.extra.update({
                "scraping_performed": self.enable_scraping,
                "full_text_scraped": scraped_content is not None,
                "resolver_used": resolved_result.resolvers if hasattr(resolved_result, 'resolvers') else None,
                "resolved_url": resolved_result.resolved_url if hasattr(resolved_result, 'resolved_url') else None,
            })
                
            if self.enable_scraping:
                enhanced_result.extra["scraper_used"] = self.scraper.__class__.__name__
                if scraped_content:
                    scraping_url = resolved_result.resolved_url if hasattr(resolved_result, 'resolved_url') and resolved_result.resolved_url else resolved_result.url
                    enhanced_result.extra["scraped_url"] = scraping_url
                else:
                    scraping_url = resolved_result.resolved_url if hasattr(resolved_result, 'resolved_url') and resolved_result.resolved_url else resolved_result.url
                    enhanced_result.extra["scraping_attempted_url"] = scraping_url

            return enhanced_result

        except Exception as e:
            if self.debug:
                logger.error(f"Error processing result {result.title}: {e}")

            # Return original result with error metadata
            enhanced_result = result.model_copy()
            if not enhanced_result.extra:
                enhanced_result.extra = {}
            enhanced_result.extra.update({
                "scraping_performed": self.enable_scraping,
                "full_text_scraped": False,
                "processing_error": str(e),
            })

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
                f" Starting SearchPipeline for {len(params.queries)} queries",
            )
            logger.info(f"Scraping enabled: {self.enable_scraping}")

        # Step 1: Get initial search result schema
        params_search = self.search_tool.input_schema(**params.model_dump())
        search_results = await self.search_tool.arun(params_search, **kwargs)

        if not search_results.results:
            if self.debug:
                logger.info("No search results to process")
            return search_results

        if self.debug:
            logger.info(f"Processing {len(search_results.results)} search results")

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
        if self.enable_scraping:
            successful_scrapes = sum(
                1
                for result in enhanced_results
                if result.extra and result.extra.get("full_text_scraped", False)
            )

            if self.debug:
                logger.info(
                    f"Successfully scraped {successful_scrapes}/{len(enhanced_results)} results",
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
