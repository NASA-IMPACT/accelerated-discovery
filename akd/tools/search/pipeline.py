from __future__ import annotations

import asyncio
from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import Field

from akd.structures import SearchResultItem
from akd.tools.link_relevancy_assessor import (
    LinkRelevancyAssessor,
    LinkRelevancyAssessorConfig,
)
from akd.tools.resolvers import (
    ADSResolver,
    ArxivResolver,
    BaseArticleResolver,
    CrossRefDoiResolver,
    DOIResolver,
    PDFUrlResolver,
    ResearchArticleResolver,
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


class SearchPipelineScrapingMode(str, Enum):
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    LINK_ASSESSMENT = "link_assessment"


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

    scraping_mode: SearchPipelineScrapingMode = Field(
        default=SearchPipelineScrapingMode.ALWAYS_ON,
        description="Mode for enabling scraping: always_on, always_off, link_assessment",
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
    min_successful_scrapes: int | None = Field(
        default=None,
        description="Minimum number of successful scrapes required (None = no minimum)",
    )

    # Link assessment configuration
    link_relevancy_assessor_config: LinkRelevancyAssessorConfig = Field(
        default_factory=LinkRelevancyAssessorConfig,
        description="Configuration for the LinkRelevancyAssessor when using LINK_ASSESSMENT mode",
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

    class ScrapingError(Exception):
        """Raised when scraping fails or times out in SearchPipeline."""

        def __init__(self, url: str, message: str):
            super().__init__(f"Scraping failed for {url}: {message}")
            self.url = url
            self.message = message

    @property
    def _default_research_article_resolver(self) -> ResearchArticleResolver:
        return ResearchArticleResolver(
            PDFUrlResolver(debug=self.debug),
            ArxivResolver(debug=self.debug),
            ADSResolver(debug=self.debug),
            DOIResolver(debug=self.debug),
            CrossRefDoiResolver(debug=self.debug),
            UnpaywallResolver(debug=self.debug),
            debug=self.debug,
        )

    @property
    def _default_scraper(self) -> ScraperToolBase:
        return CompositeScraper(
            DoclingScraper(debug=self.debug),
            Crawl4AIWebScraper(debug=self.debug),
            SimpleWebScraper(debug=self.debug),
            SimplePDFScraper(debug=self.debug),
        )

    def __init__(
        self,
        search_tool: SearchTool,
        resolver: BaseArticleResolver | None = None,
        scraper: ScraperToolBase | None = None,
        link_relevancy_assessor: LinkRelevancyAssessor | None = None,
        config: SearchPipelineConfig | None = None,
        debug: bool = False,
    ):
        """
        Initialize the SearchPipeline.

        Args:
            search_tool: The underlying search tool (e.g., SearxNG, Semantic Scholar)
            resolver: Article resolver for open access URLs (defaults to ResearchArticleResolver)
            scraper: Content scraper (defaults to CompositeScraper)
            link_relevancy_assessor: Link relevancy assessor (defaults to LinkRelevancyAssessor)
            config: Pipeline configuration
            debug: Enable debug logging
        """
        config = config or SearchPipelineConfig()
        super().__init__(config, debug)

        self.search_tool = search_tool
        self.resolver = resolver or self._default_research_article_resolver
        self.scraper = scraper or self._default_scraper
        self.link_relevancy_assessor = link_relevancy_assessor or LinkRelevancyAssessor(
            config=config.link_relevancy_assessor_config,
            debug=debug,
        )

        if debug:
            logger.debug("Initialized SearchPipeline with:")
            logger.debug(f"  - Search tool: {search_tool.__class__.__name__}")
            logger.debug(f"  - Resolver: {self.resolver.__class__.__name__}")
            logger.debug(f"  - Scraper: {self.scraper.__class__.__name__}")
            logger.debug(
                f"  - Link relevancy assessor: {self.link_relevancy_assessor.__class__.__name__}",
            )

    async def _resolve_essential_metadata(
        self,
        result: SearchResultItem,
    ) -> Optional[ResolverOutputSchema]:
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

    async def _scrape_content(self, url: str) -> Optional[str]:
        """
        Scrape full text content from a URL.

        Args:
            url: URL to scrape

        Returns:
            Scraped content or None if scraping fails (unless fail_on_scraping_errors=True)
        """
        try:
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
                msg = "No content scraped"
                if self.debug:
                    logger.warning(f"{msg} from {url}")
                if self.fail_on_scraping_errors:
                    raise self.ScrapingError(url, msg)
                return None

        except asyncio.TimeoutError:
            msg = "Scraping timeout"
            if self.debug:
                logger.warning(f"{msg} for {url}")
            if self.fail_on_scraping_errors:
                raise self.ScrapingError(url, msg)
            return None

        except Exception as e:
            msg = str(e)
            if self.debug:
                logger.warning(f"Failed to scrape {url}: {msg}")
            if self.fail_on_scraping_errors:
                raise self.ScrapingError(url, msg)
            return None

    async def _assess_link_relevancy(
        self,
        results: list[SearchResultItem],
        query: str,
        domain_context: str | None = None,
    ) -> list[SearchResultItem]:
        """
        Assess the relevancy of search results using LinkRelevancyAssessor.

        Args:
            results: List of search results to assess
            query: The search query for relevancy assessment
            domain_context: Optional domain context for better assessment

        Returns:
            List of results with relevancy assessment metadata
        """
        if not results:
            return results

        try:
            assessment_input = self.link_relevancy_assessor.input_schema(
                search_results=results,
                original_query=query,
                domain_context=domain_context,
            )

            assessment_output = await self.link_relevancy_assessor.arun(
                assessment_input,
            )

            if self.debug:
                logger.debug(
                    f"Link relevancy assessment completed. "
                    f"Assessed: {len(assessment_output.assessed_results)}, "
                    f"High relevancy: {len(assessment_output.high_relevancy_results)}",
                )

            return assessment_output.assessed_results

        except Exception as e:
            if self.debug:
                logger.warning(f"Link relevancy assessment failed: {e}")
            # Return original results if assessment fails
            return results

    async def _process_single_result(
        self,
        result: SearchResultItem,
        query: str | None = None,
    ) -> SearchResultItem:
        """
        Process a single search result through the full pipeline.

        Args:
            result: Original search result
            query: Search query for relevancy assessment (required for LINK_ASSESSMENT mode)

        Returns:
            Enhanced search result with optional scraped content
        """
        try:
            # Step 1: URL resolution (always performed)
            resolved_result = await self._resolve_essential_metadata(result)

            scraping_url = resolved_result.url
            scraped_content = None
            should_scrape = False

            if self.scraping_mode == SearchPipelineScrapingMode.ALWAYS_ON:
                should_scrape = True
            elif self.scraping_mode == SearchPipelineScrapingMode.ALWAYS_OFF:
                should_scrape = False
            elif self.scraping_mode == SearchPipelineScrapingMode.LINK_ASSESSMENT:
                # For LINK_ASSESSMENT mode, check if result has been assessed
                should_scrape = result.extra.get("should_fetch_full_content", False)
                if "should_fetch_full_content" not in result.extra and query:
                    # If not assessed yet, perform assessment for this single result
                    assessed_results = await self._assess_link_relevancy(
                        [result],
                        query,
                    )
                    if assessed_results:
                        should_scrape = assessed_results[0].extra.get(
                            "should_fetch_full_content",
                            False,
                        )
                        # Update result with assessment metadata
                        result.score = getattr(assessed_results[0], "score", None)
                        if (
                            hasattr(assessed_results[0], "extra")
                            and "relevancy_assessment" in assessed_results[0].extra
                        ):
                            result.extra["relevancy_assessment"] = assessed_results[
                                0
                            ].extra["relevancy_assessment"]
                        result.extra["should_fetch_full_content"] = should_scrape

            if should_scrape:
                scraped_content = await self._scrape_content(scraping_url)
            else:
                if self.debug:
                    logger.debug(
                        f"Skipping scraping for {result.title} (mode: {self.scraping_mode})",
                    )

            # Step 3: Create enhanced result
            search_item_data = resolved_result.model_dump(
                include=set(SearchResultItem.model_fields.keys()),
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

            enhanced_result.extra.update(
                {
                    "scraping_mode": self.scraping_mode.value,
                    "scraping_performed": should_scrape,
                    "full_text_scraped": scraped_content is not None,
                    "resolver_used": resolved_result.resolvers
                    if hasattr(resolved_result, "resolvers")
                    else None,
                    "original_url": result.url,
                },
            )

            if should_scrape:
                enhanced_result.extra["scraper_used"] = self.scraper.__class__.__name__

                if scraped_content:
                    enhanced_result.extra["scraped_url"] = scraping_url
                else:
                    enhanced_result.extra["scraping_attempted_url"] = scraping_url

            return enhanced_result

        except Exception as e:
            if self.fail_on_scraping_errors and isinstance(e, self.ScrapingError):
                raise

            if self.debug:
                logger.error(f"Error processing result {result.title}: {e}")

            # Return original result with error metadata
            enhanced_result = result.model_copy()

            enhanced_result.extra.update(
                {
                    "scraping_mode": self.scraping_mode.value,
                    "scraping_performed": False,
                    "full_text_scraped": False,
                    "processing_error": str(e),
                    "scraping_attempted_url": scraping_url,
                },
            )

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
            logger.info(f"Scraping mode: {self.scraping_mode}")

        # Step 1: Get initial search result schema
        params_search = self.search_tool.input_schema(**params.model_dump())
        search_results = await self.search_tool.arun(params_search, **kwargs)

        if not search_results.results:
            if self.debug:
                logger.info("No search results to process")
            return search_results

        if self.debug:
            logger.info(f"Processing {len(search_results.results)} search results")

        # Step 2: Perform batch link assessment if needed
        results_to_process = search_results.results
        if self.scraping_mode == SearchPipelineScrapingMode.LINK_ASSESSMENT:
            # Get the main query for assessment (use first query if multiple)
            main_query = params.queries[0] if params.queries else ""
            results_to_process = await self._assess_link_relevancy(
                search_results.results,
                main_query,
            )

        # Step 3: Process results through the pipeline
        if self.parallel_processing:
            # Process in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent_scrapes)

            async def process_with_semaphore(result):
                async with semaphore:
                    return await self._process_single_result(result)

            enhanced_results = await asyncio.gather(
                *[process_with_semaphore(result) for result in results_to_process],
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
            for result in results_to_process:
                try:
                    enhanced_result = await self._process_single_result(result)
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    if self.fail_on_scraping_errors:
                        raise
                    if self.debug:
                        logger.error(f"Failed to process result {result.title}: {e}")

        # Step 4: Validate results
        if self.scraping_mode != SearchPipelineScrapingMode.ALWAYS_OFF:
            successful_scrapes = sum(
                1
                for result in enhanced_results
                if result.extra.get("full_text_scraped", False)
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
