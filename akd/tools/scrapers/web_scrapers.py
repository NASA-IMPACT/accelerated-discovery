import re
from typing import TYPE_CHECKING, Any, List, Literal
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import ConfigDict, Field, computed_field

from ._base import (
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WebScraper,
    WebScraperToolConfig,
)

if TYPE_CHECKING:
    # Type-only imports; the runtime imports are deferred to call sites so that
    # `import akd.tools.scrapers` does not eagerly pull crawl4ai/playwright,
    # bs4, readability or markdownify. Install them via `akd[scrapers]`.
    from bs4 import BeautifulSoup
    from crawl4ai import AsyncWebCrawler, BrowserConfig


class SimpleWebScraper(WebScraper):
    """
    Tool for scraping webpage content and converting it to markdown format.
    """

    async def _fetch_webpage(self, url: str) -> str:
        """
        Fetches the webpage content with custom headers and proper error handling.

        Args:
            url (str): The URL to fetch.

        Returns:
            str: The HTML content of the webpage.

        Raises:
            HTTPError: If the HTTP request fails
            ValueError: If content length exceeds maximum
            RuntimeError: If the content is a PDF
            RequestException: For other request-related errors
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()

                # Check if response is actually a PDF based on Content-Type
                content_type = response.headers.get("content-type", "").lower()
                if self.debug:
                    logger.debug(f"Fetched URL: {url}. Headers: {response.headers}")
                if "application/pdf" in content_type:
                    raise RuntimeError(
                        f"URL returns PDF content (Content-Type: {content_type}), use PDF scraper instead: {url}",
                    )

                if len(response.content) > self.max_content_length:
                    raise ValueError(
                        f"Content length exceeds maximum of {self.max_content_length} bytes",
                    )

                return response.text

        except httpx.HTTPStatusError as http_err:
            from requests import HTTPError  # deferred: requests is in akd[scrapers]

            raise HTTPError(f"HTTP error occurred: {http_err}")
        except httpx.TimeoutException as timeout_err:
            from requests import (
                RequestException,  # deferred: requests is in akd[scrapers]
            )

            raise RequestException(f"Request timeout: {timeout_err}")
        except httpx.RequestError as req_err:
            from requests import (
                RequestException,  # deferred: requests is in akd[scrapers]
            )

            raise RequestException(f"Error fetching webpage: {req_err}")

    async def _clean_markdown(self, markdown: str) -> str:
        """
        Cleans up the markdown content by removing excessive whitespace and normalizing formatting.

        Args:
            markdown (str): Raw markdown content.

        Returns:
            str: Cleaned markdown content.
        """
        # Remove multiple blank lines
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        # Remove trailing whitespace
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
        # Ensure content ends with single newline
        markdown = markdown.strip() + "\n"
        return markdown

    async def _extract_main_content(self, soup: "BeautifulSoup") -> str:
        """
        Extracts the main content from the webpage using custom heuristics.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Main content HTML.
        """
        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Try to find main content container
        content_candidates = [
            soup.find("main"),
            soup.find(id=re.compile(r"content|main", re.I)),
            soup.find(class_=re.compile(r"content|main", re.I)),
            soup.find("article"),
        ]

        main_content = next(
            (candidate for candidate in content_candidates if candidate),
            None,
        )

        if not main_content:
            main_content = soup.find("body")

        return str(main_content) if main_content else str(soup)

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        """
        Runs the WebpageScraperTool with the given parameters.

        Args:
            params (WebpageScraperToolInputSchema): The input parameters for the tool.

        Returns:
            WebpageScraperToolOutputSchema: The output containing the markdown content and metadata.
        """

        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")

        html_content = await self._fetch_webpage(str(params.url))

        # Additional check: detect if content is actually PDF binary data
        # PDFs start with %PDF- magic bytes
        if html_content.startswith("%PDF-"):
            raise RuntimeError(
                f"Fetched content appears to be PDF binary data (starts with %PDF-): {params.url}",
            )

        # Parse HTML with BeautifulSoup (deferred: bs4 is in akd[scrapers])
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract main content using custom extraction
        main_content = await self._extract_main_content(soup)

        # Convert to markdown
        markdown_options = {
            "strip": ["script", "style"],
            "heading_style": "ATX",
            "bullets": "-",
            "wrap": True,
        }

        if not params.include_links:
            markdown_options["strip"].append("a")

        # deferred: markdownify is in akd[scrapers]
        from markdownify import markdownify

        markdown_content = markdownify(main_content, **markdown_options)

        # Clean up the markdown
        markdown_content = await self._clean_markdown(markdown_content)

        # Detect anti-bot/security checks in content
        self._validate_security_check(markdown_content, str(params.url))

        # Extract metadata (deferred: readability is in akd[scrapers])
        from readability import Document

        metadata = await self._extract_metadata(
            soup,
            Document(html_content),
            str(params.url),
        )

        return ScraperToolOutputSchema(
            content=markdown_content.strip(),
            metadata=metadata,
        )


class Crawl4AIScraperConfig(WebScraperToolConfig):
    """
    Configuration for Crawl4AI web scraper with Docker and local browser support.

    This configuration supports both local Playwright installation and Docker-based
    browser connections via Chrome DevTools Protocol (CDP).

    Docker Mode (Recommended for Servers):
        When use_docker=True, the scraper connects to a browser running in a
        Docker container instead of using a local Playwright installation.
        This is ideal for:
        - Servers where installing Playwright system dependencies is difficult
        - Containerized deployments
        - Environments requiring browser isolation

        To use Docker mode:
        1. Start browserless Chrome container:
           ```bash
           docker run -d --name playwright-cdp -p 9222:3000 \\
               -e "ENABLE_API_GET=true" browserless/chrome:latest
           ```

        2. Configure scraper:
           ```python
           config = Crawl4AIScraperConfig(
               use_docker=True,
               playwright_cdp_url="ws://127.0.0.1:9222",
               headless=True,
           )
           scraper = Crawl4AIWebScraper(config)
           ```

    Local Mode (Default):
        When use_docker=False (default), uses local Playwright installation.
        Requires Playwright to be installed with system dependencies:
        ```bash
        playwright install chromium
        ```

    Anti-Bot Protection:
        The scraper includes several features to bypass anti-bot systems like
        Radware Bot Manager, Cloudflare, and similar protections:
        - Stealth mode: Modifies browser fingerprints to avoid detection
        - User agent rotation: Randomizes user agents across requests
        - Human behavior simulation: Adds realistic delays and interactions
        - Non-headless mode: Runs visible browser (less detectable)

    Attributes:
        use_docker: Enable Docker browser connection via CDP (default: False)
        playwright_cdp_url: WebSocket URL for CDP connection (default: "ws://localhost:9222")
        fallback_to_local: Auto-fallback to local Playwright if Docker fails (default: True)
        browser_type: Browser engine - "chromium", "firefox", or "webkit" (default: "chromium")
        headless: Run browser without GUI (default: True for servers, set False for better anti-bot bypass)
        enable_stealth: Enable stealth mode to bypass bot detection (default: True)
        user_agent_mode: User agent strategy - "random" for rotation (default: "random")
        simulate_user: Simulate human-like behavior (default: True)
        wait_time: Delay after page load in seconds (default: 3.0)
        delay_before_return_html: Additional delay before extraction (default: 2.0)
        magic: Enhanced anti-detection mode (default: True)
        proxy_config: Proxy configuration dict with server, username, password (default: None)
        use_undetected_browser: Use UndetectedAdapter for extreme anti-bot (default: False, slower)
        security_check_indicators: Keywords to detect anti-bot pages (inherited from base)

    Fallback Behavior:
        By default, if Docker mode is enabled but fails, the scraper will automatically
        fallback to local Playwright (fallback_to_local=True). This ensures backward
        compatibility and reliability. Set fallback_to_local=False to strictly require
        Docker mode and fail if unavailable.

    See Also:
        - Docker setup guide: docs/docker-playwright-setup.md
        - Test script: test_docker_scraper.py
        - Usage examples: Crawl4AIWebScraper class docstring
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_docker: bool = Field(
        default=False,
        description="Use Playwright running in Docker container via CDP.",
    )
    playwright_cdp_url: str = Field(
        default="ws://localhost:9222",
        description="CDP endpoint URL for Docker Playwright connection.",
    )
    fallback_to_local: bool = Field(
        default=True,
        description="If Docker connection fails, automatically fallback to local Playwright.",
    )
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium",
        description="Browser type: chromium, firefox, or webkit.",
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode. Set to False for better anti-bot bypass (requires display).",
    )

    # Anti-bot bypass features
    enable_stealth: bool = Field(
        default=True,
        description="Enable stealth mode using playwright-stealth to modify browser fingerprints.",
    )
    user_agent_mode: Literal["default", "random"] = Field(
        default="random",
        description="User agent strategy: 'default' uses system UA, 'random' rotates user agents.",
    )
    simulate_user: bool = Field(
        default=True,
        description="Simulate human-like behavior with mouse movements and realistic interactions.",
    )
    wait_time: float = Field(
        default=3.0,
        description="Time in seconds to wait after page load before extraction.",
    )
    delay_before_return_html: float = Field(
        default=2.0,
        description="Additional delay in seconds before returning HTML content.",
    )
    magic: bool = Field(
        default=True,
        description="Enable enhanced anti-detection mode with advanced fingerprint evasion.",
    )
    proxy_config: dict | None = Field(
        default=None,
        description="Proxy configuration: {'server': 'http://...', 'username': '...', 'password': '...'}",
    )
    use_undetected_browser: bool = Field(
        default=False,
        description="Use UndetectedAdapter for extremely aggressive anti-bot systems (slower but more effective).",
    )

    # filter header and footer by default
    filter_header_footer: bool = Field(
        default=True,
        description="Filter out header and footer elements from content.",
    )

    excluded_tags: List[str] = Field(
        default=[
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
        ],
        description="HTML tags to exclude from content extraction.",
    )
    excluded_selector: str = Field(
        default=".header, .footer, .nav, .navigation, .navbar, .sidebar, "
        ".menu, .breadcrumb, .pagination, .ads, "
        ".advertisement, .social, .share, .comments, "
        ".related, .recommended, "
        "#header, #footer, #nav, #navigation, #sidebar, #menu, "
        "#ads, #advertisement, #comments, #social",
        description="CSS selectors to exclude from content extraction.",
    )

    @computed_field
    def _run_config(self) -> Any:
        # Return type is Any (not CrawlerRunConfig) so the config class can be
        # defined without crawl4ai installed; the crawl4ai import is deferred to
        # this body, which only runs when a Crawl4AI scraper is actually used.
        from crawl4ai import CrawlerRunConfig

        config_params = {
            "excluded_tags": self.excluded_tags if self.filter_header_footer else [],
            "excluded_selector": self.excluded_selector if self.filter_header_footer else "",
            "delay_before_return_html": self.delay_before_return_html,
            "simulate_user": self.simulate_user,
            "magic": self.magic,
            "user_agent_mode": self.user_agent_mode,
            "mean_delay": self.wait_time,  # Use wait_time as mean_delay for timing control
        }

        # Add proxy config if provided
        if self.proxy_config:
            config_params["proxy_config"] = self.proxy_config

        return CrawlerRunConfig(**config_params)


class Crawl4AIWebScraper(WebScraper):
    """
    Advanced web scraper using Crawl4AI with support for both local and Docker-based browsers.

    This scraper uses Crawl4AI library to fetch and extract content from web pages,
    with flexible browser backend options for different deployment scenarios.

    Features:
        - Clean markdown output from HTML content
        - JavaScript rendering support (via Playwright/Chrome)
        - Docker-based browser for containerized environments
        - Local Playwright for development
        - Configurable browser types (Chromium, Firefox, WebKit)
        - Headless and headed modes

    Usage:
        Basic usage with local Playwright:
        ```python
        from akd.tools.scrapers import Crawl4AIWebScraper, ScraperToolInputSchema

        scraper = Crawl4AIWebScraper()
        result = await scraper.arun(ScraperToolInputSchema(url="https://example.com"))
        print(result.content)  # Markdown content
        ```

        Docker-based browser (for servers without Playwright dependencies):
        ```python
        from akd.tools.scrapers import Crawl4AIWebScraper, Crawl4AIScraperConfig

        config = Crawl4AIScraperConfig(
            use_docker=True,
            playwright_cdp_url="ws://127.0.0.1:9222",
            headless=True,
        )
        scraper = Crawl4AIWebScraper(config)
        result = await scraper.arun(ScraperToolInputSchema(url="https://example.com"))
        ```

    Docker Setup:
        To use Docker mode, first start a browserless Chrome container:
        ```bash
        docker run -d --name playwright-cdp -p 9222:3000 \\
            -e "ENABLE_API_GET=true" browserless/chrome:latest
        ```

    Attributes:
        config_schema: Configuration schema class (Crawl4AIScraperConfig)
        use_docker: Whether to use Docker-based browser
        playwright_cdp_url: CDP WebSocket URL for Docker connection
        browser_type: Browser engine type
        headless: Run browser in headless mode
        debug: Enable debug logging

    Notes:
        - PDF URLs are not supported and will raise RuntimeError
        - Returns content in markdown format for better LLM processing
        - Docker mode recommended for production servers
        - Local mode recommended for development

    See Also:
        - Configuration: Crawl4AIScraperConfig
        - Docker setup: docs/docker-playwright-setup.md
        - Test script: test_docker_scraper.py
    """

    config_schema = Crawl4AIScraperConfig

    async def _get_cdp_endpoint(self, base_url: str) -> str:
        """
        Discover the full CDP WebSocket URL from the base endpoint.

        Args:
            base_url: Base CDP URL (e.g., "ws://127.0.0.1:9222")

        Returns:
            Full CDP WebSocket URL
        """
        # Convert ws:// to http:// for the JSON endpoint
        http_url = base_url.replace("ws://", "http://").replace("wss://", "https://")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{http_url}/json/version")
                response.raise_for_status()
                data = response.json()
                if self.debug:
                    logger.debug(f"CDP Version Data: {data}")

                ws_url = data.get("webSocketDebuggerUrl", "")

                # Browserless and some CDP servers return the full URL directly
                # But may use localhost instead of the actual host
                # Replace the hostname to match what we're connecting to
                if ws_url:
                    base_parsed = urlparse(base_url)
                    ws_parsed = urlparse(ws_url)

                    # Replace hostname if it's localhost/127.0.0.1
                    if ws_parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                        ws_url = ws_url.replace(
                            f"ws://{ws_parsed.hostname}",
                            f"ws://{base_parsed.hostname}",
                        )

                    return ws_url

                # Fallback: if no webSocketDebuggerUrl, just use the base_url
                return base_url

        except Exception as e:
            # If discovery fails, try using the base_url directly
            logger.warning(
                f"Failed to discover CDP endpoint from {http_url}: {e}. "
                f"Attempting to use base URL directly: {base_url}",
            )
            return base_url

    def _create_crawler(self, browser_config: "BrowserConfig") -> "AsyncWebCrawler":
        """
        Create AsyncWebCrawler with optional UndetectedAdapter.

        Args:
            browser_config: Browser configuration

        Returns:
            Configured AsyncWebCrawler instance
        """
        # deferred: crawl4ai/playwright are in akd[scrapers]
        from crawl4ai import AsyncWebCrawler, UndetectedAdapter
        from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

        if self.use_undetected_browser:
            adapter = UndetectedAdapter()
            strategy = AsyncPlaywrightCrawlerStrategy(
                browser_config=browser_config,
                browser_adapter=adapter,
            )
            return AsyncWebCrawler(crawler_strategy=strategy, config=browser_config)
        else:
            return AsyncWebCrawler(config=browser_config)

    async def fetch(self, url: str):
        # Try Docker mode first if configured
        if self.use_docker:
            try:
                return await self._fetch_with_docker(url)
            except Exception as e:
                if self.fallback_to_local:
                    logger.warning(
                        f"Docker CDP connection failed: {e}. Falling back to local Playwright.",
                    )
                    return await self._fetch_with_local(url)
                else:
                    raise

        # Use local mode
        return await self._fetch_with_local(url)

    async def _fetch_with_docker(self, url: str):
        """Fetch URL using Docker-based browser with anti-bot bypass."""
        # deferred: crawl4ai/playwright are in akd[scrapers]
        from crawl4ai import BrowserConfig

        # Discover the full CDP endpoint URL
        cdp_url = await self._get_cdp_endpoint(self.playwright_cdp_url)

        browser_config = BrowserConfig(
            browser_type=self.browser_type,
            headless=self.headless,
            verbose=self.debug,
            browser_mode="docker",
            cdp_url=cdp_url,
            use_managed_browser=True,
            enable_stealth=self.enable_stealth,
        )

        async with self._create_crawler(browser_config) as crawler:
            return await crawler.arun(url=url, config=self._run_config)

    async def _fetch_with_local(self, url: str):
        """Fetch URL using local Playwright with anti-bot bypass."""
        # deferred: crawl4ai/playwright are in akd[scrapers]
        from crawl4ai import BrowserConfig

        browser_config = BrowserConfig(
            browser_type=self.browser_type,
            headless=self.headless,
            verbose=self.debug,
            enable_stealth=self.enable_stealth,
        )

        async with self._create_crawler(browser_config) as crawler:
            return await crawler.arun(url=url, config=self._run_config)

    def _validate_http_response(self, crawl_result, url: str) -> None:
        """
        Validate HTTP response status and crawl success.

        Args:
            crawl_result: CrawlResult from crawl4ai
            url: URL being crawled

        Raises:
            HTTPError: If HTTP status code >= 400
            RuntimeError: If crawl was unsuccessful
        """
        # Check for HTTP errors (deferred: requests is in akd[scrapers])
        from requests import HTTPError

        if crawl_result.status_code and crawl_result.status_code >= 400:
            raise HTTPError(
                f"HTTP {crawl_result.status_code} error fetching {url}: "
                f"{crawl_result.error_message or 'Unknown error'}",
            )

        # Check if crawl was successful
        if not crawl_result.success:
            raise RuntimeError(
                f"Failed to crawl {url}: {crawl_result.error_message or 'Unknown error'}",
            )

    async def _arun(self, params: ScraperToolInputSchema, **kwargs) -> ScraperToolOutputSchema:
        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")

        crawl_result = await self.fetch(str(params.url))

        # Validate HTTP response
        self._validate_http_response(crawl_result, str(params.url))

        # Use Crawl4AI's built-in markdown
        markdown = crawl_result.markdown.strip()

        # Detect anti-bot/security checks in content
        self._validate_security_check(markdown, str(params.url))

        # Use original HTML for metadata extraction
        # deferred: bs4/readability are in akd[scrapers]
        from bs4 import BeautifulSoup
        from readability import Document

        soup = BeautifulSoup(crawl_result.html, "html.parser")
        metadata = await self._extract_metadata(soup, Document(crawl_result.html), str(params.url))

        return ScraperToolOutputSchema(
            content=markdown,
            metadata=metadata,
        )
