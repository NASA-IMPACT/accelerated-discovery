import re
from typing import Literal
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig
from loguru import logger
from markdownify import markdownify
from pydantic import Field
from readability import Document
from requests import HTTPError, RequestException

from ._base import (
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WebScraper,
    WebScraperToolConfig,
)


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
            RequestException: For other request-related errors
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()

                if len(response.content) > self.max_content_length:
                    raise ValueError(
                        f"Content length exceeds maximum of {self.max_content_length} bytes",
                    )

                return response.text

        except httpx.HTTPStatusError as http_err:
            raise HTTPError(f"HTTP error occurred: {http_err}")
        except httpx.TimeoutException as timeout_err:
            raise RequestException(f"Request timeout: {timeout_err}")
        except httpx.RequestError as req_err:
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

    async def _extract_main_content(self, soup: BeautifulSoup) -> str:
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

        # Parse HTML with BeautifulSoup
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

        markdown_content = markdownify(main_content, **markdown_options)

        # Clean up the markdown
        markdown_content = await self._clean_markdown(markdown_content)

        # Extract metadata
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

    Attributes:
        use_docker: Enable Docker browser connection via CDP (default: False)
        playwright_cdp_url: WebSocket URL for CDP connection (default: "ws://localhost:9222")
        fallback_to_local: Auto-fallback to local Playwright if Docker fails (default: True)
        browser_type: Browser engine - "chromium", "firefox", or "webkit" (default: "chromium")
        headless: Run browser without GUI (default: True)

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
        description="Run browser in headless mode.",
    )


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

    async def fetch(self, url: str):
        # Try Docker mode first if configured
        if self.use_docker:
            try:
                return await self._fetch_with_docker(url)
            except Exception as e:
                if self.fallback_to_local:
                    logger.warning(
                        f"Docker CDP connection failed: {e}. "
                        f"Falling back to local Playwright.",
                    )
                    return await self._fetch_with_local(url)
                else:
                    raise

        # Use local mode
        return await self._fetch_with_local(url)

    async def _fetch_with_docker(self, url: str):
        """Fetch URL using Docker-based browser."""
        # Discover the full CDP endpoint URL
        cdp_url = await self._get_cdp_endpoint(self.playwright_cdp_url)

        browser_config_params = {
            "browser_type": self.browser_type,
            "headless": self.headless,
            "verbose": self.debug,
            "browser_mode": "docker",
            "cdp_url": cdp_url,
            "use_managed_browser": True,
        }

        browser_config = BrowserConfig(**browser_config_params)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            return await crawler.arun(url=url)

    async def _fetch_with_local(self, url: str):
        """Fetch URL using local Playwright."""
        browser_config_params = {
            "browser_type": self.browser_type,
            "headless": self.headless,
            "verbose": self.debug,
        }

        browser_config = BrowserConfig(**browser_config_params)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            return await crawler.arun(url=url)

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        """
        Async version of the run method.
        """
        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")

        crawl_result = await self.fetch(str(params.url))

        html_content = crawl_result.html

        soup = BeautifulSoup(html_content, "html.parser")
        metadata = await self._extract_metadata(
            soup,
            Document(html_content),
            str(params.url),
        )

        return ScraperToolOutputSchema(
            content=crawl_result.markdown.strip(),
            metadata=metadata,
        )
