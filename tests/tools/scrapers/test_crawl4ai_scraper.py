"""
Blackbox tests for Crawl4AI web scraper.

Tests cover:
1. Default mode (local Playwright) - should work if Playwright is installed
2. Docker mode with Docker unavailable - should raise appropriate exception
3. Docker mode with Docker available - should work if Docker service is running
"""

import httpx
import pytest

from akd.tools.scrapers import (
    Crawl4AIScraperConfig,
    Crawl4AIWebScraper,
    ScraperToolInputSchema,
)


def is_docker_cdp_available(host: str = "127.0.0.1", port: int = 9222) -> bool:
    """
    Check if Docker CDP service is available.

    Args:
        host: Docker CDP host
        port: Docker CDP port

    Returns:
        True if CDP service is reachable, False otherwise
    """
    try:
        # Try to connect to the CDP HTTP endpoint
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"http://{host}:{port}/json/version")
            return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def is_playwright_installed() -> bool:
    """
    Check if Playwright is installed locally.

    Returns:
        True if Playwright is available, False otherwise
    """
    try:
        from playwright.async_api import async_playwright  # noqa

        return True
    except ImportError:
        return False


@pytest.mark.asyncio
class TestCrawl4AIScraperLocalMode:
    """Tests for Crawl4AI scraper in default/local Playwright mode."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with default configuration (local mode)."""
        config = Crawl4AIScraperConfig(
            use_docker=False,
            headless=True,
        )
        return Crawl4AIWebScraper(config)

    @pytest.fixture
    def test_url(self):
        """Test URL that should be scrapable."""
        return "https://example.com"

    @pytest.mark.skipif(
        not is_playwright_installed(),
        reason="Playwright not installed - install with: playwright install chromium",
    )
    async def test_local_mode_scrapes_successfully(self, scraper, test_url):
        """Test that scraper works in local mode when Playwright is installed."""
        params = ScraperToolInputSchema(url=test_url)
        result = await scraper.arun(params)

        # Verify we got content
        assert result.content is not None
        assert len(result.content) > 0
        assert isinstance(result.content, str)

        # Verify content looks reasonable (contains expected text)
        assert "example" in result.content.lower()

    @pytest.mark.skipif(
        not is_playwright_installed(),
        reason="Playwright not installed",
    )
    async def test_local_mode_handles_invalid_url(self, scraper):
        """Test that scraper handles invalid URLs gracefully."""
        params = ScraperToolInputSchema(
            url="https://this-domain-definitely-does-not-exist-12345.com",
        )

        with pytest.raises(Exception):  # Should raise some kind of error
            await scraper.arun(params)

    @pytest.mark.skipif(
        not is_playwright_installed(),
        reason="Playwright not installed",
    )
    async def test_local_mode_rejects_pdf_urls(self, scraper):
        """Test that scraper rejects PDF URLs."""
        params = ScraperToolInputSchema(url="https://example.com/document.pdf")

        with pytest.raises(RuntimeError, match="Can't parse url with PDF"):
            await scraper.arun(params)

    async def test_local_mode_without_playwright_fails(self):
        """Test that scraper fails gracefully when Playwright is not installed."""
        if is_playwright_installed():
            pytest.skip("Playwright is installed - cannot test failure case")

        config = Crawl4AIScraperConfig(use_docker=False)
        scraper = Crawl4AIWebScraper(config)
        params = ScraperToolInputSchema(url="https://example.com")

        with pytest.raises(Exception):  # Should raise ImportError or similar
            await scraper.arun(params)


@pytest.mark.asyncio
class TestCrawl4AIScraperDockerMode:
    """Tests for Crawl4AI scraper in Docker CDP mode."""

    @pytest.fixture
    def docker_config(self):
        """Create scraper configuration for Docker mode."""
        return Crawl4AIScraperConfig(
            use_docker=True,
            playwright_cdp_url="ws://127.0.0.1:9222",
            headless=True,
        )

    @pytest.fixture
    def test_url(self):
        """Test URL that should be scrapable."""
        return "https://example.com"

    @pytest.mark.skipif(
        is_docker_cdp_available(),
        reason="Docker CDP is available - cannot test unavailable case",
    )
    async def test_docker_mode_fails_when_docker_unavailable(
        self,
        docker_config,
        test_url,
    ):
        """Test that scraper fails appropriately when Docker CDP is not available."""
        scraper = Crawl4AIWebScraper(docker_config)
        params = ScraperToolInputSchema(url=test_url)

        # Should raise an exception due to CDP unavailability
        with pytest.raises(Exception):  # Could be RuntimeError, ConnectionError, etc.
            await scraper.arun(params)

    @pytest.mark.skipif(
        not is_docker_cdp_available(),
        reason="Docker CDP not available - start with: docker run -d --name playwright-cdp -p 9222:3000 -e 'ENABLE_API_GET=true' browserless/chrome:latest",
    )
    async def test_docker_mode_scrapes_successfully(self, docker_config, test_url):
        """Test that scraper works in Docker mode when CDP service is available."""
        scraper = Crawl4AIWebScraper(docker_config)
        params = ScraperToolInputSchema(url=test_url)
        result = await scraper.arun(params)

        # Verify we got content
        assert result.content is not None
        assert len(result.content) > 0
        assert isinstance(result.content, str)

        # Verify content looks reasonable
        assert "example" in result.content.lower()

    @pytest.mark.skipif(
        not is_docker_cdp_available(),
        reason="Docker CDP not available",
    )
    async def test_docker_mode_handles_invalid_url(self, docker_config):
        """Test that Docker mode handles invalid URLs gracefully."""
        scraper = Crawl4AIWebScraper(docker_config)
        params = ScraperToolInputSchema(
            url="https://this-domain-definitely-does-not-exist-12345.com",
        )

        with pytest.raises(Exception):  # Should raise some kind of error
            await scraper.arun(params)

    @pytest.mark.skipif(
        not is_docker_cdp_available(),
        reason="Docker CDP not available",
    )
    async def test_docker_mode_rejects_pdf_urls(self, docker_config):
        """Test that Docker mode rejects PDF URLs."""
        scraper = Crawl4AIWebScraper(docker_config)
        params = ScraperToolInputSchema(url="https://example.com/document.pdf")

        with pytest.raises(RuntimeError, match="Can't parse url with PDF"):
            await scraper.arun(params)

    @pytest.mark.skipif(
        not is_docker_cdp_available(),
        reason="Docker CDP not available",
    )
    async def test_docker_mode_custom_port(self):
        """Test that scraper can connect to Docker CDP on custom port."""
        # This test would require starting Docker on a different port
        # For now, just test that configuration accepts custom port
        config = Crawl4AIScraperConfig(
            use_docker=True,
            playwright_cdp_url="ws://127.0.0.1:9223",
            headless=True,
        )
        scraper = Crawl4AIWebScraper(config)

        # Verify configuration was set correctly
        assert scraper.playwright_cdp_url == "ws://127.0.0.1:9223"


@pytest.mark.asyncio
class TestCrawl4AIScraperConfiguration:
    """Tests for Crawl4AI scraper configuration options."""

    def test_default_config_is_local_mode(self):
        """Test that default configuration uses local mode."""
        config = Crawl4AIScraperConfig()
        assert config.use_docker is False

    def test_config_accepts_docker_mode(self):
        """Test that configuration accepts Docker mode settings."""
        config = Crawl4AIScraperConfig(
            use_docker=True,
            playwright_cdp_url="ws://127.0.0.1:9222",
            headless=True,
            browser_type="chromium",
        )

        assert config.use_docker is True
        assert config.playwright_cdp_url == "ws://127.0.0.1:9222"
        assert config.headless is True
        assert config.browser_type == "chromium"

    def test_config_accepts_different_browsers(self):
        """Test that configuration accepts different browser types."""
        for browser in ["chromium", "firefox", "webkit"]:
            config = Crawl4AIScraperConfig(browser_type=browser)
            assert config.browser_type == browser

    def test_scraper_inherits_config(self):
        """Test that scraper inherits configuration correctly."""
        config = Crawl4AIScraperConfig(
            use_docker=True,
            playwright_cdp_url="ws://test.example.com:9222",
            headless=False,
            browser_type="firefox",
        )
        scraper = Crawl4AIWebScraper(config)

        assert scraper.use_docker is True
        assert scraper.playwright_cdp_url == "ws://test.example.com:9222"
        assert scraper.headless is False
        assert scraper.browser_type == "firefox"


@pytest.mark.asyncio
class TestCrawl4AIScraperCDPEndpoint:
    """Tests for CDP endpoint discovery logic."""

    @pytest.mark.skipif(
        not is_docker_cdp_available(),
        reason="Docker CDP not available",
    )
    async def test_cdp_endpoint_discovery(self):
        """Test that CDP endpoint discovery works correctly."""
        scraper = Crawl4AIWebScraper(
            Crawl4AIScraperConfig(
                use_docker=True,
                playwright_cdp_url="ws://127.0.0.1:9222",
            ),
        )

        # Test the endpoint discovery
        ws_url = await scraper._get_cdp_endpoint("ws://127.0.0.1:9222")

        # Should return a WebSocket URL
        assert ws_url.startswith("ws://")
        assert "127.0.0.1" in ws_url or "localhost" in ws_url

    async def test_cdp_endpoint_discovery_fails_gracefully(self):
        """Test that CDP endpoint discovery handles failures gracefully."""
        scraper = Crawl4AIWebScraper(
            Crawl4AIScraperConfig(
                use_docker=True,
                playwright_cdp_url="ws://127.0.0.1:9999",  # Non-existent port
            ),
        )

        # Should return the base URL as fallback (with warning logged)
        ws_url = await scraper._get_cdp_endpoint("ws://127.0.0.1:9999")
        assert ws_url == "ws://127.0.0.1:9999"


# Utility test to check current environment
@pytest.mark.asyncio
async def test_environment_check():
    """
    Utility test to check current testing environment.

    This test always passes but prints useful diagnostic information.
    """
    playwright_available = is_playwright_installed()
    docker_available = is_docker_cdp_available()

    print("\n=== Test Environment ===")
    print(f"Playwright installed: {playwright_available}")
    print(f"Docker CDP available: {docker_available}")

    if not playwright_available:
        print("\nTo install Playwright:")
        print("  pip install playwright")
        print("  playwright install chromium")

    if not docker_available:
        print("\nTo start Docker CDP:")
        print("  docker run -d --name playwright-cdp -p 9222:3000 \\")
        print("      -e 'ENABLE_API_GET=true' browserless/chrome:latest")

    print("========================\n")

    # Always pass
    assert True
