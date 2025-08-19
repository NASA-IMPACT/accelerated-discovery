import os
import tempfile
from typing import Dict
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import Field

from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .omni import DoclingScraper, OmniScraperInputSchema
from .resolvers import (
    BaseArticleResolver,
    ArxivResolver,
    ADSResolver,
    IdentityResolver
)


class WaterfallScraperConfig(ScraperToolConfig):
    """Configuration for WaterfallScraper."""
    
    request_timeout: float = Field(
        default=30.0,
        description="Timeout for HTTP requests in seconds"
    )
    follow_redirects: bool = Field(
        default=True,
        description="Whether to follow HTTP redirects"
    )
    browser_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
        description="Browser headers for HTTP requests"
    )
    enable_prefetching: bool = Field(
        default=True,
        description="Enable prefetching with browser headers to avoid 403 errors"
    )


class WaterfallScraper(ScraperToolBase):
    """
    Waterfall scraper that tries multiple scrapers in sequence with URL resolution.
    
    Features:
    - Uses existing resolvers (ArxivResolver, ADSResolver, etc.) for URL resolution
    - Waterfall execution: tries each scraper until one succeeds
    - Simple and focused implementation
    """

    input_schema = ScraperToolInputSchema
    output_schema = ScraperToolOutputSchema
    config_schema = WaterfallScraperConfig

    def __init__(
        self,
        *scrapers: ScraperToolBase,
        resolvers: list[BaseArticleResolver] | None = None,
        config: WaterfallScraperConfig | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config or WaterfallScraperConfig(), debug=debug)
        self.scrapers = scrapers
        
        # Initialize resolvers - use provided ones or create defaults
        if resolvers is None:
            self.resolvers = [
                ArxivResolver(debug=debug),
                ADSResolver(debug=debug),
                IdentityResolver(debug=debug)  # Always resolves to original URL
            ]
        else:
            self.resolvers = list(resolvers)

    async def _resolve_url(self, url_str: str) -> str:
        """Resolve URL using existing resolver architecture."""
        # Try existing resolvers (ArXiv, ADS, etc.)
        for resolver in self.resolvers:
            try:
                if resolver.validate_url(url_str):
                    resolved_url = await resolver.resolve(url_str)
                    if resolved_url and resolved_url != url_str:
                        if self.debug:
                            logger.debug(f"Resolver {resolver.__class__.__name__} resolved: {url_str} -> {resolved_url}")
                        return resolved_url
            except Exception as e:
                if self.debug:
                    logger.debug(f"Resolver {resolver.__class__.__name__} failed: {e}")
                continue
        
        # Return original URL if no resolver worked
        return url_str

    async def _prefetch_and_scrape(self, url_str: str) -> ScraperToolOutputSchema | None:
        """Prefetch content with browser headers and scrape using DoclingScraper."""
        if not self.config.enable_prefetching:
            return None
            
        tmp_path: str | None = None
        parsed = urlparse(url_str)
        headers = self.config.browser_headers.copy()
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}"
        
        try:
            async with httpx.AsyncClient(
                follow_redirects=self.config.follow_redirects,
                timeout=self.config.request_timeout,
                headers=headers,
            ) as client:
                resp = await client.get(url_str)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                suffix = (
                    ".pdf"
                    if "pdf" in ctype or url_str.lower().endswith(".pdf")
                    else ".html"
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(resp.content)
                tmp.flush()
                tmp.close()
                tmp_path = tmp.name

            # Use DoclingScraper on the downloaded file
            docling = DoclingScraper(debug=self.debug)
            out = await docling.arun(OmniScraperInputSchema(url=tmp_path))
            if out.content and out.content.strip():
                return out
                
        except Exception as e:
            if self.debug:
                logger.debug(f"Prefetch failed for {url_str}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return None

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **_kwargs,
    ) -> ScraperToolOutputSchema:
        original_url = str(params.url)
        
        # Step 1: Resolve URL using existing resolvers
        resolved_url = await self._resolve_url(original_url)
        if self.debug and resolved_url != original_url:
            logger.debug(f"URL resolved: {original_url} -> {resolved_url}")
        
        # Step 2: Try prefetch strategy first if enabled
        if self.config.enable_prefetching:
            prefetch_result = await self._prefetch_and_scrape(resolved_url)
            if prefetch_result and prefetch_result.content.strip():
                if self.debug:
                    logger.debug("Prefetch strategy succeeded")
                prefetch_result.metadata.url = original_url  # Keep original URL
                return prefetch_result
        
        # Step 3: Fallback to waterfall scraper strategy
        resolved_params = ScraperToolInputSchema(url=resolved_url, include_links=params.include_links)
        
        result = ScraperToolOutputSchema(
            content="",
            metadata=ScrapedMetadata(
                url=original_url,
                title="",
                query="",
            ),
        )
        
        for scraper in self.scrapers:
            try:
                if self.debug:
                    logger.debug(
                        f"Running scraper={scraper.__class__.__name__} for {resolved_url}",
                    )

                result = await scraper.arun(scraper.input_schema(**resolved_params.model_dump()))
                if result.content.strip():
                    result.metadata.extra = result.metadata.extra or {}
                    result.metadata.extra["scraper"] = scraper.__class__.__name__
                    result.metadata.extra["scraper_config"] = scraper.config
                    result.metadata.extra["url_resolved"] = resolved_url != original_url
                    result.metadata.url = original_url  # Keep original URL in metadata
                    break
            except Exception as e:
                if self.debug:
                    logger.error(
                        f"Error running {scraper.__class__.__name__}: {str(e)}",
                    )
                continue
        return result