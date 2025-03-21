from typing import Optional

from .scrapers.composite import CompositeWebScraper, ResearchArticleResolver
from .scrapers.pdf_scrapers import SimplePDFScraper
from .scrapers.resolvers import (
    ADSResolver,
    ArticleResolverConfig,
    ArxivResolver,
    BaseArticleResolver,
    IdentityResolver,
)
from .scrapers.web_scrapers import (
    Crawl4AIWebScraper,
    SimpleWebScraper,
    WebpageScraperToolConfig,
    WebScraperToolBase,
)
from .search import SearchTool, SearxNGSearchTool, SearxNGSearchToolConfig


def create_default_scraper(
    config: Optional[WebpageScraperToolConfig] = None,
    debug: bool = False,
) -> WebScraperToolBase:
    config = config or WebpageScraperToolConfig()
    config.debug = debug
    return CompositeWebScraper(
        SimpleWebScraper(config),
        Crawl4AIWebScraper(config),
        SimplePDFScraper(config),
        debug=debug,
    )


def create_default_search_tool(
    config: Optional[SearxNGSearchToolConfig] = None,
) -> SearchTool:
    config = config or SearxNGSearchToolConfig()
    return SearxNGSearchTool(config)


def create_default_article_resolver(
    config: Optional[ArticleResolverConfig] = None,
) -> BaseArticleResolver:
    config = config or ArticleResolverConfig()
    return ResearchArticleResolver(
        ADSResolver(config),
        ArxivResolver(config),
        IdentityResolver(config),
    )
