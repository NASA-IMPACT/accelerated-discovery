from typing import Optional

from akd.agents.factory import create_multi_rubric_relevancy_agent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.tools.relevancy import EnhancedRelevancyCheckerConfig, RubricWeights

from .scrapers.composite import CompositeScraper, ResearchArticleResolver
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
from .source_validator import (
    SourceValidator,
    SourceValidatorConfig,
    create_source_validator,
)


def create_default_scraper(
    config: Optional[WebpageScraperToolConfig] = None,
    debug: bool = False,
) -> WebScraperToolBase:
    config = config or WebpageScraperToolConfig()
    config.debug = debug
    return CompositeScraper(
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


def create_default_source_validator(
    config: Optional[SourceValidatorConfig] = None,
    whitelist_file_path: Optional[str] = None,
    max_concurrent_requests: int = 10,
    debug: bool = False,
) -> SourceValidator:
    """
    Create a source validator with default parameters.

    Args:
        config: Optional SourceValidatorConfig. If provided, other parameters are ignored.
        whitelist_file_path: Path to source whitelist JSON file. If None, uses default path in akd/docs/pubs_whitelist.json.
        max_concurrent_requests: Maximum number of concurrent API requests.
        debug: Enable debug logging.

    Returns:
        Configured SourceValidator instance.
    """
    if config is None:
        return create_source_validator(
            whitelist_file_path=whitelist_file_path,
            max_concurrent_requests=max_concurrent_requests,
            debug=debug,
        )
    return SourceValidator(config, debug=debug)


def create_strict_literature_config_for_relevancy(
    n_iter: int = 1,
    relevance_threshold: float = 0.7,
    swapping: bool = True,
    agent: MultiRubricRelevancyAgent | None = None,
    debug: bool = False,
) -> EnhancedRelevancyCheckerConfig:
    """Create a configuration optimized for strict literature search"""
    return EnhancedRelevancyCheckerConfig(
        rubric_weights=RubricWeights(
            topic_alignment=0.35,  # Higher weight for topic alignment
            content_depth=0.25,  # Higher weight for comprehensive content
            methodological_relevance=0.20,  # Important for academic literature
            evidence_quality=0.15,  # Quality evidence matters
            recency_relevance=0.03,  # Lower weight - some classic papers are still relevant
            scope_relevance=0.02,  # Lower weight - scope can vary
        ),
        relevance_threshold=relevance_threshold,  # Higher threshold for stricter filtering
        n_iter=n_iter,  # More iterations for better confidence
        swapping=swapping,  # Keep swapping for robustness
        agent=agent or create_multi_rubric_relevancy_agent(),
        debug=debug,
    )


def create_general_content_config_for_relevancy(
    n_iter: int = 1,
    relevance_threshold: float = 0.5,
    swapping: bool = False,
    agent: MultiRubricRelevancyAgent | None = None,
    debug: bool = False,
) -> EnhancedRelevancyCheckerConfig:
    """Create a configuration for general content relevance checking"""
    return EnhancedRelevancyCheckerConfig(
        rubric_weights=RubricWeights(
            topic_alignment=0.40,  # Highest weight for topic alignment
            scope_relevance=0.25,  # Important for general content
            content_depth=0.15,  # Moderate importance
            recency_relevance=0.10,  # Moderate importance
            evidence_quality=0.05,  # Lower for general content
            methodological_relevance=0.05,  # Lower for general content
        ),
        relevance_threshold=relevance_threshold,  # Lower threshold for general content
        n_iter=n_iter,  # Fewer iterations needed
        swapping=swapping,  # May not be necessary for general content
        agent=agent or create_multi_rubric_relevancy_agent(),
        debug=debug,
    )
