from typing import Optional

from akd.agents.factory import create_multi_rubric_relevancy_agent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.tools.relevancy import EnhancedRelevancyCheckerConfig, RubricWeights

from .resolvers import (
    ADSResolver,
    ArticleResolverConfig,
    ArxivResolver,
    BaseArticleResolver,
    IdentityResolver,
    ResearchArticleResolver,
)
from .scrapers import ScraperToolBase, ScraperToolConfig
from .scrapers.composite import CompositeScraper
from .scrapers.pdf_scrapers import SimplePDFScraper
from .scrapers.web_scrapers import Crawl4AIWebScraper, SimpleWebScraper
from .search import SearchTool, SearxNGSearchTool, SearxNGSearchToolConfig


def create_default_scraper(
    config: Optional[ScraperToolConfig] = None,
    debug: bool = False,
) -> ScraperToolBase:
    config = config or ScraperToolConfig()
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
