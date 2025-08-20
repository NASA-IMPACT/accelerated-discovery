"""
URL Resolver package for AKD project.

This package contains resolvers for transforming URLs to their final destinations,
such as DOI URLs, PDF URLs, or publisher URLs.
"""

# Base classes and schemas
from ._base import (
    ArticleResolverConfig,
    BaseArticleResolver,
    ResolverInputSchema,
    ResolverOutputSchema,
)

# Individual resolvers
from .ads import ADSResolver
from .arxiv import ArxivResolver
from .composite import ResearchArticleResolver
from .crossref_doi import CrossRefDoiResolver
from .specialized import DOIResolver, PDFUrlResolver

__all__ = [
    # Base classes and schemas
    "BaseArticleResolver",
    "ResolverInputSchema",
    "ResolverOutputSchema",
    "ArticleResolverConfig",
    # Individual resolvers
    "ADSResolver",
    "ArxivResolver",
    "ResearchArticleResolver",
    "CrossRefDoiResolver",
    "DOIResolver",
    "PDFUrlResolver",
]
