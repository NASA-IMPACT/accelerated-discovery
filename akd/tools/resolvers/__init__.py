"""
URL Resolver package for AKD project.

This package contains various resolvers for transforming URLs to their final destinations,
such as DOI URLs, PDF URLs, or publisher URLs. It supports multiple input sources including
direct URLs, PDF URLs, and DOI identifiers from search results.
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
from .crossref_doi import (
    CrossRefDoiResolver,
    CrossRefDoiResolverConfig,
    CrossRefDoiResolverInputSchema,
    CrossRefDoiResolverOutputSchema,
    
)

# Composite resolver
from .composite import ResearchArticleResolver
from .identity import IdentityResolver

# Specialized resolvers
from .specialized import DOIResolver, PDFUrlResolver
from .unpaywall import UnpaywallResolver

__all__ = [
    # Base classes and schemas
    "BaseArticleResolver",
    "ResolverInputSchema",
    "ResolverOutputSchema",
    "ArticleResolverConfig",
    # Specialized resolvers
    "PDFUrlResolver",
    "DOIResolver",
    "UnpaywallResolver",
    # Individual resolvers
    "IdentityResolver",
    "ArxivResolver",
    "ADSResolver",
     # Cross ref DOI resolvers
    "CrossRefDoiResolver", 
    "CrossRefDoiResolverConfig",
    "CrossRefDoiResolverInputSchema",
    "CrossRefDoiResolverOutputSchema",
    # Composite resolver
    "ResearchArticleResolver",
    
]
