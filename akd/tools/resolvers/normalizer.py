"""
SearchResultItem normalizer using article resolvers.

This module provides a tool for normalizing and resolving SearchResultItems
before deduplication and ranking (RRF).
"""

import asyncio

from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig

from ._base import BaseArticleResolver


class SearchResultNormalizerConfig(BaseToolConfig):
    """Configuration for SearchResultItemNormalizer."""

    pass


class SearchResultNormalizerInputSchema(InputSchema):
    """Input schema for SearchResultItemNormalizer."""

    results: list[SearchResultItem] = Field(
        ...,
        description="List of search results to normalize.",
    )


class SearchResultNormalizerOutputSchema(OutputSchema):
    """Output schema for SearchResultItemNormalizer."""

    results: list[SearchResultItem] = Field(
        ...,
        description="List of normalized search results.",
    )
    stats: dict[str, int] = Field(
        default_factory=dict,
        description="Statistics: total, resolved, failed, etc.",
    )


class SearchResultItemNormalizer(
    BaseTool[SearchResultNormalizerInputSchema, SearchResultNormalizerOutputSchema],
):
    """
    Normalizes and resolves SearchResultItems using article resolvers.

    This tool performs resolver-based normalization:
    - URL resolution (arxiv → PDF, unpaywall → full text, etc.)
    - DOI extraction from URLs and resolution via CrossRef (by title/authors)
    - Metadata enrichment (PDF URLs, authors, etc.)
    - Canonical URL construction (DOI URLs)

    The resolver (e.g., ResearchArticleResolver) handles the full pipeline.

    Attributes:
        input_schema: SearchResultNormalizerInputSchema
        output_schema: SearchResultNormalizerOutputSchema
        config_schema: SearchResultNormalizerConfig
        resolver: Article resolver injected via __init__

    Usage:
        >>> from akd.tools.resolvers.composite import ResearchArticleResolver
        >>> from akd.tools.resolvers.specialized import DOIResolver
        >>> from akd.tools.resolvers.arxiv import ArxivResolver
        >>>
        >>> resolver = ResearchArticleResolver(
        ...     DOIResolver(),  # Extracts DOI from URL, creates canonical URL
        ...     ArxivResolver(),  # Resolves arxiv URLs
        ... )
        >>> normalizer = SearchResultItemNormalizer(resolver=resolver, debug=True)
        >>>
        >>> input_data = SearchResultNormalizerInputSchema(results=search_results)
        >>> output = await normalizer.arun(input_data)
        >>> normalized_results = output.results
    """

    input_schema = SearchResultNormalizerInputSchema
    output_schema = SearchResultNormalizerOutputSchema
    config_schema = SearchResultNormalizerConfig

    def __init__(
        self,
        resolver: BaseArticleResolver | None = None,
        config: SearchResultNormalizerConfig | None = None,
        debug: bool = False,
    ):
        """
        Initialize the normalizer with dependency injection.

        Args:
            resolver: Article resolver to use for normalization. If None, no normalization is performed.
            config: Configuration (currently unused, for future extensibility).
            debug: Enable debug logging.
        """
        super().__init__(config=config, debug=debug)
        self.resolver = resolver

    async def _normalize_single(self, result: SearchResultItem) -> SearchResultItem:
        """
        Normalize a single SearchResultItem using the configured resolver.

        Args:
            result: SearchResultItem to normalize.

        Returns:
            Normalized SearchResultItem.
        """
        if not self.resolver:
            return result

        try:
            # Use resolver's input schema
            resolver_input = self.resolver.input_schema(**result.model_dump())

            # Resolve
            resolved = await self.resolver.arun(resolver_input)

            if resolved:
                # Update result with resolved data (exclude resolver metadata)
                result = SearchResultItem(**resolved.model_dump(exclude={"resolvers"}))

                if self.debug:
                    logger.debug(
                        f"Resolved: url={resolved.url}, doi={resolved.doi}, resolvers={resolved.resolvers}",
                    )
        except Exception as e:
            if self.debug:
                logger.error(f"Error resolving result for url={result.url}: {e}")
            # Return original result if resolution fails

        return result

    async def _arun(
        self,
        params: SearchResultNormalizerInputSchema,
    ) -> SearchResultNormalizerOutputSchema:
        """
        Normalize a batch of SearchResultItems.

        Args:
            params: Input parameters containing list of results.

        Returns:
            Output containing normalized results and statistics.
        """
        if not params.results:
            return self.output_schema(
                results=[],
                stats={"total": 0, "resolved": 0, "failed": 0},
            )

        # Normalize all results in parallel
        normalized = await asyncio.gather(
            *[self._normalize_single(result) for result in params.results],
            return_exceptions=True,
        )

        # Filter out exceptions and collect statistics
        successful = []
        failed_count = 0

        for i, result in enumerate(normalized):
            if isinstance(result, Exception):
                if self.debug:
                    logger.error(f"Failed to normalize result {i}: {result}")
                # Keep original result if normalization fails
                successful.append(params.results[i])
                failed_count += 1
            else:
                successful.append(result)

        stats = {
            "total": len(params.results),
            "resolved": len(successful) - failed_count,
            "failed": failed_count,
        }

        if self.debug:
            logger.info(f"Normalization complete: {stats}")

        return self.output_schema(results=successful, stats=stats)


__all__ = [
    "SearchResultItemNormalizer",
    "SearchResultNormalizerConfig",
    "SearchResultNormalizerInputSchema",
    "SearchResultNormalizerOutputSchema",
]
