"""
Utilities for search results: normalization, resolution, and deduplication.

- normalize_results: Normalize and enrich search results using resolvers
- deduplicate_results: Simple cascaded deduplication (DOI → Title → URL)
- get_doi, normalize_title, normalize_url: Individual field normalizers
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from urllib.parse import urlparse

from loguru import logger
from pydantic import AnyUrl

from akd.structures import SearchResultItem
from akd.tools.resolvers._base import BaseArticleResolver
from akd.tools.resolvers.specialized import DOIResolver

# =============================================================================
# Normalization Utilities
# =============================================================================


async def normalize_results(
    results: list[SearchResultItem],
    resolver: BaseArticleResolver,
    debug: bool = False,
) -> list[SearchResultItem]:
    """
    Normalize search results in parallel using a resolver.

    This function enriches search results with:
    - DOI resolution and normalization
    - Open access URL finding
    - PDF URL extraction
    - Metadata enrichment (authors, publication dates, etc.)

    Args:
        results: List of search results to normalize
        resolver: Resolver instance to use for normalization
        debug: Enable debug logging

    Returns:
        Normalized search results with enriched metadata

    Examples:
        >>> from akd.tools.resolvers.composite import CompositeResolver
        >>> resolver = CompositeResolver(debug=True)
        >>> normalized = await normalize_results(search_results, resolver, debug=True)
    """
    if not results or not resolver:
        return results

    if debug:
        logger.debug(
            f"Normalizing {len(results)} results using {resolver.__class__.__name__}",
        )

    async def _normalize_single(result: SearchResultItem) -> SearchResultItem:
        """Normalize a single search result item."""
        try:
            resolver_input = resolver.input_schema(**result.model_dump())
            normalized = await resolver.arun(resolver_input)

            # Convert ResolverOutputSchema back to SearchResultItem
            normalized_dict = normalized.model_dump(
                include=set(SearchResultItem.model_fields.keys()),
            )
            search_result = SearchResultItem(**normalized_dict)

            # Store resolver chain info in extra for downstream processing
            search_result.extra["resolvers"] = getattr(normalized, "resolvers", [])

            return search_result
        except Exception as e:
            if debug:
                logger.warning(f"Normalization failed for {result.title}: {e}")
            return result  # Return original on failure (graceful degradation)

    # Normalize all results in parallel
    normalized_results = await asyncio.gather(
        *[_normalize_single(result) for result in results],
        return_exceptions=False,
    )

    return list(normalized_results)


def get_doi(result: "SearchResultItem") -> str | None:
    """
    Get DOI from result, checking both result.doi field and result.url.

    Uses DOIResolver static methods for extraction and normalization.

    Args:
        result: SearchResultItem to extract DOI from.

    Returns:
        Normalized DOI or None if not found.

    Examples:
        >>> result1 = SearchResultItem(doi="10.1234/example", ...)
        >>> get_doi(result1)
        '10.1234/example'
        >>> result2 = SearchResultItem(url="https://doi.org/10.5678/test", ...)
        >>> get_doi(result2)
        '10.5678/test'
    """
    return DOIResolver.normalize_doi(result.doi) or DOIResolver.normalize_doi(
        DOIResolver.extract_doi_from_url(result.url),
    )


def normalize_title(title: str | None) -> str | None:
    """
    Normalize title: lowercase, remove punctuation, collapse spaces.

    Examples:
        >>> normalize_title("Deep Learning: A Survey")
        'deep learning a survey'
    """
    if not title:
        return None

    normalized = title.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)  # Remove punctuation
    normalized = re.sub(r"\s+", " ", normalized)  # Collapse spaces
    normalized = normalized.strip()

    return normalized if normalized else None


def normalize_url(url: str | AnyUrl | None) -> str | None:
    """
    Normalize URL: remove protocol, www, trailing slash.

    Examples:
        >>> normalize_url("https://www.example.com/paper/")
        'example.com/paper'
    """
    if not url:
        return None

    url_str = str(url)
    parsed = urlparse(url_str)

    netloc = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.rstrip("/")

    normalized = f"{netloc}{path}"

    return normalized if normalized else None


# =============================================================================
# Deduplication
# =============================================================================


def deduplicate_results(
    *results: list["SearchResultItem"],
    keys: list[str] | str | None = "url",
    debug: bool = False,
) -> list[list["SearchResultItem"]]:
    """
    Deduplicate search results across multiple lists while preserving structure.

    Strategy (keeps first occurrence):
    1. Tag each result with source_id to track which list it came from
    2. Flatten all results
    3. Apply cascaded deduplication using specified keys in priority order
    4. Group results back by source_id to restore list[list[...]] structure

    Args:
        *results: Variable number of result lists from different tools/sources.
        keys: List of attribute names for cascaded deduplication in priority order.
              Default: ["doi", "title", "url"]. Checks each key in order until match found.
        debug: Enable debug logging.

    Returns:
        List of result lists (same structure as input) with duplicates removed.
        Original order within each list is preserved.
        Each result will have extra["dedup_source_id"] added.

    Examples:
        >>> # Default: DOI → Title → URL
        >>> deduped = deduplicate_results(tool1_results, tool2_results, debug=True)

        >>> # Custom: Title → URL only
        >>> deduped = deduplicate_results(tool1_results, tool2_results, keys=["title", "url"])

        >>> # DOI only
        >>> deduped = deduplicate_results(tool1_results, tool2_results, keys=["doi"])
    """
    if not results:
        return []

    # Step 1: Tag each result with source_id and flatten
    all_results = []
    for source_id, result_list in enumerate(results):
        for result in result_list:
            # Add source tracking to extra dict
            if result.extra is None:
                result.extra = {}
            result.extra["dedup_source_id"] = source_id
            all_results.append(result)

    if not all_results:
        return [[] for _ in results]

    # Use default keys if not provided
    if isinstance(keys, str):
        keys = [keys]
    dedup_keys = keys or ["doi", "title", "url"]

    # Map keys to their normalization functions
    key_normalizers = {
        "doi": lambda r: get_doi(r),
        "title": lambda r: normalize_title(r.title) if r.title else None,
        "url": lambda r: normalize_url(r.url) if r.url else None,
    }

    # Step 2: Deduplicate using cascaded matching based on keys
    seen_values: dict[str, set[str]] = {key: set() for key in dedup_keys}
    deduplicated = []

    for result in all_results:
        is_duplicate = False
        source_id = result.extra.get("dedup_source_id", "unknown")

        # Check each key in priority order (cascaded matching)
        for key in dedup_keys:
            if is_duplicate:
                break  # Already found duplicate, stop checking

            # Get normalizer for this key
            normalizer = key_normalizers.get(key)
            if not normalizer:
                if debug:
                    logger.warning(f"Unknown deduplication key: {key}, skipping")
                continue

            # Get normalized value for this key
            normalized_value = normalizer(result)
            if not normalized_value:
                continue  # Skip if value is None/empty

            # Check if we've seen this value before
            if normalized_value in seen_values[key]:
                is_duplicate = True
                if debug:
                    preview = normalized_value[:50] if len(normalized_value) > 50 else normalized_value
                    logger.debug(
                        f"Duplicate by {key}: {preview}... | source={source_id}",
                    )
            else:
                seen_values[key].add(normalized_value)

        # Keep first occurrence
        if not is_duplicate:
            deduplicated.append(result)

    # Step 3: Group results back by source_id
    results_by_source: dict[int, list["SearchResultItem"]] = defaultdict(list)

    for result in deduplicated:
        source_id = result.extra.get("dedup_source_id")
        if source_id is not None:
            results_by_source[source_id].append(result)

    # Step 4: Rebuild list[list[...]] in original order
    deduplicated_lists = []
    for source_id in range(len(results)):
        deduplicated_lists.append(results_by_source.get(source_id, []))

    if debug:
        original_total = sum(len(lst) for lst in results)
        deduped_total = sum(len(lst) for lst in deduplicated_lists)
        logger.info(
            f"Deduplication: {original_total} → {deduped_total} "
            f"(removed {original_total - deduped_total} duplicates across {len(results)} sources)",
        )

    return deduplicated_lists


__all__ = [
    "normalize_results",
    "deduplicate_results",
    "get_doi",
    "normalize_title",
    "normalize_url",
]
