"""
Utilities for search results: normalization, resolution, and deduplication.

- SearchResultItemNormalizer: Resolves and normalizes SearchResultItems using resolvers
- deduplicate_results: Simple cascaded deduplication (DOI → Title → URL)
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse

from loguru import logger
from pydantic import AnyUrl

from akd.structures import SearchResultItem
from akd.tools.resolvers.specialized import DOIResolver

# =============================================================================
# Normalization Utilities
# =============================================================================


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
    debug: bool = False,
) -> list[list["SearchResultItem"]]:
    """
    Deduplicate search results across multiple lists while preserving structure.

    Strategy (keeps first occurrence):
    1. Tag each result with source_id to track which list it came from
    2. Flatten all results
    3. Apply cascaded deduplication: DOI → Title → URL
    4. Group results back by source_id to restore list[list[...]] structure

    Args:
        *results: Variable number of result lists from different tools/sources.
        debug: Enable debug logging.

    Returns:
        List of result lists (same structure as input) with duplicates removed.
        Original order within each list is preserved.
        Each result will have extra["dedup_source_id"] added.

    Examples:
        >>> tool1_results = [result_a, result_b]
        >>> tool2_results = [result_b_duplicate, result_c]
        >>> deduped_lists = deduplicate_results(tool1_results, tool2_results, debug=True)
        >>> len(deduped_lists)  # Still 2 lists
        2
        >>> len(deduped_lists[0])  # tool1 still has 2 results
        2
        >>> len(deduped_lists[1])  # tool2 now has 1 result (b removed as duplicate)
        1
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

    # Step 2: Deduplicate using cascaded matching (DOI → Title → URL)
    seen_dois = set()
    seen_titles = set()
    seen_urls = set()
    deduplicated = []

    for result in all_results:
        is_duplicate = False
        source_id = result.extra.get("dedup_source_id", "unknown")

        # Check 1: DOI match (from both result.doi and result.url)
        doi = get_doi(result)
        if doi:
            if doi in seen_dois:
                is_duplicate = True
                if debug:
                    logger.debug(
                        f"Duplicate by DOI: {doi} | "
                        f"source={source_id}, title={result.title[:50] if result.title else 'N/A'}...",
                    )
            else:
                seen_dois.add(doi)

        # Check 2: Title match (only if not already matched by DOI)
        if not is_duplicate and result.title:
            norm_title = normalize_title(result.title)
            if norm_title:
                if norm_title in seen_titles:
                    is_duplicate = True
                    if debug:
                        logger.debug(f"Duplicate by title: {norm_title[:50]}... | source={source_id}")
                else:
                    seen_titles.add(norm_title)

        # Check 3: URL match (fallback)
        if not is_duplicate and result.url:
            norm_url = normalize_url(result.url)
            if norm_url:
                if norm_url in seen_urls:
                    is_duplicate = True
                    if debug:
                        logger.debug(f"Duplicate by URL: {norm_url} | source={source_id}")
                else:
                    seen_urls.add(norm_url)

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
    "deduplicate_results",
    "get_doi",
    "normalize_title",
    "normalize_url",
]
