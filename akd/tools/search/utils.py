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
from typing import Callable
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
    title = title or ""
    normalized = title.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)  # Remove punctuation
    normalized = re.sub(r"\s+", " ", normalized)  # Collapse spaces
    normalized = normalized.strip()
    return normalized


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


def sort_results(
    results: list[SearchResultItem],
    sort_by: str = "score",
    debug: bool = False,
) -> list[SearchResultItem]:
    """
    Sort results by the specified key. First checks for the key directly in the dict,
    then checks in the 'extra' field if it exists. Returns unsorted if key not found.
    """

    def __get_sort_key(result):
        """
        Gets the sorting key from the result object, checking the direct
        attribute first, then the 'extra' dictionary.
        """

        # 1. Try to get the attribute directly from the object.
        # We use a default of `None` to distinguish "doesn't exist"
        # from a valid "falsy" value like 0, False, or [].
        if (value := getattr(result, sort_by, None)) is not None:
            return value

        # 2. If not found (or was None), check the 'extra' attribute.
        # Safely get 'extra', defaulting to an empty dict if it's None or missing.
        extra = getattr(result, "extra", None)

        # 3. If 'extra' is a dict, try to .get() the key.
        # .get() safely returns None if the key doesn't exist.
        if isinstance(extra, dict):
            if (value := extra.get(sort_by)) is not None:
                return value

        # 4. If not found in either place, return the default sorting value.
        return float("-inf")

    try:
        # Sort in descending order (highest score first)
        # Change reverse=False if you want ascending order
        return sorted(results, key=__get_sort_key, reverse=True)
    except TypeError:
        # If sorting fails (mixed types), return as is
        if debug:
            logger.warning(f"Sorting by {sort_by} failed due to mixed types.")
        return results


def reciprocal_rank_fusion(
    *results: list["SearchResultItem"],
    k: int = 60,
    keys: list[str] | str | None = None,
    value_normalizers: dict[str, Callable] | None = None,
    normalize: bool = True,
    debug: bool = False,
) -> list["SearchResultItem"]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF) with multi-key matching and value normalization.

    Formula: RRF_score(item) = Σ(1 / (k + rank)) across all lists containing item

    Args:
        results: Variable number of ranked result lists (one per query).
        k: RRF constant (default 60, standard value from literature).
        keys: List of attribute names for deduplication (cascaded OR logic).
              Default: ["doi", "title", "url"] - matches if ANY key matches.
              Can also pass single string for backward compatibility.
        value_normalizers: Optional dict mapping key names to normalizer functions.
              Each function takes a SearchResultItem and returns a normalized string or None.
              Default normalizers: doi (extract from URL), title (lowercase, no punctuation), url (no protocol/www).
              Pass empty dict {} to disable normalization.
              Example: {"url": lambda item: my_custom_normalizer(item.url)}
        normalize: If True, apply min-max normalization to scores [0.1, 1.0] (default True).
                  Raw RRF scores are always preserved in item.extra["rrf_score"].
        debug: Enable debug logging showing matched keys and values.

    Returns:
        Fused list sorted by RRF score (descending). Original item values are preserved.

    Notes:
        - Normalization is applied ONLY for matching/deduplication, not for output values
        - Items with URLs "https://github.com/foo" and "https://www.github.com/foo/" will match
        - Items with titles "Deep Learning: A Survey" and "deep learning a survey" will match
        - The first encountered item's original values are kept in the output

    Examples:
        >>> # Default normalization (recommended)
        >>> results_q1 = [item_a, item_b, item_c]
        >>> results_q2 = [item_b, item_d, item_a]
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2)
        # item_b and item_a appear in both, so they get boosted

        >>> # URL normalization matches different URL formats
        >>> item1 = SearchResultItem(url="https://github.com/foo/bar")
        >>> item2 = SearchResultItem(url="https://www.github.com/foo/bar/")
        >>> fused = reciprocal_rank_fusion([item1], [item2], keys=["url"])
        # Result: 1 item (matched via normalized URL), original URL preserved

        >>> # Custom normalizer
        >>> custom = {"url": lambda item: item.url.lower().strip()}
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2, value_normalizers=custom)

        >>> # Disable normalization
        >>> fused = reciprocal_rank_fusion(results_q1, results_q2, value_normalizers={})
    """
    # Handle default and backward compatibility
    keys = keys or ["doi", "title", "url"]
    if isinstance(keys, str):
        keys = [keys]

    # Return empty if no results
    if not results:
        return []

    # Track: identifier -> cumulative RRF score
    # Track: identifier -> SearchResultItem
    # Track: identifier -> set of all (key, value) identifiers for this item
    identifier_to_score = defaultdict(float)
    identifier_to_item = {}
    identifier_groups = defaultdict(set)

    # Import normalizer functions (avoid circular import at module level)

    # Use provided normalizers or defaults
    value_normalizers = value_normalizers or {
        "doi": lambda item: get_doi(item),
        "title": lambda item: normalize_title(item.title) if item.title else None,
        "url": lambda item: normalize_url(item.url) if item.url else None,
    }

    # Calculate RRF scores with multi-key matching
    for rank_list in results:
        for rank, item in enumerate(rank_list, 1):
            # Collect all non-None identifiers for this item (using normalized values)
            item_identifiers = set()
            for key in keys:
                # Apply normalizer if available, otherwise use original value
                if key in value_normalizers:
                    value = value_normalizers[key](item)
                else:
                    value = getattr(item, key, None)

                if value:
                    item_identifiers.add((key, str(value)))

            if not item_identifiers:
                continue  # Skip items with no valid identifiers

            # Pick canonical identifier (prefer first key in priority order)
            canonical = None
            for key in keys:
                for item_key, item_val in item_identifiers:
                    if item_key == key:
                        canonical = (item_key, item_val)
                        break
                if canonical:
                    break

            if not canonical:
                canonical = next(iter(item_identifiers))

            # Check if any identifier matches existing groups
            matched_canonical = None
            for existing_canonical, group in identifier_groups.items():
                if item_identifiers & group:  # Set intersection - shared identifier?
                    matched_canonical = existing_canonical
                    break

            increment = 1.0 / (rank + k)
            if matched_canonical:
                # Merge with existing item - accumulate RRF score
                identifier_to_score[matched_canonical] += increment

                if debug:
                    # Show which keys matched (BEFORE updating the group!)
                    shared = item_identifiers & identifier_groups[matched_canonical]
                    matched_keys = [k for k, _ in shared]
                    canonical_key, canonical_val = matched_canonical
                    logger.debug(
                        f"[RRF] MERGED: rank={rank} matched via {matched_keys} | "
                        f"primary_key={canonical_key} value='{canonical_val[:40]}...' | +score={increment:.6f}",
                    )

                # Merge all identifiers into the group (AFTER logging!)
                identifier_groups[matched_canonical].update(item_identifiers)
            else:
                # New item - create new group
                identifier_to_score[canonical] += increment
                identifier_to_item[canonical] = item.model_copy()
                identifier_groups[canonical] = item_identifiers
                if debug:
                    available_keys = [k for k, _ in item_identifiers]
                    canonical_key, canonical_val = canonical
                    logger.debug(
                        f"[RRF] NEW: rank={rank} | "
                        f"primary_key={canonical_key} value='{canonical_val[:40]}...' | "
                        f"available_keys={available_keys}",
                    )

    # Build results with RRF scores
    fused_results = []
    for identifier, rrf_score in sorted(identifier_to_score.items(), key=lambda x: x[1], reverse=True):
        item = identifier_to_item[identifier]
        item.score = rrf_score
        item.extra = item.extra or {}
        item.extra["rrf_score"] = rrf_score
        # Track which keys were used for matching (useful for debugging)
        item.extra["rrf_matched_keys"] = list(set(k for k, _ in identifier_groups[identifier]))
        fused_results.append(item)

    if not fused_results:
        return []

    # Normalize scores if requested
    if normalize:
        import numpy as np  # deferred: numpy is in akd[search]/akd[ml]

        scores = np.array([item.score for item in fused_results])
        score_range = scores.max() - scores.min()

        if score_range > 0:
            # Scale to [0.1, 1.0] range to avoid 0.0 scores
            normalized_scores = 0.1 + 0.9 * (scores - scores.min()) / score_range
        else:
            # All scores are the same
            normalized_scores = np.ones_like(scores)

        for item, norm_score in zip(fused_results, normalized_scores):
            item.score = float(norm_score)

    return fused_results


__all__ = [
    "deduplicate_results",
    "normalize_results",
    "sort_results",
    "get_doi",
    "normalize_title",
    "normalize_url",
]
