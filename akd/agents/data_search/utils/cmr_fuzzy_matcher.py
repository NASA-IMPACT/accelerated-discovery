"""
Fuzzy matching utility for CMR keywords, instruments, and platforms.

This module provides fuzzy string matching capabilities against CMR metadata
to help find relevant instruments, platforms, and science keywords based on
partial or approximate name matches.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from .cmr_keywords_fetcher import load_cmr_keywords


@dataclass
class FuzzyMatch:
    """Result of a fuzzy match operation."""

    matched_value: str
    similarity_score: float
    original_query: str


def _extract_instrument_names(data: Dict[str, Any]) -> List[str]:
    """
    Extract all instrument names from CMR instruments data.

    Traverses the nested hierarchy: category -> class -> type -> subtype -> short_name
    and extracts both short_name and long_name values.

    Args:
        data: The 'data' section from CMR instruments JSON

    Returns:
        List of unique instrument name strings
    """
    names = set()

    def traverse_level(obj, level_path):
        if isinstance(obj, dict):
            # Handle short_name and long_name extraction
            if "value" in obj:
                names.add(obj["value"])

            # Traverse all subfields
            for key, value in obj.items():
                if key in ["short_name", "long_name"] and isinstance(value, list):
                    for item in value:
                        traverse_level(item, level_path + [key])
                elif isinstance(value, (list, dict)):
                    traverse_level(value, level_path + [key])

        elif isinstance(obj, list):
            for item in obj:
                traverse_level(item, level_path)

    if "category" in data:
        traverse_level(data["category"], ["category"])

    # Filter out generic/placeholder values
    filtered_names = {
        name for name in names if name and name != "NOT APPLICABLE" and len(name) > 1
    }

    return list(filtered_names)


def _extract_platform_names(data: Dict[str, Any]) -> List[str]:
    """
    Extract all platform names from CMR platforms data.

    Traverses the nested hierarchy: basis -> category -> sub_category -> short_name
    and extracts both short_name and long_name values.

    Args:
        data: The 'data' section from CMR platforms JSON

    Returns:
        List of unique platform name strings
    """
    names = set()

    def traverse_level(obj, level_path):
        if isinstance(obj, dict):
            # Handle short_name and long_name extraction
            if "value" in obj:
                names.add(obj["value"])

            # Traverse all subfields
            for key, value in obj.items():
                if key in ["short_name", "long_name"] and isinstance(value, list):
                    for item in value:
                        traverse_level(item, level_path + [key])
                elif isinstance(value, (list, dict)):
                    traverse_level(value, level_path + [key])

        elif isinstance(obj, list):
            for item in obj:
                traverse_level(item, level_path)

    if "basis" in data:
        traverse_level(data["basis"], ["basis"])

    # Filter out generic/placeholder values
    filtered_names = {
        name for name in names if name and name != "NOT APPLICABLE" and len(name) > 1
    }

    return list(filtered_names)


def _extract_science_keywords(data: Dict[str, Any]) -> List[str]:
    """
    Extract all science keyword terms from CMR science keywords data.

    Traverses all hierarchy levels: category -> topic -> term -> variable_level_1, etc.
    and extracts value fields at each level.

    Args:
        data: The 'data' section from CMR science keywords JSON

    Returns:
        List of unique science keyword strings
    """
    keywords = set()

    def traverse_level(obj, level_path):
        if isinstance(obj, dict):
            # Extract value field
            if "value" in obj:
                keywords.add(obj["value"])

            # Traverse all subfields
            for key, value in obj.items():
                if isinstance(value, (list, dict)):
                    traverse_level(value, level_path + [key])

        elif isinstance(obj, list):
            for item in obj:
                traverse_level(item, level_path)

    if "category" in data:
        traverse_level(data["category"], ["category"])

    # Filter out generic/placeholder values and very short terms
    filtered_keywords = {
        keyword for keyword in keywords if keyword and len(keyword) > 2
    }

    return list(filtered_keywords)


def find_instrument_matches(
    query: str,
    threshold: float = 0.7,
    data_dir: Optional[Path] = None,
) -> List[FuzzyMatch]:
    """
    Find fuzzy matches for instrument names.

    Args:
        query: The instrument name to search for
        threshold: Minimum similarity score (0.0 to 1.0)
        data_dir: Directory containing CMR data files

    Returns:
        List of FuzzyMatch objects sorted by similarity score (highest first)
    """
    # Load instruments data
    instruments_data = load_cmr_keywords("instruments", data_dir)
    instrument_names = _extract_instrument_names(instruments_data["data"])

    # Find fuzzy matches
    matches = []
    query_lower = query.lower()

    for name in instrument_names:
        # Calculate similarity score
        score = fuzz.ratio(query_lower, name.lower()) / 100.0

        if score >= threshold:
            matches.append(
                FuzzyMatch(
                    matched_value=name,
                    similarity_score=score,
                    original_query=query,
                ),
            )

    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    return matches


def find_platform_matches(
    query: str,
    threshold: float = 0.7,
    data_dir: Optional[Path] = None,
) -> List[FuzzyMatch]:
    """
    Find fuzzy matches for platform names.

    Args:
        query: The platform name to search for
        threshold: Minimum similarity score (0.0 to 1.0)
        data_dir: Directory containing CMR data files

    Returns:
        List of FuzzyMatch objects sorted by similarity score (highest first)
    """
    # Load platforms data
    platforms_data = load_cmr_keywords("platforms", data_dir)
    platform_names = _extract_platform_names(platforms_data["data"])

    # Find fuzzy matches
    matches = []
    query_lower = query.lower()

    for name in platform_names:
        # Calculate similarity score
        score = fuzz.ratio(query_lower, name.lower()) / 100.0

        if score >= threshold:
            matches.append(
                FuzzyMatch(
                    matched_value=name,
                    similarity_score=score,
                    original_query=query,
                ),
            )

    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    return matches


def find_science_keyword_matches(
    query: str,
    threshold: float = 0.7,
    data_dir: Optional[Path] = None,
) -> List[FuzzyMatch]:
    """
    Find fuzzy matches for science keywords.

    Args:
        query: The science keyword to search for
        threshold: Minimum similarity score (0.0 to 1.0)
        data_dir: Directory containing CMR data files

    Returns:
        List of FuzzyMatch objects sorted by similarity score (highest first)
    """
    # Load science keywords data
    science_data = load_cmr_keywords("science_keywords", data_dir)
    science_keywords = _extract_science_keywords(science_data["data"])

    # Find fuzzy matches
    matches = []
    query_lower = query.lower()

    for keyword in science_keywords:
        # Calculate similarity score
        score = fuzz.ratio(query_lower, keyword.lower()) / 100.0

        if score >= threshold:
            matches.append(
                FuzzyMatch(
                    matched_value=keyword,
                    similarity_score=score,
                    original_query=query,
                ),
            )

    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    return matches
