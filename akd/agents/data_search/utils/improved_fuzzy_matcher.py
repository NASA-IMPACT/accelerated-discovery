"""
Improved Fuzzy Matcher for CMR Instruments and Platforms

A multi-algorithm fuzzy matching system optimized for partial strings,
abbreviations, and typos. Uses weighted combination of string similarity
metrics to provide better matches than character n-gram TF-IDF.

This module provides drop-in functionality for ranking instrument and platform
matches with improved performance on short queries and partial strings.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Union


def _normalize_text(text: str) -> str:
    """
    Normalize text using NFKC Unicode normalization, case folding,
    punctuation replacement, and whitespace collapsing.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Case folding (lowercasing)
    text = text.casefold()

    # Replace ASCII punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.ASCII)

    # Collapse whitespace to single spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_names_from_json(json_path: Union[str, Path]) -> List[str]:
    """
    Extract instrument or platform names from CMR JSON files.

    Args:
        json_path: Path to the CMR JSON file

    Returns:
        List of extracted names (short_name values)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    names = set()

    def extract_short_names(obj):
        """Recursively extract short_name values from nested JSON structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "short_name" and isinstance(value, list):
                    # Extract the "value" field from each short_name item
                    for item in value:
                        if isinstance(item, dict) and "value" in item:
                            name = item["value"]
                            if name and name != "NOT APPLICABLE":
                                names.add(name)
                        elif isinstance(item, str) and item != "NOT APPLICABLE":
                            # Handle cases where short_name is directly a string
                            names.add(item)
                else:
                    extract_short_names(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_short_names(item)

    extract_short_names(data)
    return sorted(list(names))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance (number of single-character edits)
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _jaro_similarity(s1: str, s2: str) -> float:
    """
    Calculate the Jaro similarity between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaro similarity score (0.0 to 1.0)
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_window = max(len1, len2) // 2 - 1
    if match_window < 0:
        match_window = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3.0


def _jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Calculate the Jaro-Winkler similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        p: Prefix scaling factor (default: 0.1)

    Returns:
        Jaro-Winkler similarity score (0.0 to 1.0)
    """
    jaro_sim = _jaro_similarity(s1, s2)

    if jaro_sim < 0.7:
        return jaro_sim

    # Calculate common prefix length (up to 4 chars)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro_sim + (prefix * p * (1 - jaro_sim))


def _longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Calculate the length of the longest common subsequence.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of LCS
    """
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[len1][len2]


def _character_overlap_ratio(s1: str, s2: str) -> float:
    """
    Calculate the character overlap ratio between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Character overlap ratio (0.0 to 1.0)
    """
    if not s1 or not s2:
        return 0.0

    set1 = set(s1.lower())
    set2 = set(s2.lower())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def _partial_ratio(query: str, target: str) -> float:
    """
    Calculate partial ratio similarity (substring matching).

    Args:
        query: Query string
        target: Target string

    Returns:
        Partial ratio score (0.0 to 1.0)
    """
    if not query or not target:
        return 0.0

    shorter = query if len(query) <= len(target) else target
    longer = target if len(query) <= len(target) else query

    if len(shorter) == 0:
        return 0.0

    # Check for exact substring match
    if shorter in longer:
        return 1.0

    # Find best sliding window match
    best_ratio = 0.0
    for i in range(len(longer) - len(shorter) + 1):
        substring = longer[i : i + len(shorter)]
        ratio = _jaro_winkler_similarity(shorter, substring)
        best_ratio = max(best_ratio, ratio)

    return best_ratio


def _substring_bonus(query: str, target: str) -> float:
    """
    Calculate substring bonus score.

    Args:
        query: Query string
        target: Target string

    Returns:
        Substring bonus (0.0 or 1.0)
    """
    if not query or not target:
        return 0.0

    query_lower = query.lower()
    target_lower = target.lower()

    # Direct substring match
    if query_lower in target_lower or target_lower in query_lower:
        return 1.0

    # Check if query is abbreviation (all chars present in order)
    if len(query) < len(target):
        target_idx = 0
        for char in query_lower:
            found = False
            for i in range(target_idx, len(target_lower)):
                if target_lower[i] == char:
                    target_idx = i + 1
                    found = True
                    break
            if not found:
                return 0.0
        return 0.8  # High score for abbreviation match

    return 0.0


def _fuzzy_match_score(query: str, target: str) -> float:
    """
    Calculate composite fuzzy match score using multiple algorithms.

    Args:
        query: Query string
        target: Target string

    Returns:
        Composite similarity score (0.0 to 1.0)
    """
    if not query or not target:
        return 0.0

    if query == target:
        return 1.0

    # Normalize strings for comparison
    query_norm = query.lower()
    target_norm = target.lower()

    # Calculate individual metrics
    jaro_winkler = _jaro_winkler_similarity(query_norm, target_norm)

    # Normalized Levenshtein distance
    lev_distance = _levenshtein_distance(query_norm, target_norm)
    max_len = max(len(query_norm), len(target_norm))
    levenshtein_norm = 1.0 - (lev_distance / max_len) if max_len > 0 else 0.0

    # Partial ratio for substring matching
    partial_ratio = _partial_ratio(query_norm, target_norm)

    # LCS ratio
    lcs_length = _longest_common_subsequence_length(query_norm, target_norm)
    lcs_ratio = lcs_length / max_len if max_len > 0 else 0.0

    # Substring bonus
    substring_bonus = _substring_bonus(query_norm, target_norm)

    # Character overlap
    char_overlap = _character_overlap_ratio(query_norm, target_norm)

    # Weighted combination
    score = (
        0.30 * jaro_winkler
        + 0.20 * levenshtein_norm
        + 0.20 * partial_ratio
        + 0.15 * lcs_ratio
        + 0.10 * substring_bonus
        + 0.05 * char_overlap
    )

    return min(1.0, score)


def _improved_fuzzy_match(
    user_input: str,
    corpus: List[str],
    k_each: int,
    min_score: float,
) -> List[Dict]:
    """
    Improved fuzzy matching using multiple string similarity algorithms.

    Args:
        user_input: Normalized user query
        corpus: List of normalized corpus items
        k_each: Maximum number of results to return
        min_score: Minimum similarity score threshold

    Returns:
        List of matching items with scores
    """
    if not corpus or not user_input:
        return []

    # Calculate scores for all corpus items
    scored_items = []
    for item in corpus:
        score = _fuzzy_match_score(user_input, item)
        if score >= min_score:
            scored_items.append({"name": item, "score": score})

    # Sort by score (descending) and take top k
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:k_each]


def rank_instrument_platform(
    user_input: str,
    instruments: List[str],
    platforms: List[str],
    k_each: int = 3,
    min_score: float = 0.3,
) -> List[Dict]:
    """
    Rank instruments and platforms based on improved fuzzy matching with user input.

    Uses a weighted combination of multiple string similarity algorithms including
    Jaro-Winkler, Levenshtein distance, partial ratio, LCS, and substring matching
    to provide better results for partial strings and abbreviations.

    Args:
        user_input: User query string to match against
        instruments: List of instrument names to search
        platforms: List of platform names to search
        k_each: Maximum number of results per namespace (default: 3)
        min_score: Minimum similarity score threshold (default: 0.3)

    Returns:
        List of dictionaries with keys:
        - 'name': The matched instrument/platform name
        - 'kind': Either 'instrument' or 'platform'
        - 'score': Composite similarity score (0.0 to 1.0)

        Results are sorted by score (highest first) and deduplicated.
    """
    # Input validation
    if not user_input or not user_input.strip():
        return []

    if not instruments and not platforms:
        return []

    # Normalize input
    normalized_input = _normalize_text(user_input)
    if not normalized_input:
        return []

    # Normalize corpus items
    normalized_instruments = [_normalize_text(name) for name in instruments if name]
    normalized_platforms = [_normalize_text(name) for name in platforms if name]

    # Filter out empty normalized names
    valid_instruments = [
        (norm, orig) for norm, orig in zip(normalized_instruments, instruments) if norm
    ]
    valid_platforms = [
        (norm, orig) for norm, orig in zip(normalized_platforms, platforms) if norm
    ]

    results = []

    # Process instruments
    if valid_instruments:
        norm_inst_corpus = [item[0] for item in valid_instruments]
        orig_inst_names = [item[1] for item in valid_instruments]

        inst_matches = _improved_fuzzy_match(
            normalized_input,
            norm_inst_corpus,
            k_each,
            min_score,
        )

        for match in inst_matches:
            # Find original name
            norm_name = match["name"]
            orig_idx = norm_inst_corpus.index(norm_name)
            results.append(
                {
                    "name": orig_inst_names[orig_idx],
                    "kind": "instrument",
                    "score": match["score"],
                },
            )

    # Process platforms
    if valid_platforms:
        norm_plat_corpus = [item[0] for item in valid_platforms]
        orig_plat_names = [item[1] for item in valid_platforms]

        plat_matches = _improved_fuzzy_match(
            normalized_input,
            norm_plat_corpus,
            k_each,
            min_score,
        )

        for match in plat_matches:
            # Find original name
            norm_name = match["name"]
            orig_idx = norm_plat_corpus.index(norm_name)
            results.append(
                {
                    "name": orig_plat_names[orig_idx],
                    "kind": "platform",
                    "score": match["score"],
                },
            )

    # Deduplicate by (name, kind) keeping highest score
    seen = {}
    for result in results:
        key = (result["name"], result["kind"])
        if key not in seen or result["score"] > seen[key]["score"]:
            seen[key] = result

    # Sort by score (highest first) and return
    final_results = list(seen.values())
    final_results.sort(key=lambda x: x["score"], reverse=True)

    return final_results


def load_cmr_data(
    instruments_path: Union[str, Path],
    platforms_path: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """
    Load instrument and platform names from CMR JSON files.

    Args:
        instruments_path: Path to CMR instruments JSON file
        platforms_path: Path to CMR platforms JSON file

    Returns:
        Tuple of (instruments_list, platforms_list)
    """
    instruments = _extract_names_from_json(instruments_path)
    platforms = _extract_names_from_json(platforms_path)
    return instruments, platforms


def create_cmr_matcher(
    instruments_path: Union[str, Path],
    platforms_path: Union[str, Path],
):
    """
    Create a CMR matcher function with pre-loaded data.

    Args:
        instruments_path: Path to CMR instruments JSON file
        platforms_path: Path to CMR platforms JSON file

    Returns:
        Function that takes (user_input, k_each=3, min_score=0.3) and returns matches
    """
    instruments, platforms = load_cmr_data(instruments_path, platforms_path)

    def matcher(user_input: str, k_each: int = 3, min_score: float = 0.3) -> List[Dict]:
        return rank_instrument_platform(
            user_input,
            instruments,
            platforms,
            k_each,
            min_score,
        )

    return matcher


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python improved_fuzzy_matcher.py <search_term>")
        sys.exit(1)

    search_term = sys.argv[1]

    # Load CMR data from JSON files
    script_dir = Path(__file__).parent
    instruments_path = script_dir / "cmr_enums" / "cmr_instruments.json"
    platforms_path = script_dir / "cmr_enums" / "cmr_platforms.json"

    instruments, platforms = load_cmr_data(instruments_path, platforms_path)

    # Perform fuzzy matching (top 3 instruments + top 3 platforms)
    results = rank_instrument_platform(
        search_term,
        instruments,
        platforms,
        k_each=3,
        min_score=0.3,
    )

    import json

    print(json.dumps(results, indent=2))
