"""
Utility to fetch and cache CMR keywords, instruments, and science keywords.

This module provides functions to download the latest metadata from CMR
and save it to JSON files for use by the data search agent.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

CMR_BASE = "https://cmr.earthdata.nasa.gov/search/keywords/{slug}"

KEYWORD_SLUGS = {
    "instruments": "instruments",
    "platforms": "platforms",
    "science_keywords": "science_keywords",
}

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "cmr_enums"


def create_cmr_session() -> requests.Session:
    """Create a requests session with retry logic for CMR API."""
    session = requests.Session()
    session.headers.update({"User-Agent": "akd-cmr-keywords-fetcher/1.0"})

    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    return session


def fetch_cmr_keywords(
    slug: str,
    output_dir: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> Path:
    """
    Fetch CMR keywords for a specific slug and save to JSON file.

    Args:
        slug: The CMR keyword slug (e.g., 'instruments', 'science_keywords')
        output_dir: Directory to save the JSON file (defaults to utils/data/)
        session: Optional requests session (will create one if not provided)

    Returns:
        Path to the saved JSON file

    Raises:
        requests.HTTPError: If the CMR API request fails
        ValueError: If the slug is not supported
    """
    if slug not in KEYWORD_SLUGS.values():
        raise ValueError(
            f"Unsupported slug '{slug}'. Must be one of: {list(KEYWORD_SLUGS.values())}",
        )

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if session is None:
        session = create_cmr_session()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data from CMR
    url = CMR_BASE.format(slug=slug)
    response = session.get(url, params={"pretty": "false"}, timeout=45)
    response.raise_for_status()

    # Prepare output data with metadata
    output_data = {
        "source": response.url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slug": slug,
        "data": response.json(),
    }

    # Save to file
    output_file = output_dir / f"cmr_{slug}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Fetched {slug}: {output_file}")
    return output_file


def fetch_all_cmr_keywords(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Fetch all supported CMR keyword types and save to JSON files.

    Args:
        output_dir: Directory to save the JSON files (defaults to utils/data/)

    Returns:
        Dict mapping keyword type to saved file path

    Raises:
        requests.HTTPError: If any CMR API request fails
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    session = create_cmr_session()
    results = {}

    print(f"Fetching CMR keywords to: {output_dir}")

    for keyword_type, slug in KEYWORD_SLUGS.items():
        try:
            file_path = fetch_cmr_keywords(slug, output_dir, session)
            results[keyword_type] = file_path
        except Exception as e:
            print(f"✗ Failed to fetch {keyword_type}: {e}")
            raise

    print(f"✓ Successfully fetched {len(results)} keyword types")
    return results


def load_cmr_keywords(
    keyword_type: str,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load previously fetched CMR keywords from JSON file.

    Args:
        keyword_type: Type of keywords to load ('instruments', 'platforms', 'science_keywords')
        data_dir: Directory containing the JSON files (defaults to utils/data/)

    Returns:
        Dictionary containing the loaded keyword data

    Raises:
        FileNotFoundError: If the keyword file doesn't exist
        ValueError: If the keyword type is not supported
    """
    if keyword_type not in KEYWORD_SLUGS:
        raise ValueError(
            f"Unsupported keyword type '{keyword_type}'. Must be one of: {list(KEYWORD_SLUGS.keys())}",
        )

    if data_dir is None:
        data_dir = DEFAULT_OUTPUT_DIR

    slug = KEYWORD_SLUGS[keyword_type]
    file_path = data_dir / f"cmr_{slug}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    """Command line interface for fetching CMR keywords."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch CMR keywords, instruments, and science keywords",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keyword-type",
        choices=list(KEYWORD_SLUGS.keys()),
        help="Fetch only specific keyword type (default: fetch all)",
    )

    args = parser.parse_args()

    try:
        if args.keyword_type:
            slug = KEYWORD_SLUGS[args.keyword_type]
            fetch_cmr_keywords(slug, args.output_dir)
        else:
            fetch_all_cmr_keywords(args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
