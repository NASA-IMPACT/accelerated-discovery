# Data Search Utilities

This directory contains utilities for the data search agent operations.

## CMR Keywords Fetcher

The `cmr_keywords_fetcher.py` utility downloads the latest metadata from NASA's Common Metadata Repository (CMR) and saves it to JSON files for use by the data search agent.

### What it fetches

- **Instruments**: Scientific instruments used in Earth observation
- **Platforms**: Satellite and aircraft platforms
- **Science Keywords**: Hierarchical scientific keyword taxonomy

### Usage

#### Fetch all keyword types (recommended)

```bash
# From project root
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py
```

#### Fetch specific keyword type

```bash
# Fetch only instruments
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py --keyword-type instruments

# Fetch only platforms
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py --keyword-type platforms

# Fetch only science keywords
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py --keyword-type science_keywords
```

#### Custom output directory

```bash
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py --output-dir /path/to/custom/dir
```

### Output Files

Files are saved to `akd/agents/data_search/utils/cmr_enums/` by default:

- `cmr_instruments.json` - Earth observation instruments
- `cmr_platforms.json` - Satellite and aircraft platforms
- `cmr_science_keywords.json` - Scientific keyword hierarchy

### File Structure

Each JSON file contains:

```json
{
  "source": "https://cmr.earthdata.nasa.gov/search/keywords/instruments?pretty=false",
  "fetched_at": "2025-09-19T12:55:16Z",
  "slug": "instruments",
  "data": {
    // CMR keyword data structure
  }
}
```

### Programmatic Usage

```python
from akd.agents.data_search.utils import fetch_cmr_keywords, fetch_all_cmr_keywords

# Fetch all keyword types
results = fetch_all_cmr_keywords()

# Fetch specific type
instruments_file = fetch_cmr_keywords("instruments")

# Load previously fetched data
from akd.agents.data_search.utils.cmr_keywords_fetcher import load_cmr_keywords
instruments_data = load_cmr_keywords("instruments")
```

### Updating Keywords

Run the fetcher periodically to keep keyword data current:

```bash
# Add to cron or run manually as needed
uv run python akd/agents/data_search/utils/cmr_keywords_fetcher.py
```

The utility includes retry logic and proper error handling for reliable fetching.

## CMR Fuzzy Matcher

The `cmr_fuzzy_matcher.py` utility provides fuzzy string matching against CMR metadata to find relevant instruments, platforms, and science keywords based on partial or approximate name matches.

### What it matches

- **Instruments**: Fuzzy matching against instrument short names and long names
- **Platforms**: Fuzzy matching against platform short names and long names
- **Science Keywords**: Fuzzy matching against hierarchical science keyword terms

### Usage

```python
from akd.agents.data_search.utils import (
    find_instrument_matches,
    find_platform_matches,
    find_science_keyword_matches,
    FuzzyMatch
)

# Find instrument matches
matches = find_instrument_matches("MODIS", threshold=0.8)
for match in matches:
    print(f"{match.matched_value} (score: {match.similarity_score:.3f})")

# Find platform matches
matches = find_platform_matches("landsat", threshold=0.7)
for match in matches:
    print(f"{match.matched_value} (score: {match.similarity_score:.3f})")

# Find science keyword matches
matches = find_science_keyword_matches("precipitation", threshold=0.7)
for match in matches:
    print(f"{match.matched_value} (score: {match.similarity_score:.3f})")
```

### Function Parameters

All matching functions accept:
- `query` (str): The term to search for
- `threshold` (float): Minimum similarity score (0.0 to 1.0, default: 0.7)
- `data_dir` (Optional[Path]): Custom directory for CMR data files

### FuzzyMatch Results

Each match returns a `FuzzyMatch` object containing:
- `matched_value`: The matching CMR term
- `similarity_score`: Similarity score (0.0 to 1.0)
- `original_query`: The original search query

Results are sorted by similarity score in descending order.

### Dependencies

The fuzzy matcher requires `rapidfuzz` for string matching:

```bash
uv add rapidfuzz
```
