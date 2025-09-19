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
