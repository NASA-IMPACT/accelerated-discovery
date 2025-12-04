# Data Search Agent

Multi-repository data search agent that discovers NASA Earth science data across CMR, PDS4, and external sources.

## Quick Start

```bash
# Fast development mode (single-path, fast model)
uv run akd/agents/data_search/cli.py "MODIS sea surface temperature 2023" --single-path --model gpt-5-nano

# Full production mode
uv run akd/agents/data_search/cli.py "flood risk Mississippi River" --model gpt-5-mini
```

## How It Works

The agent orchestrates a multi-stage workflow:

1. **Topic Splitting**: Breaks query into 1-6 functional topics
2. **Scientific Decomposition**: Decomposes each topic into 1-6 scientific observables
3. **Repository Routing**: Routes each decomposition to the best data repository
4. **Handler Dispatch**: Repository-specific handlers search for data (CMR, PDS4, etc.)
5. **Auto-Save**: Results saved with full metadata to `captured_data/`

See [DATA_FLOW.md](DATA_FLOW.md) for detailed architecture documentation.

## CLI Reference

### Basic Usage

```bash
uv run akd/agents/data_search/cli.py "your natural language query" [options]
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--single-path` | Execute only [0] branch at each decision point (fast testing) | `false` |
| `--model MODEL` | Model for all components | `gpt-5-mini` |
| `--no-save` | Disable auto-save | Auto-save enabled |
| `--no-metadata` | Disable metadata capture (git, prompts) | Metadata enabled |
| `--debug` | Enable debug logging | `false` |

### Per-Component Model Configuration

Override models for specific pipeline stages:

```bash
uv run akd/agents/data_search/cli.py "your query" \
  --topic-model gpt-5-nano \
  --decomp-model gpt-5-mini \
  --routing-model gpt-5-mini \
  --cmr-known-model gpt-5-mini \
  --cmr-searchable-model gpt-5-nano \
  --cmr-filtering-model gpt-5-mini \
  --cmr-ranking-model gpt-5-mini
```

### Execution Modes

#### Single-Path Mode (Fast)
```bash
uv run akd/agents/data_search/cli.py "atmospheric CO2 2020-2023" --single-path --model gpt-5-nano
```
- Selects only `[0]` branch at each decision point
- Still generates all options for evaluation
- ~2-5 minutes with gpt-5-nano
- Perfect for development and testing

#### Full Workflow Mode (Production)
```bash
uv run akd/agents/data_search/cli.py "atmospheric CO2 2020-2023" --model gpt-5-mini
```
- Processes all branches (up to 6 topics × 6 decompositions)
- Comprehensive results
- Longer runtime (~10-30 minutes)
- Recommended for production searches

## Output Files

### Location

Results are auto-saved to:
```
captured_data/search_{query_slug}_{timestamp}.json
```

Example: `captured_data/search_modis_sea_surface_temperature_2023_1760404324.json`

### File Structure

```json
{
  "agent_output": {
    "topics": [
      {
        "topic": { /* Topic object */ },
        "decomposition_results": [
          {
            "decomposition": { /* Scientific decomposition */ },
            "repository": "CMR",
            "query_approaches": [ /* CMR query strategies */ ],
            "searchable_queries": [ /* Executable CMR queries */ ],
            "data_results": [ /* Collection/granule results */ ],
            "total_results_found": 25
          }
        ]
      }
    ],
    "search_metadata": {
      "search_id": "search_...",
      "original_query": "...",
      "timestamp": "2025-01-15T10:30:00",
      "duration_seconds": 45.2,
      "topics_processed": 1,
      "single_path_mode": true
    },
    "total_results": 25
  },
  "execution_metadata": {
    "timestamp": "2025-01-15T10:30:00",
    "config": {
      /* Complete agent configuration */
      "single_path_mode": true,
      "auto_save": true,
      "topic_splitting_model": "gpt-5-nano",
      /* ... all other config fields */
    },
    "git_info": {
      "commit_hash": "7a6f14ff...",
      "branch": "data-search-agent-streamlined",
      "is_dirty": true
    }
  },
  "prompts": {
    /* All prompt templates used in this execution */
    "components/topic_splitting.md": "...",
    "cmr/known_parameters.md": "...",
    /* ... */
  }
}
```

### Metadata Capture

The auto-save feature captures:

- **Agent Output**: Complete search results
- **Execution Metadata**:
  - Full configuration used
  - Git commit hash, branch, and dirty status
  - Timestamp and duration
- **Prompts**: All prompt templates used (enables reproducibility)

To disable metadata capture (slightly faster):
```bash
uv run akd/agents/data_search/cli.py "query" --no-metadata
```

To disable auto-save entirely:
```bash
uv run akd/agents/data_search/cli.py "query" --no-save
```

## Examples

### Development Workflow

```bash
# Fast iteration with single-path mode
uv run akd/agents/data_search/cli.py "MODIS ocean color" --single-path --model gpt-5-nano --debug

# Test specific query with production config
uv run akd/agents/data_search/cli.py "MODIS ocean color" --model gpt-5-mini

# Fine-tune per-component models
uv run akd/agents/data_search/cli.py "MODIS ocean color" \
  --topic-model gpt-5-nano \
  --cmr-known-model gpt-5-mini \
  --cmr-ranking-model gpt-5-mini
```

### Production Searches

```bash
# Comprehensive atmospheric CO2 search
uv run akd/agents/data_search/cli.py "atmospheric CO2 levels from 2020 to 2023" --model gpt-5-mini

# Flood risk analysis
uv run akd/agents/data_search/cli.py "help me gather data to study the flood risk of the Mississippi River" --model gpt-5-mini

# Sea surface temperature
uv run akd/agents/data_search/cli.py "MODIS sea surface temperature data from 2023" --model gpt-5-mini
```

## Loading Saved Results

Use `demo_loader.py` to analyze captured data:

```bash
# List available data
uv run examples/demo_loader.py captured_data/search_*.json --list

# Analyze specific components
uv run examples/demo_loader.py captured_data/search_*.json --component known_parameters --topic 0 --decomp 0

# View timing analysis
uv run examples/demo_loader.py captured_data/search_*.json --timing
```

## Configuration

The agent has two levels of configuration:

**Workflow Limits** (`constants.py`):
- Controls parallelism: topics, decompositions, approaches, search variations
- Single source of truth for performance tuning
- See [Performance Tuning](#performance-tuning) below

**Runtime Behavior** (`config.py` files):
- MCP endpoint URLs
- Search limits (page sizes, result counts, timeouts)
- Model selections (gpt-5-mini, gpt-5-nano)
- Feature flags (e.g., `include_keyword_only_approach`)

See [DATA_FLOW.md](DATA_FLOW.md) for complete configuration reference.

## Performance Tuning

All workflow limits are centralized in `akd/agents/data_search/constants.py` for easy performance tuning:

```python
# Universal limits (all handlers)
MAX_TOPICS = 3                      # Research question → functional topics
MIN_TOPICS = 1
MAX_DECOMPOSITIONS_PER_TOPIC = 3    # Topic → observable phenomena
MIN_DECOMPOSITIONS_PER_TOPIC = 1

# CMR handler limits
CMR_MAX_LLM_APPROACHES = 4          # LLM-generated approaches (before keyword-only)
CMR_MIN_LLM_APPROACHES = 1
CMR_MAX_TOTAL_APPROACHES_WITH_KEYWORD = 5  # After keyword-only injection (4 + 1)
CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH = 3 # Keyword variations per approach
CMR_MIN_SEARCH_VARIATIONS_PER_APPROACH = 0
```

**How it works**:
1. Edit values in `constants.py`
2. Changes automatically propagate to:
   - LLM prompts (via Instructor schema constraints)
   - Pydantic validation (min_items/max_items)
   - Calculated limits (e.g., max searchable queries = 5 × 3 = 15)

**Tuning recommendations**:
- **Faster execution**: Reduce MAX_TOPICS and MAX_DECOMPOSITIONS_PER_TOPIC
- **Broader coverage**: Increase CMR_MAX_LLM_APPROACHES and CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH
- **API cost optimization**: Lower all max values, use `single_path_mode=True`

See "Workflow Limits & Performance Tuning" section in [DATA_FLOW.md](DATA_FLOW.md) for complete details.

## Troubleshooting

**Slow API calls**: Use `gpt-5-nano` for faster responses:
```bash
uv run akd/agents/data_search/cli.py "query" --model gpt-5-nano --single-path
```

**No results found**: Check debug output to see routing decisions:
```bash
uv run akd/agents/data_search/cli.py "query" --debug
```

**Import errors**: Ensure you're using `uv run` not raw `python`:
```bash
# ✅ Correct
uv run akd/agents/data_search/cli.py "query"

# ❌ Wrong
python akd/agents/data_search/cli.py "query"
```

## Architecture Documentation

For detailed technical documentation, see:
- [DATA_FLOW.md](DATA_FLOW.md) - Complete workflow architecture
- [handlers/](handlers/) - Repository-specific handler implementations
- [components/](components/) - Universal workflow components
