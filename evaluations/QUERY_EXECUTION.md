# Query Execution

## Overview

The evaluation runner executes all queries from the truth set through the data search agent and tracks results for quality evaluation.


## Quick Start

```bash
# Run all 20 queries
uv run python evaluations/run_evaluation.py --model gpt-5-mini

# Resume interrupted run
uv run python evaluations/run_evaluation.py --model gpt-5-mini --resume

# Test single query
uv run python evaluations/run_evaluation.py --query-number 1 --model gpt-5-mini
```

## How It Works

### Input
- **Truth set**: `evaluations/truth_set_20251027.json`
  - 20 queries with expected CMR concept IDs

### Execution Flow

1. **Load queries** from truth set JSON
2. **For each query**:
   - Run via CLI: `uv run akd/agents/data_search/cli.py "<query>" --model gpt-5-mini`
   - CLI auto-saves results to `captured_data/{search_id}_{query_slug}.json`
   - Extract metadata from output file
   - Save progress to `evaluation_runs.json`
3. **Continue** even if individual queries fail

### Output

**`evaluations/evaluation_runs.json`** - Maps queries to results:

```json
{
  "run_timestamp": "2025-10-27T...",
  "model": "gpt-5-mini",
  "queries_total": 20,
  "queries_completed": 20,
  "results": [
    {
      "query_number": 1,
      "query_text": "How can we assess agricultural drought...",
      "output_file": "captured_data/df864220_how_can_we_assess_agricultural_drought_severity_an.json",
      "search_id": "df864220_how_can_we_assess_agricultural_drought_severity_an",
      "duration_seconds": 187.3,
      "topics_processed": 2,
      "total_cmr_results": 584,
      "total_filtered_results": 67,
      "status": "completed"
    }
  ]
}
```

## Features

- **Resume capability**: Automatically skips completed queries with `--resume`
- **Error resilience**: Continues execution if individual queries fail
- **Progress tracking**: Saves state after each query
- **Flexible testing**: Run specific queries with `--query-number`

## Output File Structure

Each query generates a detailed output file in `captured_data/`:

```
{search_id}_{query_slug}.json
```

Contains:
- **agent_output**: Topics, decompositions, search approaches, and CMR collections
- **search_metadata**: Timing, configuration, query details
- **summary**: High-level results with collection counts per topic/decomp

## Next Steps

After execution, use these outputs for:
- Quality scoring (comparing returned collections vs. truth set)
- Performance analysis (timing, collection counts)
- Failure analysis (which queries failed, why)
