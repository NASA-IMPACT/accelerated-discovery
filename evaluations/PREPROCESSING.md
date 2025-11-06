# Truth Set and Baseline Comparison Data

This directory contains the ground truth data and baseline comparison results for evaluating data search query performance.

## Files

### Ground Truth (SME Truth Set)
- `truth_set_20251027.csv` - Original SME truth set data in CSV format
- `truth_set_20251027.json` - Clean JSON representation of the SME truth set
- `convert_truth_set.py` - Python script to convert SME CSV to JSON

### Baseline Comparison (ChatGPT Results)
- `DataAgentQueries - ChatGPT.csv` - ChatGPT results in CSV format (from Excel with merged cells)
- `chatgpt_results.json` - Clean JSON representation of ChatGPT results
- `convert_chatgpt_results.py` - Python script to convert ChatGPT CSV to JSON

### Documentation
- `README.md` - Main evaluation system documentation
- `PREPROCESSING.md` - This file

## SME Truth Set JSON Structure

The SME truth set JSON file follows a hierarchical tree structure representing ground truth:

```
Query → Topics → Decomps → CMR Concept IDs
```

### SME Truth Set Schema

```json
{
  "queries": [
    {
      "query_number": 1,
      "sme": "Nidhi",
      "query_text": "How can we assess agricultural drought...",
      "topics": [
        {
          "topic": "Agricultural Drought",
          "decomps": [
            {
              "decomp": "Soil Moisture (Surface)",
              "minimum": true,
              "cmr_concept_ids": ["C2776463935-NSIDC_ECS", ...]
            }
          ]
        }
      ]
    }
  ]
}
```

### Field Descriptions

- **query_number**: Unique identifier for each query (integer)
- **sme**: Subject Matter Expert who created the query
- **query_text**: The full text of the search query
- **topic**: Thematic grouping within a query (e.g., "Agricultural Drought", "Crop Productivity")
- **decomp**: Decomposition/component of a topic (e.g., "Soil Moisture (Surface)", "Precipitation")
- **minimum**: Boolean indicating if this decomp is a minimum requirement
- **cmr_concept_ids**: Array of CMR concept IDs that satisfy this decomp

## Data Considerations

### What's Included

The conversion process includes:
- Query metadata (query number, SME, query text)
- Topic groupings
- Decomposition components
- Minimum requirement flags
- **CMR Concept IDs from the "CMR Concept Id" column only**

### What's Excluded

The following CSV columns are **intentionally excluded** from the JSON output:
- **Possible Matches** - Alternative/additional matches not considered ground truth
- **Notes** - Free-text annotations and comments
- **Perfect Match** - Match quality indicators
- **Expected Dataset** - Dataset names (only IDs are kept)
- **Landing Page link** - URLs to CMR pages
- **Decomps** column label (the content is used but not the "Expected Dataset" description)

### Multi-line Handling

The CSV contains multi-line entries where:
- Rows with a query number start a new entry
- Rows without a query number continue the previous entry
- The script properly merges these continuation rows

### CMR ID Extraction

CMR Concept IDs are extracted using regex pattern: `C\d+-[A-Z_]+`

This captures IDs like:
- `C2776463935-NSIDC_ECS`
- `C3383993430-NSIDC_ECS`
- `C2723754859-GES_DISC`

IDs are automatically deduplicated while preserving order.

## Regenerating the SME Truth Set JSON

To regenerate the SME truth set JSON file from the CSV:

```bash
cd evaluations
uv run python convert_truth_set.py
```

The script will:
1. Read `truth_set_20251027.csv`
2. Parse and handle multi-line entries
3. Build the hierarchical tree structure
4. Extract and deduplicate CMR concept IDs
5. Write to `truth_set_20251027.json`

## Regenerating the ChatGPT Results JSON

To regenerate the ChatGPT results JSON file from the CSV:

```bash
cd evaluations
uv run python convert_chatgpt_results.py
```

The script will:
1. Read `DataAgentQueries - ChatGPT.csv` (from Excel with merged cells)
2. Forward-fill merged columns: SME, Prompt Type, MCP, Query Text
3. Keep decompositions and concept IDs unique per row
4. Fix known typos (e.g., "emyl" → "Emily")
5. Extract and deduplicate CMR concept IDs
6. Write to `chatgpt_results.json`

**Important**: ChatGPT results are NOT ground truth - they represent ChatGPT's output for baseline comparison.

## Statistics

**SME Truth Set** (ground truth):
- **20 queries**
- **39 topics**
- **80 decomps** (74 minimum, 6 non-minimum)

**ChatGPT Results** (baseline comparison):
- **20 query variants** (5 base queries × 4 configurations)
- **71 decomps** returned by ChatGPT
- **280 CMR concept IDs** suggested by ChatGPT

## Running Evaluations

Once the JSON truth set is generated, you can run all queries through the data search agent:

```bash
# Run all queries with gpt-5-mini
uv run python evaluations/run_evaluation.py --model gpt-5-mini

# Resume a previous run (skip completed queries)
uv run python evaluations/run_evaluation.py --model gpt-5-mini --resume

# Test with a single query
uv run python evaluations/run_evaluation.py --query-number 1 --model gpt-5-mini
```

The evaluation runner:
1. Reads queries from `truth_set_20251027.json`
2. Runs each query through `akd/agents/data_search/cli.py`
3. Collects output files from `captured_data/` directory
4. Saves progress to `evaluation_runs.json` (with resume capability)
5. Generates a mapping between query numbers and search result files

### Evaluation Output

The evaluation creates `evaluations/evaluation_runs.json`:

```json
{
  "run_timestamp": "2025-10-27T...",
  "model": "gpt-5-mini",
  "queries_total": 20,
  "queries_completed": 20,
  "results": [
    {
      "query_number": 1,
      "query_text": "How can we assess...",
      "output_file": "captured_data/c6291282_...",
      "search_id": "c6291282_...",
      "duration_seconds": 582.47,
      "topics_processed": 2,
      "total_filtered_results": 67,
      "status": "completed"
    }
  ]
}
```

## Notes

- The script uses `uv run` as per project requirements
- All CMR concept IDs come from the "CMR Concept Id" column only
- The `minimum` field is parsed from "Yes"/"No" values in the CSV
- Topics and decomps are grouped naturally based on the CSV structure
