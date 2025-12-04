# Evaluation System for Data Search Agent

This directory contains the complete evaluation system for measuring the accuracy and performance of the NASA Earth science data search agent.

## Overview

The evaluation system compares agent search results against Subject Matter Expert (SME) ground truth to measure how effectively the system finds the correct datasets for scientific research questions.

**Key Components**:
- **Truth Set**: SME Truth Set (20 queries) - Expert-curated ground truth with minimum requirements
- **Baseline Comparison**: ChatGPT Results (20 queries) - ChatGPT's output for comparative evaluation
- **Evaluation Runner**: Executes queries through the agent and captures results
- **Query Re-execution**: Retrieves ALL concept IDs from CMR for complete analysis
- **Accuracy Calculation**: Measures how many expected datasets were found
- **Legacy Scoring**: Evaluates top-5 ranked results (user experience focus)

## Quick Start

### Complete Evaluation Workflow

```bash
# 1. Run all 20 evaluation queries through the agent
cd /Users/cdavis/github/accelerated-discovery
uv run python evaluations/run_evaluation.py --model gpt-5-mini

# 2. Re-execute all CMR queries with full pagination
uv run python evaluations/reexecute_cmr_queries.py evaluations/run_YYYYMMDD_HHMMSS

# 3. Calculate total recall (full CMR results)
uv run python evaluations/calculate_accuracy.py evaluations/run_YYYYMMDD_HHMMSS

# 4. (Optional) Calculate top-5 recall (user experience)
uv run python evaluations/score_evaluation.py --runs evaluations/run_YYYYMMDD_HHMMSS/evaluation_runs.json

# 5. (Optional) Export to Excel for manual review
uv run python evaluations/export_to_excel.py --runs evaluations/run_YYYYMMDD_HHMMSS/evaluation_runs.json
```

## File Structure

```
evaluations/
├── README.md                              # This file - workflow documentation
│
├── Truth Set (Ground Truth):
│   ├── truth_set_20251027.csv             # SME truth set (CSV format)
│   ├── truth_set_20251027.json            # SME truth set (JSON) - 20 queries, with duplicates
│   └── truth_set_20251027_deduplicated.json  # Deduplicated version (default for metrics)
│
├── Baseline Comparison (ChatGPT Results):
│   ├── DataAgentQueries - ChatGPT.csv     # ChatGPT results (CSV from Excel)
│   └── chatgpt_results.json               # ChatGPT results (JSON) - 20 queries, 4 configurations
│
├── Evaluation Runs:
│   └── run_YYYYMMDD_HHMMSS/               # Evaluation run directory
│       ├── evaluation_runs.json           # Maps query_number → output files
│       ├── top_5_recall.json              # Top-5 recall (what users see)
│       ├── total_recall.json              # Total recall (full CMR results)
│       ├── results_export.xlsx            # Excel export for manual review
│       └── reexecuted/                    # Re-executed CMR query results
│           └── {search_id}_concepts.json  # Full concept IDs from CMR
│
├── Documentation:
│   ├── PREPROCESSING.md                   # Truth set structure documentation
│   ├── QUERY_EXECUTION.md                 # How to run evaluations
│   ├── SCORING.md                         # Legacy scoring methodology
│   └── EXCEL_EXPORT.md                    # Excel export format spec
│
└── Scripts:
    ├── run_evaluation.py                  # Execute evaluation queries
    ├── reexecute_cmr_queries.py           # Re-run CMR queries with full pagination
    ├── calculate_accuracy.py              # Calculate total recall (full CMR results)
    ├── score_evaluation.py                # Calculate top-5 recall (user experience)
    ├── export_to_excel.py                 # Export results to Excel
    ├── convert_truth_set.py               # Convert SME CSV → JSON (ground truth)
    ├── convert_chatgpt_results.py         # Convert ChatGPT CSV → JSON (baseline results)
    ├── compare_chatgpt_to_truth.py        # Compare ChatGPT coverage of ground truth
    └── deduplicate_truth_set.py           # Create deduplicated truth set
```

## Detailed Workflow

### 1. Running Evaluation Queries

Execute all 20 queries from the truth set through the data search agent:

```bash
# Run all queries (takes ~30-60 minutes)
uv run python evaluations/run_evaluation.py --model gpt-5-mini

# Resume interrupted run
uv run python evaluations/run_evaluation.py --model gpt-5-mini --resume

# Test single query
uv run python evaluations/run_evaluation.py --query-number 1 --model gpt-5-mini
```

**Output**: Creates `evaluations/run_YYYYMMDD_HHMMSS/` directory with:
- `evaluation_runs.json` - Metadata mapping queries to output files
- Output files saved to `captured_data/run_YYYYMMDD_HHMMSS/{search_id}.json`

**What it does**:
- Loads queries from `truth_set_20251027.json`
- Runs each query through `akd/agents/data_search/cli.py`
- Agent performs: topic splitting → decomposition → parameter extraction → CMR search → ranking
- Agent returns top 25 ranked collections per decomposition
- Results auto-saved with execution metadata and prompts

See [QUERY_EXECUTION.md](QUERY_EXECUTION.md) for details.

### 2. Re-executing CMR Queries

Extract all `mcp_parameters_sent` from agent output and re-execute with full pagination to get ALL concept IDs:

```bash
# Re-execute all queries for a run (takes ~5-10 minutes)
uv run python evaluations/reexecute_cmr_queries.py evaluations/run_20251027_213136
```

**Output**: Creates `evaluations/run_YYYYMMDD_HHMMSS/reexecuted/` with:
- One `{search_id}_concepts.json` file per evaluation query
- Each file contains: `{"data": [concept_ids...], "log": [errors...]}`

**What it does**:
- Reads agent output files from the evaluation run
- Extracts all `mcp_parameters_sent` (typically 40-70 queries per evaluation query)
- Re-executes each CMR query with full pagination (page_size=50, all pages)
- Deduplicates and saves ALL concept IDs returned by CMR
- Logs any query failures (e.g., invalid parameters)

**Why this matters**:
- Agent only returns top 25 collections per decomposition (after ranking/filtering)
- CMR may have returned 100s or 1000s of collections for the underlying queries
- Re-execution gives us the COMPLETE set to measure true recall

### 3. Calculate Accuracy

Measure how many expected concept IDs (from truth set) appear in the full CMR results:

```bash
# Calculate accuracy for a run
uv run python evaluations/calculate_accuracy.py evaluations/run_20251027_213136

# Use custom truth set
uv run python evaluations/calculate_accuracy.py evaluations/run_20251027_213136 \
    --truth-set evaluations/truth_set_custom.json

# Save to custom location
uv run python evaluations/calculate_accuracy.py evaluations/run_20251027_213136 \
    --output my_accuracy.json
```

**Output**: Creates `total_recall.json` with:
- Overall accuracy percentage
- Per-query accuracy breakdown
- Per-decomposition matching details (expected vs found concept IDs)
- Summary statistics

**Scoring Logic**:
1. For each query in truth set:
   - Load full concept IDs from `reexecuted/{search_id}_concepts.json`
   - For each decomposition marked `minimum: true`:
     - Check if expected concept IDs appear in full results
     - Score = (# found / # expected) × 100%
   - Query score = average of all minimum decomposition scores

2. Overall accuracy = average of all query scores

**Example**:
- Query has 5 minimum decompositions
- Decomp 1: Expected 1 ID, found 1 → 100%
- Decomp 2: Expected 1 ID, found 0 → 0%
- Decomp 3: Expected 2 IDs, found 1 → 50%
- Decomp 4: Expected 1 ID, found 1 → 100%
- Decomp 5: Expected 1 ID, found 1 → 100%
- **Query score = (100 + 0 + 50 + 100 + 100) / 5 = 70%**

### 4. Legacy Top-5 Scoring (Optional)

Measure how many expected concept IDs appear in the agent's top-5 ranked results:

```bash
# Score top-5 results (legacy approach)
uv run python evaluations/score_evaluation.py evaluations/run_20251027_213136
```

**Output**: Creates `top_5_recall.json`

**Difference from total recall**:
- **Top-5 recall**: Checks if expected IDs are in agent's top 5 ranked collections per decomposition
- **Total recall**: Checks if expected IDs are in ALL collections returned by CMR
- Top-5 measures "user experience" (what users actually see)
- Total recall measures "system capability" (could the agent have found it)

See [SCORING.md](SCORING.md) for legacy scoring methodology.

### 5. Export to Excel (Optional)

Export results to Excel for manual SME review:

```bash
# Export to Excel
uv run python evaluations/export_to_excel.py \
    --runs evaluations/run_20251027_213136/evaluation_runs.json

# Custom output location
uv run python evaluations/export_to_excel.py \
    --runs evaluations/run_20251027_213136/evaluation_runs.json \
    --output my_results.xlsx
```

**Output**: Creates `results_export.xlsx` with hierarchical merged cells showing:
- Query → Topic → Decomposition → Top 5 collections

See [EXCEL_EXPORT.md](EXCEL_EXPORT.md) for format specification.

## Truth Set and Baseline Comparison

### 1. SME Truth Set (`truth_set_20251027.json`) - Ground Truth

**Purpose**: Expert-curated ground truth for measuring agent correctness against known-good datasets.

The SME truth set contains 20 queries from Subject Matter Experts with expected datasets:

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
              "minimum": true,  ← Required for correct answer
              "cmr_concept_ids": ["C2776463935-NSIDC_ECS"]  ← Expected IDs
            }
          ]
        }
      ]
    }
  ]
}
```

**Key Fields**:
- `minimum: true` - This decomposition is **required** for a correct answer (scored)
- `minimum: false` - This decomposition is optional (not scored)
- `cmr_concept_ids` - The ground truth concept IDs that should be found

**Statistics**:
- 20 queries (10 from Nidhi, 10 from Emily)
- 39 topics total
- 80 decompositions (74 minimum, 6 non-minimum)
- Only the 74 minimum decomps are scored

See [PREPROCESSING.md](PREPROCESSING.md) for complete truth set documentation.

**Regenerating from CSV**:
```bash
uv run python evaluations/convert_truth_set.py
```

### 2. ChatGPT Baseline Results (`chatgpt_results.json`) - NOT Ground Truth

**Purpose**: ChatGPT's output on the same evaluation queries for comparative analysis. This is a **results set**, not ground truth - it represents what ChatGPT returned when asked the same questions.

The ChatGPT results cover **5 base queries** across **4 different configurations** (20 total query variants):

**Configuration Matrix**:
- **2 SMEs**: Emily, Bernard
- **2 Prompt Types**: Basic, Angles
- **2 MCP Settings**: With MCP (Bernard), Without MCP (Emily)

**Structure**:
```json
{
  "metadata": {
    "source": "ChatGPT baseline results",
    "description": "ChatGPT results for comparison. NOT ground truth.",
    "total_queries": 20,
    "total_decompositions": 71,
    "total_concept_ids": 280
  },
  "queries": [
    {
      "query_number": 1,
      "sme": "Emily",
      "prompt_type": "Basic",
      "mcp": false,
      "query_text": "How did sea-ice concentration...",
      "conversation_link": "https://docs.google.com/...",
      "decompositions": [
        {
          "decomp": "sea-ice concentration",
          "cmr_concept_ids": ["C1513368202-NSIDCV0", ...],
          "notes": "Optional notes"  // Only if present
        }
      ]
    }
  ]
}
```

**Key Differences from SME Truth Set**:
- **Not ground truth**: Represents ChatGPT's output, not expert-validated correct answers
- **No `minimum` field**: ChatGPT doesn't distinguish required vs optional decomps
- **Includes `prompt_type`**: Compares Basic vs Angles prompt engineering approaches
- **Includes `mcp` flag**: Compares with/without MCP integration
- **Includes `conversation_link`**: Links to original ChatGPT conversations for reference
- **Different purpose**: Baseline comparison, not correctness evaluation

**Base Queries** (each evaluated 4 times):
1. Sea-ice concentration and thickness in Kara Sea (March 2020-2025)
2. Cyclone Mocha inundation extent and persistence (Myanmar, May 2023)
3. Global lightning flash density changes (2001-2024) vs CAPE patterns
4. Burn scars in Peloponnese, Greece (July-Sept 2021) and EVI recovery
5. Heat index extremes and socio-economic vulnerability (Maricopa County, July-Aug 2023)

**Statistics**:
- 20 query variants (5 base × 4 configurations)
- 71 total decompositions returned by ChatGPT
- 280 total CMR concept IDs suggested by ChatGPT
- Enables comparison: Emily (no MCP) vs Bernard (with MCP)
- Enables comparison: Basic prompts vs Angles prompts

**Use Cases**:
- Compare our agent's results against ChatGPT's suggestions
- Analyze which prompt engineering approaches work better
- Identify datasets ChatGPT finds that our agent might miss (and vice versa)
- Measure overlap between ChatGPT and SME ground truth

**Regenerating from CSV**:
```bash
uv run python evaluations/convert_chatgpt_results.py
```

**Important Notes**:
- **This is NOT ground truth** - ChatGPT's suggestions may be incomplete or incorrect
- The source CSV (`DataAgentQueries - ChatGPT.csv`) originates from an Excel file with **merged cells**
- The converter script handles merged cell forward-filling for: SME, Prompt Type, MCP, Query Text
- Decompositions and concept IDs are **NOT** forward-filled (unique per row)
- Known typo "emyl" is automatically corrected to "Emily"

### 3. Comparing ChatGPT to Ground Truth

To measure how well ChatGPT found the required ground truth datasets:

```bash
uv run python evaluations/compare_chatgpt_to_truth.py
```

**What it measures**: For each of the 4 ChatGPT configurations, what percentage of the `minimum=true` concept IDs from the SME truth set did ChatGPT find?

**Output**: Creates `chatgpt_vs_truth_comparison.json` and prints:
- **Average Coverage**: Per-query average of how many required concepts were found
- **Overall Coverage**: Total concepts found across all queries
- **Per-query breakdown**: Shows which specific concept IDs were missing

**Example Output**:
```
Bernard | Angles | Yes MCP
  Average Coverage: 48.7%
  Overall Coverage: 56.2% (9/16 concepts)
  Queries Evaluated: 5

Emily | Basic | No MCP
  Average Coverage: 37.3%
  Overall Coverage: 31.2% (5/16 concepts)
  Queries Evaluated: 5
```

**Key Insights**:
- Compares ChatGPT baseline performance against expert ground truth
- Shows which prompt/MCP configurations work better
- Identifies queries where ChatGPT struggled to find correct datasets
- Not a perfect comparison (ChatGPT may suggest valid alternatives not in truth set)

## Understanding Results

### Total Recall Output

```json
{
  "overall_accuracy": 85.5,
  "queries_processed": 18,
  "queries_skipped": 2,
  "query_scores": [
    {
      "query_number": 1,
      "accuracy_percent": 83.3,
      "total_expected_concepts": 6,
      "total_found_concepts": 5,
      "decomposition_details": [
        {
          "topic": "Agricultural Drought",
          "decomp": "Soil Moisture (Surface)",
          "expected_concept_ids": ["C2776463935-NSIDC_ECS"],
          "found_concept_ids": ["C2776463935-NSIDC_ECS"],
          "matched": true,
          "score_percent": 100.0
        }
      ]
    }
  ],
  "summary": {
    "queries_perfect_score": 12,
    "queries_partial_score": 5,
    "queries_zero_score": 1,
    "decomps_fully_matched": 65,
    "decomps_not_matched": 9
  }
}
```

**Interpreting Scores**:
- **Overall accuracy**: Average of all query scores - higher is better
- **Perfect score queries**: 100% accuracy - found all expected datasets
- **Partial score queries**: Found some but not all expected datasets
- **Zero score queries**: Found none of the expected datasets
- **Decomps fully matched**: All expected concept IDs found for these decomps
- **Decomps not matched**: No expected concept IDs found

### Common Issues

**Low accuracy on specific query**:
- Check `decomposition_details` to see which decomps failed
- Look at `found_concept_ids` vs `expected_concept_ids`
- Review agent output to understand why certain datasets weren't found

**Missing reexecuted files**:
- Run `reexecute_cmr_queries.py` first
- Check for errors in `reexecuted/{search_id}_concepts.json` log field
- Some queries may have invalid parameters (temporal format issues)

**Queries skipped**:
- Missing from evaluation_runs.json (wasn't executed)
- Missing reexecuted results file
- Check warnings in accuracy calculation output

## Common Commands

```bash
# Complete workflow from scratch
uv run python evaluations/run_evaluation.py --model gpt-5-mini
uv run python evaluations/reexecute_cmr_queries.py evaluations/run_YYYYMMDD_HHMMSS
uv run python evaluations/calculate_accuracy.py evaluations/run_YYYYMMDD_HHMMSS

# Re-calculate accuracy after fixing issues
uv run python evaluations/reexecute_cmr_queries.py evaluations/run_YYYYMMDD_HHMMSS
uv run python evaluations/calculate_accuracy.py evaluations/run_YYYYMMDD_HHMMSS

# View results
cat evaluations/run_YYYYMMDD_HHMMSS/total_recall.json | python3 -m json.tool | less

# Check specific query details
cat evaluations/run_YYYYMMDD_HHMMSS/total_recall.json | \
    python3 -c "import json, sys; d = json.load(sys.stdin); \
    q = [q for q in d['query_scores'] if q['query_number'] == 1][0]; \
    print(json.dumps(q, indent=2))"
```

## Related Documentation

- [PREPROCESSING.md](PREPROCESSING.md) - Truth set structure and conversion
- [QUERY_EXECUTION.md](QUERY_EXECUTION.md) - Running evaluation queries
- [SCORING.md](SCORING.md) - Legacy scoring methodology (top-5)
- [EXCEL_EXPORT.md](EXCEL_EXPORT.md) - Excel export format specification

## Troubleshooting

**"ERROR: reexecuted directory not found"**
→ Run `reexecute_cmr_queries.py` first to generate full CMR results

**"WARNING: Could not load reexecuted results"**
→ Check if file exists and has valid JSON format
→ Look for errors in the file's `log` field

**"ERROR: evaluation_runs.json not found"**
→ Run `run_evaluation.py` first to execute queries

**Temporal format errors (400 Bad Request)**
→ Some evaluation queries have malformed temporal strings (`:59:59Z` instead of `:59Z`)
→ These are logged in reexecuted results but don't stop processing
→ Original data issue, not a script bug
