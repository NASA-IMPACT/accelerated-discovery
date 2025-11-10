# Excel Export Format Specification

## Overview

The evaluation system can export results to Excel format with a hierarchical structure using merged cells to represent the relationship between queries, decompositions, and individual dataset results.

**Important**: The export includes only the **top 5 ranked collections per decomposition**, matching the evaluation criteria in SCORING.md where users typically review only the highest-ranked results.

## Sheet Structure

### Column Schema

| Column | Name | Data Type | Purpose |
|--------|------|-----------|---------|
| A | (Unnamed) | Empty | Reserved/unused index column |
| B | Query | Text | The original user query |
| C | Topic | Text | Primary topic extracted from the query |
| D | Decomp | Text | Query decomposition or search angle |
| E | (Unnamed) | Text | Scientific justification for the decomposition approach |
| F | CMR Title | Text | Title of the CMR dataset result |
| G | Start | Date | Temporal start date for the dataset |
| H | End | Date | Temporal end date for the dataset |
| I | Bbox | Text | Spatial bounding box (N/S/E/W coordinates) |
| J | Link | URL | CMR concept URL for the dataset |
| K | Bol | Text | Boolean flag ('y'/'n') indicating match quality |
| L | Comment | Text | Optional comments about the result |
| M | Missing Perfect | Text | Flag indicating if perfect match is missing |

### Hierarchical Merged Cell Structure

The format uses merged cells to create a hierarchical representation:

#### Level 1: Query (Column B)
- **Merge pattern**: Single query merged across ALL rows for that query
- **Purpose**: Groups all decompositions and results under one user query
- **Example merge**: B2:B14 (13 rows for a single query)

#### Level 2: Topic (Column C)
- **Merge pattern**: Single topic merged across ALL rows for that query
- **Purpose**: Identifies the primary topic of the query
- **Example merge**: C2:C14 (spans same rows as query)

#### Level 3: Decomposition (Column D)
- **Merge pattern**: Multiple decomposition groups within a single query
- **Purpose**: Represents different search angles or approaches
- **Example merges**:
  - D2:D6 (5 rows for first decomposition)
  - D7:D11 (5 rows for second decomposition)
  - D12:D14 (3 rows for third decomposition)

#### Level 4: Justification (Column E)
- **Merge pattern**: Mirrors the decomposition groups exactly
- **Purpose**: Provides scientific rationale for each decomposition approach
- **Example merges**: E2:E6, E7:E11, E12:E14 (matches decomposition groups)

#### Level 5: Individual Results (Columns F-M)
- **Merge pattern**: No merging - one dataset per row
- **Purpose**: Lists individual CMR dataset results for each decomposition

## Data Hierarchy Example

```
Query (merged across 13 rows)
├── Topic (merged across 13 rows)
│   ├── Decomposition 1 (merged across 5 rows)
│   │   ├── Justification 1 (merged across 5 rows)
│   │   ├── Dataset Result 1 (row 1)
│   │   ├── Dataset Result 2 (row 2)
│   │   ├── Dataset Result 3 (row 3)
│   │   ├── Dataset Result 4 (row 4)
│   │   └── Dataset Result 5 (row 5)
│   ├── Decomposition 2 (merged across 5 rows)
│   │   ├── Justification 2 (merged across 5 rows)
│   │   ├── Dataset Result 6 (row 6)
│   │   ├── Dataset Result 7 (row 7)
│   │   ├── Dataset Result 8 (row 8)
│   │   ├── Dataset Result 9 (row 9)
│   │   └── Dataset Result 10 (row 10)
│   └── Decomposition 3 (merged across 3 rows)
│       ├── Justification 3 (merged across 3 rows)
│       ├── Dataset Result 11 (row 11)
│       ├── Dataset Result 12 (row 12)
│       └── Dataset Result 13 (row 13)
```

## Merged Cell Ranges

For a typical single-query export with 3 decompositions and 13 total results:

1. **B2:B14** - Query column (spans all 13 data rows)
2. **C2:C14** - Topic column (spans all 13 data rows)
3. **D2:D6** - First decomposition (5 rows)
4. **D7:D11** - Second decomposition (5 rows)
5. **D12:D14** - Third decomposition (3 rows)
6. **E2:E6** - First justification (matches first decomposition)
7. **E7:E11** - Second justification (matches second decomposition)
8. **E12:E14** - Third justification (matches third decomposition)

## Design Rationale

### Visual Grouping
The merged cell structure provides immediate visual clarity when viewing the spreadsheet:
- Easy to see which results belong to which decomposition
- Clear grouping of related datasets
- Reduces visual clutter by not repeating identical text

### Data Integrity
- Each decomposition can have a variable number of results
- The merge ranges automatically adjust to the number of results
- Maintains the hierarchical relationship without requiring separate sheets

### Compatibility
- Standard Excel format (.xlsx)
- Readable by all major spreadsheet applications
- Preserves structure when opening/editing manually
- Can be parsed programmatically using libraries like `openpyxl` or `pandas`

## Automated Export Script

### Usage

The `export_to_excel.py` script converts captured_data JSON files into this Excel format:

```bash
# Export an entire evaluation run to Excel
uv run python evaluations/export_to_excel.py --runs evaluations/run_20251027_213136/evaluation_runs.json

# Specify custom output location
uv run python evaluations/export_to_excel.py --runs evaluations/run_20251027_213136/evaluation_runs.json --output my_results.xlsx
```

### Data Mapping

The script extracts data from the JSON structure as follows:

| Excel Column | Source JSON Path |
|--------------|------------------|
| B (Query) | `agent_output.search_metadata.original_query` |
| C (Topic) | `topics[i].topic.title` |
| D (Decomp) | `decomposition_results[j].decomposition.title` |
| E (Justification) | `decomposition_results[j].decomposition.scientific_justification` |
| F (CMR Title) | `data_results[k].entry_title` |
| G (Start) | `data_results[k].time_start` (formatted YYYY-MM-DD) |
| H (End) | `data_results[k].time_end` (formatted YYYY-MM-DD) |
| I (Bbox) | `searchable_queries[*].bounding_box` (first non-null) |
| J (Link) | Constructed: `https://cmr.earthdata.nasa.gov/search/concepts/{concept_id}` |
| K-M | Empty (for manual annotation) |

### Output

- **Default location**: Same directory as `evaluation_runs.json` → `results_export.xlsx`
- **Header row**: Row 1 contains column names
- **Data rows**: Start at row 2, continuous with no gaps between queries
- **File format**: Excel 2007+ (.xlsx) with merged cells and formatting

### Verification

Verify the Excel structure using:

```bash
uv run python evaluations/verify_excel.py evaluations/run_20251027_213136/results_export.xlsx
```

This shows worksheet dimensions, headers, sample data, and merged cell ranges.

## Usage Notes

### Reading the Format
When parsing programmatically:
- Use `openpyxl` to detect merged cell ranges
- The top-left cell of a merged range contains the actual value
- Other cells in the range will read as `None` in pandas
- Use `merged_cells.ranges` to identify the grouping structure

### Creating the Format
When generating exports:
1. Write all data values to individual cells
2. Identify grouping boundaries (query, decomposition)
3. Apply merges using the hierarchy pattern
4. Ensure justification merges match decomposition merges exactly

### Multi-Query Exports
For multiple queries in a single sheet:
- Each query starts a new merge group
- No merging occurs across different queries
- Maintain the same column structure throughout
- Consider adding visual separators (borders, colors) between queries
