# Evaluation Scoring Methodology

## Overview

Compare agent output against subject matter expert (SME) ground truth to measure retrieval accuracy.

## Scoring Logic

### Decomposition-Level Scoring

Each `minimum: true` decomposition in the truth set is scored **binary**:

- **100%** - At least one truth set CMR concept ID found in agent output
- **0%** - No truth set CMR concept IDs found in agent output

**Ignore**: All `minimum: false` decompositions (not required for valid answer)

### Query-Level Scoring

**Query Score = Average of all minimum decomposition scores**

Example:
- Query has 5 decompositions marked `minimum: true`
- 4 decompositions match (100% each)
- 1 decomposition missing (0%)
- **Query Score = (100 + 100 + 100 + 100 + 0) / 5 = 80%**

## Matching Process

1. Extract **top 5 collections** from each decomposition in agent output
2. For each `minimum: true` decomposition in truth set:
   - Check if ANY of its `cmr_concept_ids` appear in agent's top 5 for any decomposition
   - Score: 100% if found, 0% if not
3. Average all minimum decomposition scores

**Note**: Only the top 5 ranked collections per agent decomposition are evaluated. This reflects the practical constraint that users typically review only the highest-ranked results.

## Data Sources

**Truth Set**: `evaluations/truth_set_20251027.json`
- 20 queries from SMEs (Nidhi, Emily)
- Hierarchical: Query → Topics → Decompositions → CMR Concept IDs
- `minimum: true` = required for correct answer

**Agent Output**: `captured_data/{search_id}.json`
- Full agent execution output
- Check `agent_output.topics[*].decomposition_results[*].collections[*].concept_id`

## Example

**Truth Set** (Query 1, Agricultural Drought):
```json
{
  "decomp": "Soil Moisture (Surface)",
  "minimum": true,
  "cmr_concept_ids": ["C2776463935-NSIDC_ECS"]
}
```

**Agent Output** contains `C2776463935-NSIDC_ECS` → **100%**

**Agent Output** missing this ID → **0%**
