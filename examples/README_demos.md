# Data Search Agent Demo System

This directory contains three complementary demo scripts for the data search agent workflow:

## 📋 Demo Scripts Overview

### 1. `demo_simple.py` - Single Path Demo
**Purpose**: Clean, linear demonstration of the workflow with one topic and one decomposition.

**Usage**:
```bash
uv run demo_simple.py                          # Use default query
uv run demo_simple.py --query "your query"     # Custom query
uv run demo_simple.py --component topic        # Test individual component
```

**What it does**:
- Calls each component individually and explicitly
- Processes only the **first topic** and **first decomposition**
- Shows step-by-step workflow with clear output
- Perfect for understanding the basic pipeline

### 2. `demo_capture.py` - Full Workflow Capture
**Purpose**: Run complete multi-path workflow and save all intermediate results for later testing.

**Usage**:
```bash
uv run demo_capture.py                                    # Use default query
uv run demo_capture.py --query "MODIS temperature data"  # Custom query
uv run demo_capture.py --output my_results.json          # Custom output file
```

**What it does**:
- Processes **ALL topics** and **ALL decompositions** (complete multi-path)
- Captures every intermediate result from each component
- Saves everything to JSON in `captured_data/` directory
- Creates comprehensive test data for debugging

**Output Structure**:
```json
{
  "query": "your research query",
  "timestamp": "2025-01-15T10:30:00",
  "topics": [
    {
      "index": 0,
      "topic": { /* Topic object */ },
      "routing": { /* Repository routing results */ },
      "decompositions": [
        {
          "index": 0,
          "decomposition": { /* ScientificDecomposition object */ },
          "known_params": { /* Known parameters output */ },
          "searchable_params": { /* Searchable queries */ },
          "collections_raw": [ /* Collection search results */ ],
          "collections_ranked": [ /* Ranked collections */ ],
          "granules": [ /* Granule search results */ ]
        }
      ]
    }
  ],
  "metadata": { /* Summary statistics and config */ }
}
```

### 3. `demo_loader.py` - Component Testing with Saved Data
**Purpose**: Test individual components using captured data without re-running the entire pipeline.

**Usage**:
```bash
# List available data
uv run demo_loader.py captured_data/my_file.json --list

# Test specific components
uv run demo_loader.py captured_data/my_file.json --component topic_splitting
uv run demo_loader.py captured_data/my_file.json --component known_parameters --topic 0 --decomp 1
uv run demo_loader.py captured_data/my_file.json --component searchable_parameters --topic 1 --decomp 0

# Compare fresh vs saved results
uv run demo_loader.py captured_data/my_file.json --component known_parameters --compare
```

**What it does**:
- Loads captured JSON files
- Reconstructs Pydantic objects from saved data
- Tests any component with appropriate saved inputs
- Compares fresh results with saved results
- Perfect for iterating on prompts and component logic

## 🔄 Typical Debugging Workflow

1. **Capture data once**:
   ```bash
   uv run demo_capture.py --query "your research query"
   ```

2. **Test component iterations**:
   ```bash
   # Test known parameters with different topics/decompositions
   uv run demo_loader.py captured_data/captured_*.json --component known_parameters --topic 0 --decomp 0
   uv run demo_loader.py captured_data/captured_*.json --component known_parameters --topic 1 --decomp 0

   # Test searchable parameters after updating prompts
   uv run demo_loader.py captured_data/captured_*.json --component searchable_parameters --topic 0 --decomp 1
   ```

3. **Compare results**:
   ```bash
   # See if your prompt changes improved results
   uv run demo_loader.py captured_data/captured_*.json --component known_parameters --compare
   ```

## 📁 File Organization

```
examples/
├── demo_simple.py           # Single-path demo
├── demo_capture.py          # Multi-path capture
├── demo_loader.py           # Component testing
├── captured_data/           # Saved workflow data
│   ├── captured_flood_risk_20250115_103000.json
│   ├── captured_temperature_data_20250115_110000.json
│   └── ...
└── README_demos.md          # This file
```

## 🧪 Component Testing Matrix

The loader supports testing these components with appropriate inputs:

| Component | Inputs Required | Example Command |
|-----------|----------------|-----------------|
| `topic_splitting` | query only | `--component topic_splitting` |
| `repository_routing` | query + topic | `--component repository_routing --topic 0` |
| `scientific_decomposition` | query + topic | `--component scientific_decomposition --topic 1` |
| `known_parameters` | query + topic + decomp | `--component known_parameters --topic 0 --decomp 1` |
| `searchable_parameters` | query + topic + decomp + known_params | `--component searchable_parameters --topic 1 --decomp 0` |
| `collection_ranking` | query + topic + decomp + collections | `--component collection_ranking --topic 0 --decomp 0` |

## 🔧 Configuration

All demos use the same model configuration for consistency:

```python
MODEL_CONFIG = {
    "topic_splitting": "gpt-5-mini",
    "scientific_decomposition": "gpt-5-mini",
    "repository_routing": "gpt-5-mini",
    "collection_ranking": "gpt-5-mini",
    "cmr_query": "gpt-5-mini",
}
```

## 💡 Tips

- **Start with `demo_simple.py`** to understand the basic workflow
- **Use `demo_capture.py`** to create test data for the queries you care about
- **Use `demo_loader.py`** for focused debugging of specific components
- **Captured data files** can be version controlled for regression testing
- **Use `--list`** with the loader to see what data is available
- **Use `--compare`** to validate that prompt changes improve results

## 🚨 Notes

- Capture demo runs the expensive collection/granule searches - expect longer runtime
- Loader demo only runs the LLM components - much faster iteration
- All demos require valid OpenAI API key in environment
- Captured data includes the exact agent configuration used for reproducibility
