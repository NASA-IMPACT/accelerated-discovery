# Data Search Agent Testing Framework

A comprehensive testing framework for the NASA Earth science data discovery system, providing systematic validation of component outputs against known test cases using LLM-as-judge evaluation.

## Overview

This testing framework enables:
- **Systematic Testing**: Run test queries against the data search system and capture results
- **LLM-based Evaluation**: Compare actual vs expected results using binary LLM judges
- **Component-level Testing**: Test individual components in isolation
- **Regression Detection**: Compare test runs to identify improvements or regressions
- **Comprehensive Reporting**: Generate detailed test reports with pass/fail analysis

## Architecture

```
testing/
├── test_cli.py                 # Main CLI interface
├── test_runner.py             # Test execution engine
├── test_data_manager.py       # CSV test data management
├── test_models.py             # Data structures and schemas
├── evaluators/                # LLM-based evaluation components
│   ├── base_evaluator.py
│   ├── topic_evaluator.py
│   └── decomposition_evaluator.py
├── test_data/                 # Test cases and expected results
│   └── queries.csv           # Test queries with expected outputs
├── test_runs/                # Saved test execution results
└── captured_data/            # Captured workflow data files
```

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager
- OpenAI API key configured
- Access to the data search agent system

### Installation

```bash
# Navigate to the testing directory
cd examples/testing

# Ensure dependencies are installed (from parent directory)
cd .. && uv sync && cd testing
```

### Basic Usage

```bash
# List available test cases
uv run python test_cli.py --list

# Run a limited test suite (2 test cases)
uv run python test_cli.py --mode full --max-tests 2

# Run a single specific test
uv run python test_cli.py --mode single --query "urbanization heat island"

# Evaluate captured workflow data
uv run python test_cli.py --mode evaluate --captured-file captured_data/my_file.json --component topic_splitting
```

## Test Data Format

Test cases are defined in `test_data/queries.csv` with the following structure:

| Column | Description |
|--------|-------------|
| `query` | Research question to test |
| `topic` | Expected topic name |
| `division` | Category (earth, space, etc.) |
| `decomps` | Expected scientific decomposition |
| `minimum` | Whether decomposition is required (yes/no) |
| `justification` | Justification for the decomposition |
| `useful background` | Additional context information |

Example:
```csv
query,topic,division,decomps,minimum,justification
"How has urbanization influenced the Urban Heat Island effect?",urbanization,earth,Land Cover/Land Use,yes,"Urban expansion replaces natural surfaces..."
```

## Testing Modes

### 1. Full Test Suite (`--mode full`)

Runs all test cases through the complete workflow pipeline:

```bash
# Run all test cases
uv run python test_cli.py --mode full

# Limit to first 3 test cases
uv run python test_cli.py --mode full --max-tests 3

# Filter by category
uv run python test_cli.py --mode full --category earth
```

**Process:**
1. Loads test cases from CSV
2. Executes each query through the data search workflow
3. Captures complete workflow data to JSON files
4. Generates test run summary and detailed reports

### 2. Single Test (`--mode single`)

Tests a specific query:

```bash
uv run python test_cli.py --mode single --query "urbanization heat island"
```

**Use cases:**
- Debug specific test failures
- Test new queries
- Rapid iteration during development

### 3. Fast Smoke Test (`--mode fast-smoke`)

Ultra-fast single-path execution for rapid validation:

```bash
# Use first test case from CSV
uv run python test_cli.py --mode fast-smoke

# Use custom query
uv run python test_cli.py --mode fast-smoke --query "sea ice extent"
```

**Characteristics:**
- Uses `gpt-5-nano` for all components (fastest model)
- Single-path execution: selects `[0]` at every branch point
- Skips repository routing (assumes CMR)
- Sequential execution (no parallelization)
- Minimal collection/granule counts

**Execution Path:**
```
Topic Splitting → topics[0]
↓ (skip routing)
Scientific Decomposition → decompositions[0]
↓
Known Parameters → approaches[0]
↓
Searchable Parameters → queries[0]
↓
Collection Search → collections[0]
↓
Granule Search → granules[0:5]
```

**Expected Runtime:** 20-40 seconds (vs 60-120s for normal single test)

**Use cases:**
- Quick smoke tests before deployments
- Rapid pipeline validation during development
- CI/CD integration for fast feedback
- Debugging component connectivity issues

### 4. Evaluation Mode (`--mode evaluate`)

Evaluates captured workflow data using LLM judges:

```bash
# Evaluate topic splitting
uv run python test_cli.py --mode evaluate \
  --captured-file captured_data/my_test.json \
  --component topic_splitting

# Evaluate scientific decomposition
uv run python test_cli.py --mode evaluate \
  --captured-file captured_data/my_test.json \
  --component scientific_decomposition
```

**Available Components:**
- `topic_splitting`: Evaluates topic identification
- `scientific_decomposition`: Evaluates scientific decompositions
- `known_parameters`: Evaluates parameter extraction
- `searchable_parameters`: Evaluates search query generation

## LLM Evaluation System

The framework uses specialized LLM evaluators for each component:

### Topic Evaluator
- **Criteria**: Coverage, semantic similarity, completeness, relevance
- **Focus**: Whether generated topics capture expected research areas
- **Output**: Pass/fail with confidence score and detailed coverage analysis

### Decomposition Evaluator
- **Criteria**: Scientific accuracy, minimum requirements coverage, relevance
- **Focus**: Essential decompositions are covered (especially minimum required ones)
- **Output**: Pass/fail with minimum requirement tracking and gap analysis

### Base Evaluation Format

All evaluators return structured results:
```python
{
    "passed": true/false,
    "confidence": 0.85,
    "reasoning": "Detailed explanation...",
    "specific_feedback": {
        "missing_topics": [...],
        "coverage_analysis": {...}
    }
}
```

## Test Results

### Output Structure

Each test run generates:

1. **Results JSON** (`test_runs/{run_id}_results.json`):
   - Complete test execution data
   - Component outputs and evaluations
   - Timing and performance metrics

2. **Summary Markdown** (`test_runs/{run_id}_summary.md`):
   - Human-readable test summary
   - Pass/fail breakdown
   - Individual test case results

3. **Captured Data** (`captured_data/{run_id}_{test_id}.json`):
   - Complete workflow execution data
   - All intermediate component outputs
   - Can be used for component-level testing

### Example Results

```
📊 TEST SUITE RESULTS
🔍 Run ID: test_run_20250928_144500
📈 Total Tests: 3
✅ Passed: 2
❌ Failed: 1
🚨 Errors: 0
📊 Success Rate: 66.7%
⏱️  Duration: 245.3s
```

## Integration with Existing Demo System

The testing framework reuses and extends the existing demo infrastructure:

- **demo_capture.py**: Used for workflow execution and data capture
- **demo_loader.py**: Used for component-level testing with saved data
- **Workflow compatibility**: All captured data is compatible with existing tools

## Configuration

### Model Configuration

Components use configurable LLM models (see `demo_capture.py`):

**Standard Testing:**
```python
MODEL_CONFIG = {
    "topic_splitting": "gpt-5-mini",
    "scientific_decomposition": "gpt-5-mini",
    "repository_routing": "gpt-5-mini",
    "collection_ranking": "gpt-5-mini",
    "cmr_query": "gpt-5-mini",
}
```

**Fast Smoke Testing:**
All components use `gpt-5-nano` for maximum speed. Configuration is automatically applied when using `--mode fast-smoke`.

### Evaluation Configuration

Evaluators support configuration:
```python
evaluator = TopicEvaluator(
    model="gpt-5-mini",
    max_retries=3,
    retry_delay=1.0,
    debug=True
)
```

## Advanced Usage

### Custom Test Cases

Add new test cases to `test_data/queries.csv`:

```csv
query,topic,division,decomps,minimum,justification
"Your research question",topic_name,earth,decomposition_name,yes,"Why this decomposition is needed"
```

### Component-Level Testing

Test specific components using captured data:

```bash
# First capture data
uv run ../demo_capture.py --query "your test query"

# Then evaluate specific components
uv run python test_cli.py --mode evaluate \
  --captured-file captured_data/captured_your_test_*.json \
  --component topic_splitting
```

### Regression Testing

Compare test runs:

```bash
# List available test runs
uv run python test_cli.py --list-runs

# TODO: Implement regression comparison
# uv run python test_cli.py --mode regression --baseline run_A --current run_B
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the testing directory
2. **Missing API Key**: Set `OPENAI_API_KEY` environment variable
3. **No Test Cases Found**: Check CSV file path and format
4. **Evaluation Failures**: Check LLM model availability and rate limits

### Debug Mode

Enable detailed logging:
```bash
export LOGURU_LEVEL=DEBUG
uv run python test_cli.py --mode full --max-tests 1
```

### Test Data Validation

Validate test data format:
```bash
uv run python test_cli.py --list
# Look for warnings about malformed test cases
```

## Extending the Framework

### Adding New Evaluators

1. Inherit from `BaseLLMEvaluator`
2. Implement required methods:
   - `_get_system_prompt()`
   - `_format_user_prompt()`
   - `_get_component_name()`

### Adding New Test Modes

1. Add mode to CLI argument choices
2. Implement handler method in `TestCLI` class
3. Add corresponding functionality to `TestRunner`

### Custom Evaluation Criteria

Modify evaluator system prompts to adjust evaluation criteria:
- Scientific accuracy requirements
- Minimum coverage thresholds
- Domain-specific terminology handling

## Performance Considerations

- **Full test suites** can take 10-30 minutes depending on test count
- **LLM evaluations** add 2-5 seconds per component per test case
- **Parallel execution** is supported for collection and granule searches
- **Rate limiting** is handled automatically with exponential backoff

## Future Enhancements

- **Regression testing**: Automated comparison between test runs
- **Semantic similarity**: Enhanced evaluation using embedding models
- **Performance benchmarking**: Track component execution times
- **Test case generation**: Automatic test case creation from successful queries
- **CI/CD integration**: Automated testing in development workflows
