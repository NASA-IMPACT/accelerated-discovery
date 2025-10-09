# Data Search System - Data Flow Documentation

This document provides a comprehensive overview of the NASA Earth science data discovery system, detailing how natural language queries are transformed into actual data files through a sophisticated pipeline of LLM-powered components and NASA CMR API interactions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow Pipeline](#data-flow-pipeline)
4. [Parallel Processing](#parallel-processing)
5. [Input/Output Specifications](#inputoutput-specifications)
6. [Error Handling & Retry Logic](#error-handling--retry-logic)
7. [Component Reference](#component-reference)

## System Architecture

The data search system follows a two-layer architecture:

```
┌─────────────────────────────────────────┐
│              Agent Layer                │
│  ┌─────────────────────────────────────┐ │
│  │       CMRDataSearchAgent            │ │
│  │  (Orchestrates entire workflow)     │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │         LLM Components              │ │
│  │  • TopicSplitting                   │ │
│  │  • RepositoryRouter                 │ │
│  │  • ScientificDecomposition          │ │
│  │  • KnownParameters                  │ │
│  │  • SearchableParameters             │ │
│  │  • CollectionRanking                │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│              Tools Layer                │
│  ┌─────────────────────────────────────┐ │
│  │    CMRCollectionSearchTool          │ │
│  │  (Searches for dataset collections) │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │      CMRGranuleSearchTool           │ │
│  │    (Retrieves actual data files)    │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            External APIs                │
│  • NASA CMR API (via MCP server)       │
│  • CMR Keywords/Instruments API        │
│  • OpenAI/Anthropic LLM APIs           │
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component has a single responsibility and well-defined interfaces
2. **Parallelization**: Independent operations execute concurrently for performance
3. **Retry Logic**: Robust error handling with exponential backoff for API calls
4. **Schema Validation**: Pydantic models ensure data integrity throughout the pipeline
5. **Progress Tracking**: Real-time updates via WebSocket for frontend integration

## Core Components

### Tools Layer (`akd/tools/data_search/`)

**File: `_base.py`**
- **BaseDataSearchTool**: Foundation class providing:
  - MCP (Model Context Protocol) server communication
  - HTTP client with retry logic and circuit breaker patterns
  - Request/response validation and error handling
  - Standardized output schemas

**File: `cmr_collection_search.py`**
- **CMRCollectionSearchTool**: Searches NASA's Common Metadata Repository for dataset collections
  - Input: Keywords, platform, instrument, temporal/spatial constraints
  - Output: List of matching collections with metadata
  - MCP Endpoint: `search_collections`

**File: `cmr_granule_search.py`**
- **CMRGranuleSearchTool**: Retrieves actual data files (granules) from collections
  - Input: Collection concept ID, temporal/spatial filters
  - Output: List of downloadable granules with URLs
  - MCP Endpoint: `get_granules`

### Agent Layer (`akd/agents/data_search/`)

**File: `cmr_data_search.py`**
- **CMRDataSearchAgent**: Main orchestrator coordinating the entire workflow
  - Manages component pipeline execution
  - Handles parallel processing coordination
  - Provides progress tracking and WebSocket integration
  - Maintains search state and metadata

**Component Pipeline** (in `components/` directory):

All components inherit from **BaseDataSearchComponent** which provides:
- Automatic prompt template loading
- Retry logic with exponential backoff for rate limiting
- Standardized error handling and logging
- Memory management helpers

1. **TopicSplittingComponent** (`topic_splitting.py`)
2. **RepositoryRouterComponent** (`repository_router.py`)
3. **ScientificDecompositionComponent** (`scientific_decomposition.py`)
4. **KnownParametersComponent** (`known_parameters.py`)
5. **SearchableParametersComponent** (`searchable_parameters.py`)
6. **ApproachCollectionFilteringComponent** (`approach_collection_filtering.py`)
7. **FinalCollectionRankingComponent** (`final_collection_ranking.py`)

### Base Component Architecture

**File: `components/_base.py`**
- **BaseDataSearchComponent**: Abstract base class for all LLM-powered components
  - Provides automatic prompt template loading via `template_name` attribute
  - Implements retry logic with exponential backoff (configurable per component)
  - Standardizes initialization patterns and memory management
  - Reduces code duplication across 8 component implementations
  - **Key Methods**:
    - `_execute_with_retry()`: Standard retry logic for LLM calls
    - `_execute_with_retry_custom()`: Retry logic for custom callables (e.g., schema overriding)
    - `_add_user_message()`: Add to conversation memory
    - `_set_messages()`: Set messages directly (for ranking components)
    - `_format_user_prompt_from_template()`: Format using loaded template
  - **Configuration Attributes**:
    - `template_name`: Prompt template prefix (e.g., "topic_splitting")
    - `default_temperature`: LLM temperature (0.0 = deterministic, 0.1 = slight variation)
    - `retry_enabled`: Enable/disable retry logic (False for ranking components)
    - `max_retries`: Maximum retry attempts (default: 3)
    - `retry_base_delay`: Base delay for exponential backoff (default: 1.0s)

### Utility Components

**File: `utils/prompt_loader.py`**
- Loads and formats prompt templates from the `components/prompts/` directory
- Provides helper functions for prompt template management
- Used by BaseDataSearchComponent for automatic prompt loading

**File: `utils/cmr_keywords_fetcher.py`**
- Fetches and caches CMR metadata (instruments, platforms, science keywords)
- Provides standardized interface to NASA's controlled vocabularies

**File: `utils/cmr_fuzzy_matcher.py`**
- Fuzzy string matching against CMR controlled vocabularies
- Helps normalize user input to CMR-compatible terms

## Data Flow Pipeline

### Overview

The system transforms a natural language query into actual data files through this pipeline:

```
Natural Language Query
       ↓
[1] Topic Splitting (1-6 topics)
       ↓
[2] Repository Routing (per topic)
       ↓
[3] Scientific Decomposition (1-6 per topic)
       ↓
[4] Known Parameters Extraction (1-5 approaches per decomposition)
       ↓
[5] Search Variations Generation (0-5 variations per approach)
       ↓
[6] Collection Search & Ranking (parallel execution across all variations)
       ↓
[7] Granule Search (parallel across collections)
       ↓
Data Files with Download URLs
```

### Detailed Step-by-Step Flow

#### Step 1: Topic Splitting
**Component**: `TopicSplittingComponent`
**Location**: `akd/agents/data_search/components/topic_splitting.py:75`

**Input**:
```python
{
    "query": "Find MODIS sea surface temperature data from 2023 over the Pacific Ocean"
}
```

**Process**:
1. Uses LLM with specialized system prompt to identify functional topics
2. Analyzes query for distinct areas requiring separate data discovery
3. Temperature = 0.0 for consistent analysis

**Output**:
```python
{
    "topics": [
        {
            "title": "Sea Surface Temperature Measurements",
            "functional_context": "Remote sensing observations of ocean thermal properties..."
        }
    ]
}
```

#### Step 2: Repository Routing
**Component**: `RepositoryRouterComponent`
**Location**: `akd/agents/data_search/components/repository_router.py:96`

**Input**: Original query + individual topic

**Process**:
1. LLM determines which data repositories can provide relevant datasets
2. Routes to NASA repositories (CMR, PDS4, GCN, etc.) or external sources
3. Provides rationales for routing decisions

**Output**:
```python
{
    "route": {
        "repositories": ["CMR"],
        "rationales": ["CMR contains NASA satellite-based ocean temperature datasets"]
    }
}
```

#### Step 3: Scientific Decomposition
**Component**: `ScientificDecompositionComponent`
**Location**: `akd/agents/data_search/components/scientific_decomposition.py:83`

**Input**: Original query + topic

**Process**:
1. Decomposes functional topic into 1-6 specific observable phenomena
2. Each decomposition includes scientific justification
3. Focus on measurable quantities available in Earth observation data

**Output**:
```python
{
    "decompositions": [
        {
            "title": "Thermal Infrared Sea Surface Temperature",
            "scientific_justification": "Infrared sensors measure thermal emission from ocean surface..."
        },
        {
            "title": "Microwave Sea Surface Temperature",
            "scientific_justification": "Microwave radiometry provides all-weather SST measurements..."
        }
    ]
}
```

#### Step 4: Known Parameters Extraction
**Component**: `KnownParametersComponent`
**Location**: `akd/agents/data_search/components/known_parameters.py:146`

**Input**: Original query + topic + decomposition

**Process**:
1. Extracts hard filters directly identifiable from context
2. Maps to CMR search parameters (instrument, platform, processing level, etc.)
3. Generates 1-5 query approaches per decomposition

**Output**:
```python
{
    "query_approaches": [
        {
            "instrument": "MODIS",
            "platform": "Terra",
            "processing_level": "Level 2",
            "temporal": "2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
            "bounding_box": "120,-60,180,60"
        }
    ]
}
```

#### Step 5: Searchable Parameters Generation
**Component**: `SearchableParametersComponent`
**Location**: `akd/agents/data_search/components/searchable_parameters.py:154`

**Input**: Original query + topic + decomposition + query approaches

**Process**:
1. For each query approach, generates 0-5 search variations
2. Each variation uses different keyword strategies (or no keywords)
3. Creates multiple targeted CMR queries per approach
4. **Key insight**: CMR uses AND logic, so fewer keywords = more results

**Output**:
```python
{
    "searchable_queries": [
        {
            "instrument": "MODIS",
            "platform": "Terra",
            "combined_keyword_string": ""  # No keywords - use instrument/platform filtering
        },
        {
            "instrument": "MODIS",
            "platform": "Terra",
            "combined_keyword_string": "sea surface temperature"  # Formal terminology
        },
        {
            "instrument": "MODIS",
            "platform": "Terra",
            "combined_keyword_string": "SST"  # Common abbreviation
        }
    ]
}
```

#### Step 6: Collection Search & Ranking (Approach-Aware Pipeline)
**Execution**: `akd/agents/data_search/cmr_data_search.py:712-775`
**Ranking**: `akd/agents/data_search/cmr_data_search.py:929-1038`

The collection search and ranking process now uses a four-stage approach-aware pipeline for better scalability and quality:

**Stage 1: Per-Query Collection Limiting**
Location: `akd/agents/data_search/cmr_data_search.py:712-775`

```python
# Group queries by source approach
for query in searchable_queries:
    approach_idx = query.approach_index  # Each query tagged with approach
    approach_queries[approach_idx].append(query)

# Execute queries grouped by approach
for approach_idx, queries in approach_queries.items():
    for query in queries:
        result = await collection_search_tool.arun(query)
        # Limit to top N per query (default: 5)
        limited = result.collections[:config.collections_per_query]
        approach_collections[approach_idx].extend(limited)
```

**Output**: Up to 5 approaches × 5 queries/approach × 5 collections/query = max 125 collections grouped by approach

**Stage 2: Per-Approach Deduplication**
Location: `akd/agents/data_search/cmr_data_search.py:776-812`

```python
global_seen_ids = set()
for approach_idx in sorted(approach_collections.keys()):
    deduplicated = []
    for collection in approach_collections[approach_idx]:
        concept_id = collection.get("concept_id")
        if concept_id and concept_id not in global_seen_ids:
            global_seen_ids.add(concept_id)
            deduplicated.append(collection)
    deduplicated_by_approach[approach_idx] = deduplicated
```

**Output**: Each approach has ≤25 deduplicated collections

**Stage 3: Per-Approach Filtering and Ranking (Parallel)**
Location: `akd/agents/data_search/cmr_data_search.py:814-927`
Component: `akd/agents/data_search/components/approach_collection_filtering.py`

For each approach in parallel:
```python
# Create filtering task for each approach
filter_input = ApproachCollectionFilteringInputSchema(
    original_query=query,
    topic_title=topic.title,
    decomposition_title=decomp.title,
    # Approach-specific context
    approach_instrument=approach.instrument,
    approach_platform=approach.platform,
    approach_processing_level=approach.processing_level,
    approach_temporal_range=approach.temporal,
    approach_spatial_bounds=approach.bounding_box,
    collections=approach_collections[i],  # ≤25 collections
    max_collections=5  # Select top 5 per approach
)

# LLM applies two-part process:
# 1. Binary filtering (spatial/temporal/resolution/measurement mismatches)
# 2. Selection and ranking of best 0-5 collections
```

**Filtering Criteria** (ApproachCollectionFilteringComponent):
1. **Spatial mismatch**: Collection doesn't cover required area
2. **Temporal mismatch**: Collection doesn't cover required time period
3. **Resolution inadequacy**: Insufficient spatial or temporal resolution
4. **Wrong measurement**: Measures different phenomenon than decomposition

**Output**: Up to 5 approaches × 5 collections = max 25 filtered collections

**Stage 4: Final Cross-Approach Ranking**
Location: `akd/agents/data_search/cmr_data_search.py:929-1038`
Component: `akd/agents/data_search/components/final_collection_ranking.py`

```python
# Flatten all approach results
all_filtered = []
for approach_idx in filtered_by_approach.keys():
    all_filtered.extend(filtered_by_approach[approach_idx])

# Final ranking across all approaches
final_input = FinalCollectionRankingInputSchema(
    original_query=query,
    topic_title=topic.title,
    decomposition_title=decomp.title,
    collections=all_filtered,  # All filtered collections
    max_collections=25  # Return top 25
)

# LLM ranks all collections comparatively (no filtering)
final_result = await final_ranking_component.arun(final_input)
```

**Ranking Criteria** (FinalCollectionRankingComponent):
- **Scientific Relevance (50%)**: Directness of measurement, quality, validation
- **Data Quality (30%)**: Processing level, algorithm maturity, uncertainty
- **Practical Utility (20%)**: Coverage, availability, complementarity

**Final Output**: Up to 25 collections, ranked 1-25 by scientific value

**Configuration Parameters**:
```python
collections_per_query: int = 5          # Top N from each CMR query
max_collections_per_approach: int = 5   # Top N per approach after filtering
final_collection_count: int = 25        # Final ranked output size
approach_filtering_model: str = "gpt-4o-mini"
final_ranking_model: str = "gpt-4o-mini"
```

#### Step 7: Granule Search
**Execution**: `akd/agents/data_search/cmr_data_search.py:709`

**Process**:
1. For each selected collection, search for granules (data files)
2. Apply temporal/spatial constraints from original query
3. Execute searches in parallel across collections
4. Collect granules with download URLs and metadata

**Granule Search Flow**:
```python
for collection in ranked_collections:
    granule_params = {
        "collection_concept_id": collection["concept_id"],
        "temporal": params.temporal_range,
        "bounding_box": params.spatial_bounds,
        "page_size": config.granule_search_page_size
    }

    tool_input = CMRGranuleSearchTool.input_schema(**granule_params)
    result = await granule_search_tool.arun(tool_input)
    granules.extend(result.results["granules"])
```

## Parallel Processing

The system employs several levels of parallelization for optimal performance:

### 1. Topic-Level Parallelism
**Location**: `akd/agents/data_search/cmr_data_search.py:346-405`

Topics are processed in parallel after routing:
```python
# Repository routing executes in parallel for all topics
routing_tasks = [
    self.repository_router_component.process(original_query, topic)
    for topic in topics_output.topics
]
routing_results = await asyncio.gather(*routing_tasks, return_exceptions=True)

# All CMR topics process in parallel through complete pipeline
cmr_topic_tasks = [
    self._process_single_topic(topic, original_query, params)
    for topic, route in zip(topics, routes)
    if NASARepositoryEnum.CMR in route.repositories
]
cmr_results = await asyncio.gather(*cmr_topic_tasks, return_exceptions=True)

# Within each topic, decompositions are also processed in parallel using asyncio.gather:
decomp_tasks = [
    self._process_decomposition(decomp, topic, query, params)
    for decomp in decompositions
]
decomp_results = await asyncio.gather(*decomp_tasks, return_exceptions=True)
```

### 2. Decomposition-Level Parallelism
**Location**: `akd/agents/data_search/cmr_data_search.py:520`

Within each topic, all scientific decompositions execute in parallel:
```python
# Create tasks for parallel execution
decomp_tasks = [
    self._process_single_decomposition(topic, decomp, original_query, params)
    for decomp in decomp_output.decompositions
]

# Execute all decompositions in parallel
decomp_results = await asyncio.gather(*decomp_tasks, return_exceptions=True)
```

### 3. Approach-Level Filtering Parallelism
**Location**: `akd/agents/data_search/cmr_data_search.py:886-900`

Per-approach collection filtering executes in parallel:
```python
# Create filtering task for each approach
filtering_tasks = []
for approach_idx in sorted(approach_collections.keys()):
    filter_input = ApproachCollectionFilteringInputSchema(
        # ... approach-specific parameters
        collections=approach_collections[approach_idx],
    )
    filtering_component = ApproachCollectionFilteringComponent(config=component_config)
    task = filtering_component.arun(filter_input)
    filtering_tasks.append((approach_idx, collections, task))

# Execute in parallel (up to 5 approaches concurrently)
if len(filtering_tasks) > 1:
    results = await asyncio.gather(
        *[task for _, _, task in filtering_tasks],
        return_exceptions=True
    )
```

**Benefits**:
- Reduces wall-clock time for multi-approach queries
- Each approach can have different filtering criteria based on instrument/platform
- Failures in one approach don't block others

### 4. Granule Search Parallelism
**Location**: `akd/agents/data_search/cmr_data_search.py:1108`

Granule searches across collections execute in parallel:
```python
granule_tasks = [
    self._execute_granule_search(search_params, concept_id)
    for collection in selected_collections
    if (concept_id := collection.get("concept_id"))
]

if self.config.enable_parallel_search and len(granule_tasks) > 1:
    results = await asyncio.gather(*granule_tasks, return_exceptions=True)
```

### Error Handling in Parallel Operations

All parallel operations include robust error handling:
```python
# Filter successful results from parallel execution
successful_results = []
for i, result in enumerate(results):
    if isinstance(result, Exception):
        log_component_action("Search", "FAILED", {"error": str(result)})
    else:
        successful_results.append(result.results)
```

## Performance Optimizations

### Recent Improvements (2024-2025)

**1. Topic and Repository Routing Parallelization (2025)**
- **Issue**: Topics and routing were processed sequentially, limiting throughput for multi-topic queries
- **Solution**: Implemented `asyncio.gather()` for parallel topic routing and processing
- **Impact**: 14% reduction in wall-clock time for single-topic queries; expected 40-70% for multi-topic queries
- **Location**: `akd/agents/data_search/cmr_data_search.py:326-405`

**2. Decomposition Parallelization (2024-2025)**
- **Issue**: Scientific decompositions were processed serially, causing ~10x slower performance
- **Solution**: Implemented `asyncio.gather()` for parallel decomposition processing within each topic
- **Impact**: Reduced processing time from ~17 minutes to ~2 minutes for typical workflows
- **Location**: `akd/agents/data_search/cmr_data_search.py:520`

**3. Validation Limits**
- **Issue**: Searchable parameters component limited to 15 queries but generated up to 25
- **Solution**: Updated validation limit from 15 to 25 in `SearchableParametersOutput`
- **Impact**: Eliminated validation errors that caused workflow failures
- **Location**: `akd/agents/data_search/components/searchable_parameters.py:106`

**4. Model Configuration**
- **Current**: All components use `gpt-5-mini` for optimal cost/performance balance
- **Previous**: Mixed `gpt-4o` and `gpt-4o-mini` configuration
- **Impact**: Consistent performance across all components with OpenAI's latest efficient model

### Performance Monitoring

Use the timing collection system for performance analysis:
```bash
# Capture workflow with timing data
uv run demo_capture.py --query "your query" --output timing_test.json

# Analyze performance bottlenecks
uv run analyze_timing.py timing_test.json --report bottlenecks
```

## Input/Output Specifications

### Agent Input Schema
**Location**: `akd/agents/data_search/_base.py:13`

```python
class DataSearchAgentInputSchema(InputSchema):
    query: str  # Natural language research question
    temporal_range: Optional[str]  # "YYYY-MM-DD,YYYY-MM-DD"
    spatial_bounds: Optional[str]  # "west,south,east,north"
    max_results: int = 50  # Maximum granules to return
```

### Agent Output Schema
**Location**: `akd/agents/data_search/_base.py:147`

```python
class DataSearchAgentOutputSchema(OutputSchema):
    # New topic-based structure
    topics: List[TopicResult]  # Results organized by topic
    search_metadata: dict  # Search provenance and metadata
    total_results: int  # Total granules found

    # Legacy compatibility fields
    angles: List[AngleSearchResult] = []
    granules: List[dict] = []
    collections_searched: List[dict] = []
```

### Topic Result Structure
**Location**: `akd/agents/data_search/_base.py:62`

```python
class TopicResult(BaseModel):
    topic: Dict[str, Any]  # Original topic with title and context
    data_source: str  # "CMR", "USGS", etc.
    decomposition_results: List[DecompositionResult]
    note: Optional[str]  # Routing notes or availability info
```

### Decomposition Result Structure
**Location**: `akd/agents/data_search/_base.py:84`

```python
class DecompositionResult(BaseModel):
    decomposition: Dict[str, Any]  # Scientific decomposition
    query_approaches: List[Dict[str, Any]]  # Known parameter approaches
    searchable_queries: List[Dict[str, Any]]  # Complete search queries
    collections: List[Dict[str, Any]]  # Ranked collections
    granules: List[Dict[str, Any]]  # Final data files
    total_collections_found: int
    total_granules_found: int
```

### Tool Input/Output Schemas

**Collection Search Tool**:
```python
# Input
class CMRCollectionSearchInputSchema(DataSearchToolInputSchema):
    keyword: Optional[str]
    short_name: Optional[str]
    platform: Optional[str]
    instrument: Optional[str]
    processing_level: Optional[str]
    # Inherited: temporal, bounding_box, page_size, page_num

# Output
class CMRCollectionSearchOutputSchema(DataSearchToolOutputSchema):
    collections: list  # CMR collection metadata
    # Inherited: results, total_hits, query_time_ms, page_info
```

**Granule Search Tool**:
```python
# Input
class CMRGranuleSearchInputSchema(DataSearchToolInputSchema):
    collection_concept_id: str  # Required
    producer_granule_id: Optional[str]
    downloadable: Optional[bool]
    # Inherited: temporal, bounding_box, page_size, page_num

# Output
class CMRGranuleSearchOutputSchema(DataSearchToolOutputSchema):
    granules: list  # CMR granule metadata with download URLs
    collection_concept_id: str
    # Inherited: results, total_hits, query_time_ms, page_info
```

## Error Handling & Retry Logic

### Tool-Level Error Handling
**Location**: `akd/tools/data_search/_base.py:234`

All tools inherit comprehensive error handling:
```python
async def _make_http_request(self, tool_name: str, arguments: dict) -> dict:
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.post(str(config.mcp_endpoint), ...)

                if response.status_code == 200:
                    return self._parse_mcp_response(response.text)
                else:
                    # HTTP error - retry with exponential backoff
                    wait_time = config.retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)

        except httpx.TimeoutException:
            # Timeout - retry with exponential backoff
        except Exception as e:
            # Other errors - retry or fail
```

### Component-Level Error Handling
**Location**: `akd/agents/data_search/components/_base.py:97-152`

LLM components inherit standardized retry logic from BaseDataSearchComponent:
```python
# All components with retry_enabled=True use this pattern
async def _execute_with_retry(self, operation_name: str, custom_error_prefix: Optional[str] = None) -> TOutput:
    """Execute LLM call with retry logic and exponential backoff."""
    if not self.retry_enabled:
        return await self.get_response_async()

    for attempt in range(self.max_retries + 1):
        try:
            response = await self.get_response_async()
            return response
        except Exception as e:
            if attempt == self.max_retries:
                raise RuntimeError(f"{error_prefix} after {self.max_retries + 1} attempts: {e}")

            # Rate limiting detection
            if "429" in str(e) or "rate" in str(e).lower():
                delay = self.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(f"{error_prefix}: {e}")
```

**Components with retry enabled**: TopicSplitting, RepositoryRouter, ScientificDecomposition, KnownParameters, SearchableParameters (5 of 7)

**Components with retry disabled**: ApproachCollectionFiltering, FinalCollectionRanking (2 ranking components)

### Agent-Level Error Handling
**Location**: `akd/agents/data_search/cmr_data_search.py:418`

The main agent provides graceful degradation:
```python
try:
    # Execute complete pipeline
    return final_response
except Exception as e:
    error_msg = f"Topic-based data search failed: {e}"
    search_logger.error(error_msg)

    await self._emit_progress_safely("on_search_error", error_msg)
    return self._create_error_response(original_query, error_msg)
```

## Component Reference

### Configuration
**Location**: `akd/agents/data_search/cmr_data_search.py:53`

```python
class CMRDataSearchAgentConfig(DataSearchAgentConfig):
    # MCP server configuration
    mcp_endpoint: HttpUrl = "http://localhost:8080/mcp/cmr/mcp/"

    # Search behavior
    collection_search_page_size: int = 20
    granule_search_page_size: int = 50

    # New approach-aware ranking configuration
    collections_per_query: int = 5           # Top N from each CMR query
    max_collections_per_approach: int = 5    # Top N per approach after filtering
    final_collection_count: int = 25         # Final ranked output size

    min_collection_relevance_score: float = 0.3

    # Performance tuning
    collection_search_timeout: float = 30.0
    granule_search_timeout: float = 45.0
    enable_parallel_search: bool = True

    # Model configuration per component
    topic_splitting_model: str = "gpt-5-mini"
    scientific_decomposition_model: str = "gpt-5-mini"
    repository_routing_model: str = "gpt-5-mini"
    collection_ranking_model: str = "gpt-5-mini"  # Legacy (for angles workflow)
    cmr_query_model: str = "gpt-5-mini"
    approach_filtering_model: str = "gpt-5-mini"  # Per-approach filtering
    final_ranking_model: str = "gpt-5-mini"       # Final cross-approach ranking
```

### Prompt Templates
**Location**: `akd/agents/data_search/components/prompts/`

Each component uses specialized prompts:
- `topic_splitting_system.md` / `topic_splitting_user.md`
- `repository_routing_system.md` / `repository_routing_user.md`
- `scientific_decomposition_system.md` / `scientific_decomposition_user.md`
- `known_parameters_system.md` / `known_parameters_user.md`
- `searchable_parameters_system.md` / `searchable_parameters_user.md`
- `collection_ranking_system.md` / `collection_ranking_user.md` (legacy)
- **NEW**: `approach_filtering_system.md` / `approach_filtering_user.md`
- **NEW**: `final_ranking_system.md` / `final_ranking_user.md`

### Progress Tracking
**Location**: `akd/agents/data_search/cmr_data_search.py:229`

Real-time progress updates via WebSocket:
```python
async def _emit_progress_safely(self, method_name: str, *args, **kwargs):
    if not self.progress_handler:
        return

    try:
        method = getattr(self.progress_handler, method_name)
        await method(*args, **kwargs)
    except Exception as e:
        # Progress failures don't stop the search
        self.agent_logger.warning(f"Progress update failed: {e}")
```

### Logging and Observability
**Location**: Throughout codebase

Comprehensive logging using structured events:
```python
from akd.utils.logging import log_component_action, log_search_event

log_search_event(search_id, "SEARCH_STARTED", {"query": query})
log_component_action("TopicSplitting", "STARTED", {"query": query})
log_component_action("CollectionSearch", "EXECUTE", {"params": params})
```

### Legacy Compatibility
**Location**: `akd/agents/data_search/_base.py:162`

The system maintains backward compatibility with the previous "angles" structure while transitioning to the new topic-based workflow.

---

This documentation represents the complete data flow architecture as of the current codebase. For implementation details of individual components, refer to the source files referenced throughout this document.
