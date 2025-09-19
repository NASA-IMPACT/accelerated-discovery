# Agent Workflow Summary

## Core Philosophy

The data search agent addresses the fundamental challenge that **one single CMR query cannot reliably answer complex science questions**. Instead of attempting to generate perfect queries in a single shot, the system systematically decomposes and refines the original question through multiple structured steps.

## High-Level Flow

```
Science Question → Topic Splitting → Scientific Decomposition → Query Generation → Collection Filtering
                     ↓ (parallelized)     ↓ (parallelized)        ↓ (parallelized)      ↓
                   [Topic 1]           [Decomp 1.1]           [Query Approach 1]    [Ranked Collections]
                   [Topic 2]           [Decomp 1.2]           [Query Approach 2]
                   [...]               [Decomp 2.1]           [...]
                                      [...]
```

## Component Breakdown

### 1. Repository Router
**Purpose**: Determine which data sources can provide information for each topic
- **Input**: Science Question + Topics (ordered)
- **Output**: Per-topic routes: for each topic, a list of `repositories` and aligned `rationales`
- **Behavior**: If a topic’s repositories include `CMR`, that topic is processed via CMR; otherwise a note is emitted indicating alternative repositories

### 2. Topic Splitting
**Purpose**: Identify distinct functional areas of inquiry within the science question
- **Input**: Science Question
- **Output**: List of Topics [1-n]
- **Decision Logic**: LLM-driven using guidelines; splits occur when LLM outputs multiple topics
- **Key Principle**: Topics must be functionally separate (requiring different datasets), not theoretically different
- **Example**: "Urbanization and UHI effect" → ["Urbanization", "UHI effect"]

### 3. Scientific Decomposition
**Purpose**: Break abstract topics into observable phenomena that can be measured
- **Input**: Original Science Question + Topic
- **Output**: List of [Decomposition, Scientific Justification] pairs
- **Decision Logic**: LLM-driven; branches when LLM outputs multiple decompositions
- **Key Feature**: LLM provides scientific reasoning for why each decomposition relates to the topic
- **Example**: "Urbanization" → ["Landcover", "Population Density", "Night Lights"]

### 4. Query Generation (Two-Phase)

#### Phase A: Known Parameters
**Purpose**: Extract hard filters that can be directly identified
- **Input**: Topic + Decomposition + Original Question
- **Output**: Query Approaches (complete parameter sets)
- **Parameters**: Instruments, spatial/temporal bounds, processing levels, resolutions
- **Example**: `{instrument: "Sentinel-2 MSI", temporal: "oct 2017 - nov 2018"}`

#### Phase B: Searchable Parameters
**Purpose**: Generate search terms and synonyms for dataset discovery
- **Input**: Science Question + Query Approach
- **Output**: Complete CMR query parameter sets (not literal API calls)
- **Parameters**: Keywords (abstract search), Science Keywords
- **Example**: Add `keyword: "land cover"` to complete the parameter set

### 5. Query Execution
**Purpose**: Execute CMR queries and retrieve collections
- **Implementation**: Handled by existing CMR wrapper code
- **Input**: Complete query parameter sets
- **Output**: Raw collection results with metadata

### 6. Collection Filtering & Ranking
**Purpose**: Filter and rank results based on relevance to original question
- **Input**: Original Query + Topic + Decomposition + Collections with metadata
- **Output**: Ranked, filtered collection list **per decomposition**
- **Scope**: Results merged and ranked only within each decomposition area
- **Filtering Criteria**: Spatial/temporal coverage, resolution requirements, processing level, abstract relevance

## Key Design Principles

### Branching & Parallelization
- Most components create opportunities for parallel processing
- All parallel paths eventually consolidate into ranked results
- Results maintain decomposition structure rather than random shuffling


## Data Flow & Context Preservation

### Global Context
- Original science question maintained throughout entire flow
- Downstream components consider full context, not just local inputs

### Local Context
- Each branch operates on specific topic/decomposition subset
- Scientific justifications link decompositions back to original question

## Final User Output Format

The user will see a structured result organized by:

1. **Topics**: Each identified topic from the original question
2. **Decompositions**: For each topic, the scientific decompositions with justifications
3. **Collections**: For each decomposition, a ranked and filtered list of relevant collections
4. **Alternative Sources**: For topics not resolvable via CMR, recommendations for other data sources (USGS, etc.)

**Example Structure**:
```
Topic: "Urbanization"
├── Decomposition: "Land cover" (Justification: Direct measure of urban expansion)
│   ├── Collection 1: Landsat 8 OLI Land Cover
│   └── Collection 2: MODIS Land Cover Type
├── Decomposition: "Night lights" (Justification: Proxy for urban activity intensity)
│   └── Collection 1: VIIRS DNB Nighttime Lights
└── [Additional decompositions...]

Topic: "Soil Analysis"
└── Note: "Data available through USGS Earth Resources Observation and Science Center"
```

## Example End-to-End Flow

**Science Question**: "Weekly land cover changes in Tennessee, Oct 2017 - Nov 2018"

1. **Repository Router**: Topic routed to CMR
2. **Topic Splitting**: ["Land cover changes"]
3. **Scientific Decomposition**: [("Land cover", "Direct observable for vegetation changes")]
4. **Known Parameters**:
   - Approach 1: `{instrument: "Sentinel-2 MSI", temporal: "oct 2017 - nov 2018", spatial: "Tennessee bounds"}`
   - Approach 2: `{instrument: "HLS", temporal: "oct 2017 - nov 2018", spatial: "Tennessee bounds"}`
5. **Searchable Parameters**: Add `{keyword: "landcover"}`
6. **Query Execution**: Execute CMR queries and retrieve collections
7. **Collection Filtering**: Filter by temporal resolution (weekly capability), rank by relevance

---

# Implementation Guide

This section explains how the theoretical workflow above is implemented in the codebase.

## Component Mapping

### 1. Topic Splitting → `TopicSplittingComponent`
- **Location**: `akd/agents/data_search/components/topic_splitting.py`
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/topic_splitting_system.md`
  - User: `components/prompts/topic_splitting_user.md`
- **Key Logic**: Guidelines ensure functional (not theoretical) topic separation

### 2. Repository Router → `RepositoryRouterComponent`
- **Location**: `akd/agents/data_search/components/repository_router.py`
- **Implementation**: LLM-based component with domain expertise
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/repository_routing_system.md`
  - User: `components/prompts/repository_routing_user.md`
  - Output shape: `{ routes: [{ repositories: [...], rationales: [...] }, ... ] }` aligned with input topics
- **Logic**: LLM analyzes topics using repository domain knowledge (CMR vs USGS/EPA/NOAA)
- **Current Scope**: Routes to CMR or provides detailed alternative source recommendations with reasoning

### 3. Scientific Decomposition → `ScientificDecompositionComponent`
- **Location**: `akd/agents/data_search/components/scientific_decomposition.py`
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/scientific_decomposition_system.md`
  - User: `components/prompts/scientific_decomposition_user.md`
- **Key Logic**: Decomposes topics into measurable observables with scientific justification

### 4. Known Parameters → `KnownParametersComponent`
- **Location**: `akd/agents/data_search/components/known_parameters.py`
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/known_parameters_system.md`
  - User: `components/prompts/known_parameters_user.md`
- **Key Logic**: Extracts hard filters (instruments, dates, spatial bounds) directly identifiable from context

### 5. Searchable Parameters → `SearchableParametersComponent`
- **Location**: `akd/agents/data_search/components/searchable_parameters.py`
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/searchable_parameters_system.md`
  - User: `components/prompts/searchable_parameters_user.md`
- **Key Logic**: Generates keywords and synonyms for metadata search

### 6. Collection Filtering & Ranking → `CollectionRankingComponent`
- **Location**: `akd/agents/data_search/components/collection_ranking.py`
- **LLM Integration**: Uses InstructorBaseAgent with structured output
- **Prompts**:
  - System: `components/prompts/collection_ranking_system.md`
  - User: `components/prompts/collection_ranking_user.md`
- **Key Logic**: Ranks collections by relevance to decomposition and research question

## Main Orchestration

### CMR Data Search Agent → `CMRDataSearchAgent`
- **Location**: `akd/agents/data_search/cmr_data_search.py`
- **Main Method**: `_arun()` implements the complete workflow
- **Key Methods**:
  - `_process_single_topic()`: Handles topic → decompositions → results
  - `_process_single_decomposition()`: Handles decomposition → queries → collections → granules
  - `_execute_searchable_queries()`: Converts SearchableQuery objects to CMR API calls
  - `_rank_collections()`: Uses CollectionRankingComponent for filtering

## Data Schemas

### Input/Output Structures
- **Topic**: `components/topic_splitting.py` - Functional topic with context
- **ScientificDecomposition**: `components/scientific_decomposition.py` - Observable with justification
- **QueryApproach**: `components/known_parameters.py` - Known parameter sets
- **SearchableQuery**: `components/searchable_parameters.py` - Complete query with keywords
- **TopicResult**: `_base.py` - Final result structure per topic
- **DecompositionResult**: `_base.py` - Final result structure per decomposition

## Query Execution Integration

### CMR Search Tools
- **Collection Search**: `akd/tools/data_search/cmr_collection_search.py`
- **Granule Search**: `akd/tools/data_search/cmr_granule_search.py`
- **Integration**: SearchableQuery objects converted to tool input parameters in `_execute_searchable_queries()`

## Configuration

### Model Assignment (Optimized)

- **Topic Splitting**: Uses `topic_splitting_model` (e.g., gpt-5-mini or gpt-4o) — requires broad reasoning for research question analysis
- **Repository Routing**: Uses `repository_routing_model` (e.g., gpt-5-mini or gpt-4o-mini) — domain-specific routing decisions
- **Scientific Decomposition**: Uses `scientific_decomposition_model` (e.g., gpt-5-mini or gpt-4o-mini) — focused domain expertise
- **Known Parameters**: Uses `cmr_query_model` (e.g., gpt-5-mini or gpt-4o-mini) — structured parameter extraction
- **Searchable Parameters**: Uses `cmr_query_model` (e.g., gpt-5-mini or gpt-4o-mini) — keyword generation
- **Collection Ranking**: Uses `collection_ranking_model` (e.g., gpt-5-mini or gpt-4o-mini) — focused ranking and evaluation

**Legacy Compatibility**: `angle_generation_model` parameter still supported for backward compatibility, maps to `topic_splitting_model`.

### Agent Configuration
- **Location**: `akd/agents/data_search/cmr_data_search.py` - `CMRDataSearchAgentConfig`
- **Key Settings**: Model assignments, timeouts, parallelization, collection limits

## Testing & Validation

### Demo Implementation
- **Location**: `examples/demo.py`
- **Individual Component Testing**: Command-line arguments support testing each component separately:
  - `--test topic-splitting` - Test topic splitting component
  - `--test repository-routing` - Test repository routing component
  - `--test scientific-decomposition` - Test scientific decomposition component
  - `--test known-parameters` - Test known parameters component
  - `--test searchable-parameters` - Test searchable parameters component
  - `--test collection-ranking` - Test collection ranking component
- **End-to-End Testing**: `test_new_workflow()` - Tests complete workflow with structured output
- **Output Format**: Matches the topic → decomposition → collections structure defined above
- **Dependency Handling**: Individual tests automatically create required dependencies (e.g., topics needed for decomposition testing)

## Error Handling & Reliability

### Robust Error Handling
All components implement proper error handling without quality-degrading fallbacks:
- **No Hardcoded Fallbacks**: Components fail cleanly rather than returning low-quality default data. If a critical step (e.g., collection ranking) fails, the workflow raises an error rather than substituting defaults.
- **Retry Logic**: Exponential backoff for rate limiting (3 retries with increasing delays)
- **Contextual Error Messages**: Detailed error information for debugging and user feedback
- **Graceful Failures**: RuntimeError with context when LLM calls consistently fail

### Rate Limiting Strategy
- **Base Delay**: 1 second initial delay for rate limit retries
- **Exponential Backoff**: 2^attempt multiplier for subsequent retries
- **Max Retries**: 3 attempts before permanent failure
- **Error Detection**: Identifies "429" status codes and "rate" keywords in error messages

## Prompt Engineering Notes

All prompts follow the workflow guidelines and include:
- **System Prompts**: Define component purpose, guidelines, and output requirements
- **User Prompts**: Provide context (original query, topic, decomposition) and specific instructions
- **Structured Outputs**: Use Pydantic models for consistent, validated responses
- **Scientific Focus**: Emphasize measurable observables and NASA/Earth science data availability
