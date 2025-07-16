# AKD Mapping System

## Overview

The AKD mapping system enables **seamless data transformation between heterogeneous agent schemas** in multi-agent workflows. It provides type-safe, intelligent mapping with graceful degradation, making it the critical element that allows different agents to work together in type-safe multi-agent workflows.

## The Problem: Heterogeneous Multi-Agent Communication

In the AKD framework, different agents may have incompatible input/output schemas:

```python
# QueryAgent output
class QueryAgentOutput(OutputSchema):
    queries: List[str] = Field(description="Generated search queries")
    category: str = Field(description="Query category")

# LiteratureSearchAgent input  
class LitAgentInput(InputSchema):
    query: str = Field(description="Single search query")
    max_results: int = Field(default=10, description="Maximum results")
```

**Without mapping**, these agents cannot communicate directly in a workflow.

## Architecture: LangGraph Integration

The mapper enables runtime data transformation in LangGraph workflows:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Workflow State                         │
│                                                                         │
│  Initial Query → PlannerState → Human Approval → Agent Execution        │
│                       ↓                                                 │
│           ┌─────────────────────────────────────────────┐               │
│           │            Node Results Storage             │               │
│           │  {                                          │               │
│           │    "query_1": QueryAgentOutput,             │               │
│           │    "lit_1": LitAgentOutput,                 │               │
│           │    "extract_1": ExtractionOutput            │               │
│           │  }                                          │               │
│           └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↑
                                    │
                      Runtime Data Transformation
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       Agent Node Execution                              │
│                                                                         │
│  1. Get previous output: state.node_results["query_1"]                  │
│  2. Map to current input: QueryOutput → LitInput                        │
│  3. Execute current agent: LitAgent.arun(mapped_input)                  │
│  4. Store result: state.node_results["lit_1"] = output                  │
│                                                                         │
│           ┌─────────────────────────────────────────────┐               │
│           │          WaterfallMapper                    │               │
│           │                                             │               │
│           │  Direct → Semantic → LLM → Fallback         │               │
│           │  Match     Match      AI     String         │               │
│           │                                             │               │
│           └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Waterfall Mapping Strategy

The system uses a **progressive waterfall approach** with four stages:

### 1. Direct Field Matching

```python
# Perfect field name matches
source = QueryOutput(queries=["solar energy"], category="science")
target_schema = QueryInput  # has 'query' field

# Direct mapping fails (queries ≠ query), moves to next stage
```

### 2. Semantic Field Matching  

```python
# Fuzzy semantic matching with domain knowledge
semantic_groups = {
    "queries": ["query", "search", "question", "term"],
    "documents": ["results", "data", "items", "papers"],
    "scores": ["confidence", "relevance", "rating"]
}

# "queries" → "query" (semantic similarity: 0.85)
```

### 3. Structure-Based Matching

```python
# Type and naming pattern matching
# List[str] → str (extract first item)
# "search_query" → "query" (substring match)
```

### 4. LLM Fallback

```python
# AI-powered intelligent parsing
llm_prompt = """
Transform this data to match the target schema:
SOURCE: {"queries": ["solar energy research"], "category": "science"}
TARGET: QueryInput with fields: query (str), context (str)
"""
# Result: {"query": "solar energy research", "context": "science"}
```

<!-- ## Key Features

### Type-Safe Design

- **Input**: Pydantic model instances (not raw dicts)
- **Output**: Validated Pydantic model instances  
- **Schema Introspection**: Full access to field metadata
- **AKDSerializer Integration**: Proper model conversions

### Intelligent Mapping

- **Confidence Scoring**: Each mapping gets a confidence score (0.0-1.0)
- **Circuit Breaker**: Failed strategies are temporarily disabled
- **Caching**: Performance optimization for repeated mappings
- **Graceful Degradation**: Always produces valid output

### Human-in-the-Loop Ready

- **Low Confidence Alerts**: Request human approval for uncertain mappings
- **Mapping Transparency**: Full visibility into transformation decisions
- **Interactive Hints**: Humans can provide mapping hints -->

## Usage Examples

### Basic Agent-to-Agent Mapping

```python
from akd.mapping.mappers import WaterfallMapper, MappingInput
from akd.agents.query import QueryAgentOutputSchema
from akd.agents.litsearch import LitAgentInputSchema

# Initialize mapper
mapper = WaterfallMapper()

# Previous agent output
query_output = QueryAgentOutputSchema(
    queries=["carbon capture materials", "direct air capture"],
    category="materials_science"
)

# Map to next agent input
result = await mapper.arun(MappingInput(
    source_model=query_output,
    target_schema=LitAgentInputSchema,
    mapping_hints={"queries": "query"}  # Use first query
))

# Use mapped result
lit_agent = LiteratureSearchAgent()
lit_result = await lit_agent.arun(result.mapped_model)
```

<!-- ### LangGraph Workflow Integration

```python
async def query_to_literature_node(state: PlannerState) -> PlannerState:
    """LangGraph node that transforms QueryAgent output to LitAgent input"""
    
    # Get previous node output
    query_output_data = state.node_results["query_node"]
    query_output = QueryAgentOutputSchema(**query_output_data)
    
    # Transform to literature agent input
    mapping_result = await mapper.arun(MappingInput(
        source_model=query_output,
        target_schema=LitAgentInputSchema
    ))
    
    # Check mapping confidence
    if mapping_result.mapping_confidence < 0.7:
        # Request human approval for low-confidence mapping
        state.request_human_approval("mapping_approval", {
            "source_schema": "QueryAgentOutputSchema",
            "target_schema": "LitAgentInputSchema", 
            "confidence": mapping_result.mapping_confidence,
            "unmapped_fields": mapping_result.unmapped_fields
        })
        return state
    
    # Execute literature agent
    lit_agent = LiteratureSearchAgent()
    lit_output = await lit_agent.arun(mapping_result.mapped_model)
    
    # Store result for next node
    state.update_node_result("lit_node", lit_output.model_dump())
    
    return state
``` -->

### Configuration and Customization

```python
from akd.mapping.mappers import MappingConfig

# Advanced configuration
config = MappingConfig(
    # Strategy controls
    enable_direct_matching=True,
    enable_semantic_matching=True,
    enable_llm_fallback=True,
    
    # Quality thresholds
    semantic_threshold=0.7,          # Minimum semantic similarity
    circuit_breaker_threshold=5,    # Failures before disabling strategy
    
    # Performance settings
    enable_caching=True,
    max_retries=2,
    
    # LLM settings
    llm_model="gpt-4o-mini"
)

mapper = WaterfallMapper(config=config)
```

### Error Handling and Fallbacks

```python
try:
    result = await mapper.arun(MappingInput(
        source_model=complex_output,
        target_schema=TargetSchema
    ))
    
    if result.mapping_confidence < 0.5:
        print(f"Low confidence mapping: {result.mapping_confidence}")
        print(f"Strategy used: {result.used_strategy}")
        print(f"Unmapped fields: {result.unmapped_fields}")
    
except Exception as e:
    print(f"Mapping failed: {e}")
    # System provides minimal fallback
```

## AKD Agent Examples

### Literature Search Pipeline

```python
# Step 1: Query → Literature Search  
query_output = QueryAgentOutputSchema(queries=["perovskite solar cells"])
lit_input = await map_schemas(query_output, LitAgentInputSchema)

# Step 2: Literature → Extraction
lit_output = LitAgentOutputSchema(results=[...])  
extract_input = await map_schemas(lit_output, ExtractionInputSchema)

# Step 3: Extraction → Relevancy
extract_output = ExtractionOutputSchema(extractions=[...])
relevancy_input = await map_schemas(extract_output, RelevancyInputSchema)
```

## Testing

### Comprehensive Test Suite

```bash
# Run all mapping tests
pytest tests/mapping/ -v
```

## Performance Considerations

### Caching Strategy

```python
# Automatic caching based on source/target schema combination
cache_key = hash(source_schema_name + target_schema_name + field_names)
```

### Circuit Breaker Pattern

```python
# Strategies are disabled after repeated failures
if strategy_failures["semantic"] > circuit_breaker_threshold:
    skip_semantic_matching = True
```

### Parallel Processing

```python
# Semantic matching uses async parallel processing
tasks = [match_field(field) for field in target_fields]
results = await asyncio.gather(*tasks)
```

## Integration with `akd.agents`

```python
# Your existing agent
class MyCustomAgent(BaseAgent):
    input_schema = MyInputSchema
    output_schema = MyOutputSchema
    
    async def _arun(self, params: MyInputSchema) -> MyOutputSchema:
        # Your agent logic
        return MyOutputSchema(...)

# Automatically works with mapper
mapper_result = await mapper.arun(MappingInput(
    source_model=other_agent_output,
    target_schema=MyInputSchema
))

my_agent = MyCustomAgent()
my_result = await my_agent.arun(mapper_result.mapped_model)
```

## Advanced Features

### Custom Semantic Groups

```python
config = MappingConfig()
mapper = WaterfallMapper(config=config)

# Add domain-specific semantic groups
mapper.semantic_mapper.semantic_groups["result"] = [
    "results", "data", "items", "papers"
]
```

### Mapping Hints for Complex Transformations

```python
result = await mapper.arun(MappingInput(
    source_model=complex_output,
    target_schema=TargetSchema,
    mapping_hints={
        "source_list_field": "target_string_field",  # Take first item
        "nested.source.field": "flat_target_field",  # Flatten structure
        "computed_field": "static_value"             # Use static value
    }
))
```

### Human Approval Integration

```python
async def mapping_with_approval(source_model, target_schema):
    result = await mapper.arun(MappingInput(
        source_model=source_model,
        target_schema=target_schema
    ))
    
    if result.mapping_confidence < 0.7:
        # In LangGraph workflow, this triggers human intervention
        approval = await request_human_approval({
            "mapping_confidence": result.mapping_confidence,
            "unmapped_fields": result.unmapped_fields,
            "suggested_hints": generate_mapping_suggestions(result)
        })
        
        if approval.provide_hints:
            # Retry with human-provided hints
            result = await mapper.arun(MappingInput(
                source_model=source_model,
                target_schema=target_schema,
                mapping_hints=approval.mapping_hints
            ))
    
    return result
```
<!-- 
## Contributing

### Adding New Mapping Strategies

1. Inherit from `BaseMappingStrategy`
2. Implement `map_models()` method
3. Add to `WaterfallMapper` strategy list
4. Write comprehensive tests

```python
class CustomMappingStrategy(BaseMappingStrategy):
    async def map_models(
        self, 
        source_model: BaseModel,
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        # Your custom mapping logic
        return {
            "mapped": mapped_data,
            "confidence": confidence_score,
            "unmapped": unmapped_fields,
            "metadata": {"strategy": "custom"}
        }
``` -->

<!-- ## Troubleshooting

### Common Issues

**Low Mapping Confidence**

- Check field name similarity
- Provide explicit mapping hints
- Verify schema documentation quality

**LLM Fallback Not Working**

- Ensure OpenAI API key is configured
- Check network connectivity
- Verify model availability

**Performance Issues**

- Enable caching for repeated mappings
- Reduce semantic threshold for faster matching
- Disable LLM fallback for speed-critical paths

### Debug Mode

```python
config = MappingConfig(debug=True)
mapper = WaterfallMapper(config=config)

# Detailed logging of mapping decisions
result = await mapper.arun(mapping_input)
``` -->

---

**The AKD mapping system makes heterogeneous multi-agent workflows possible by providing intelligent, reliable, and transparent data transformation between any two agent schemas.**
