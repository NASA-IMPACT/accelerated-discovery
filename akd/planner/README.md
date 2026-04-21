# AKD Planner Module

LLM-based workflow planning system that converts natural language research goals into executable multi-agent workflows.

## Overview

The planner module provides:

- **Interactive workflow planning** via LLM conversation
- **Automatic agent selection** from registry
- **Field mapping** between agent inputs/outputs
- **Executable workflow generation** with runtime data flow (io_map)
- **Multi-layer validation** for correctness

**Components**:

- Agent Registry - Discovery and schema management
- LLM Planner - Interactive conversational planning
- Field Mapping - Three-tier mapping strategy with LLM generation
- Workflow Builder - Executable format generation with io_map
- Format Serialization - JSON save/load with validation

## Quick Start

```python
from akd.planner.llm_planner import quick_plan

# Quick planning (non-interactive)
session = await quick_plan("Find recent papers on AlphaFold protein structure prediction")
response = await session.start()
workflow = await session.generate_workflow()

# Save to file
workflow.save_to_file("my_workflow.json")
```

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    LLMWorkflowPlanner                        │
│  Orchestrates the planning process                           │
└──────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────┐
        ▼                  ▼                  ▼              ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│ AgentRegistry│  │FieldMapping     │  │FieldMapping  │  │ Workflow     │
│              │  │ Registry        │  │ Generator    │  │ Builder      │
│ - Loads      │  │                 │  │              │  │              │
│   agents     │  │ - Explicit      │  │ - LLM-based  │  │ - Builds     │
│ - Schemas    │  │   mappings      │  │   mapping    │  │   executable │
│ - Metadata   │  │ - LLM-approved  │  │ - Confidence │  │   format     │
│              │  │   mappings      │  │   scoring    │  │ - io_map     │
└──────────────┘  └─────────────────┘  └──────────────┘  └──────────────┘
```

### Data Structures

#### WorkflowPlan (High-Level Intent)

```python
WorkflowPlan(
    workflow_description="Literature search and gap analysis",
    research_goal="Find recent papers on AlphaFold",
    suggested_agents=[
        AgentSuggestion(
            agent_id="research_agent",
            agent_name="Deep Search Agent",
            reason="Search scientific literature",
            confidence=0.95,
            required_inputs=["query", "category"],
            expected_outputs=["results", "synthesis"],
            depends_on=None
        ),
        AgentSuggestion(
            agent_id="synthesis_agent",
            depends_on=["research_agent"]
        )
    ],
    workflow_steps=[
        "1. Deep Search finds papers",
        "2. Gap Analysis identifies gaps"
    ]
)
```

#### WorkflowFormat (Executable Specification)

```json
{
  "workflow_type": "AKDResearchWorkflow",
  "version": "1.0.0",
  "nodes": [
    {
      "type": "research_agent",
      "input": {
        "fields": [
          {"query": "recent papers on AlphaFold"},
          {"category": "Biochemistry"}
        ]
      },
      "output": {"fields": []},
      "io_map": null
    },
    {
      "type": "synthesis_agent",
      "input": {
        "fields": [
          {"gap": "research gaps in AlphaFold"}
        ]
      },
      "output": {"fields": []},
      "io_map": {
        "search_results": "$.research_agent.outputs.results"
      }
    }
  ],
  "edges": [
    {"from_node": "START", "to_node": "research_agent"},
    {"from_node": "research_agent", "to_node": "synthesis_agent"},
    {"from_node": "synthesis_agent", "to_node": "END"}
  ]
}
```

## End-to-End Workflow

### Step 1: User Query → Planner Initialization

```python
planner = await create_planner()
session = await planner.plan_workflow("Find recent papers on AlphaFold")
```

**Initialized Components:**

- `AgentRegistry` - Loads agents from `registry.json`
- `FieldMappingRegistry` - Loads explicit + LLM-generated mappings
- `FieldMappingGenerator` - LLM for generating new mappings
- `WorkflowBuilder` - Builds executable workflow format

### Step 2: Interactive Conversation (7 Phases)

The LLM guides the conversation through phases:

```
Phase 1: INITIAL_REQUIREMENTS
  └─ Analyze user query, determine if clarification needed

Phase 2: GOAL_CLARIFICATION
  └─ Ask clarifying questions to refine research goal

Phase 3: AGENT_SELECTION
  └─ Select appropriate agents from registry based on capabilities

Phase 4: IO_SPECIFICATION
  └─ Determine required inputs and expected outputs for each agent

Phase 5: WORKFLOW_CONSTRUCTION
  └─ Order agents in execution sequence, define workflow steps

Phase 6: VALIDATION
  └─ LLM reviews plan for completeness and correctness

Phase 7: FINALIZATION
  └─ Create final WorkflowPlan with all details
```

### Step 3: WorkflowPlan Creation

Pydantic model with validation:

```python
# ✓ Pydantic validates workflow_steps reference agents
@field_validator('workflow_steps')
def validate_steps_reference_agents(cls, steps, info):
    # Checks each step mentions at least one agent
    # Non-blocking: logs warnings only
```

### Step 4: Input Extraction

LLM extracts inputs for each agent using **full conversation context**:

```python
filled_inputs = {
    "research_agent": {
        "query": "AlphaFold accuracy improvements in structure prediction 2023-2025",  # Synthesized from conversation
        "category": "Biochemistry",  # LLM inferred from conversation
        "max_results": 50            # From user preference in conversation
    },
    "synthesis_agent": {
        "gap": "research gaps in AlphaFold accuracy improvements"  # Refined from conversation
    }
}
```

**Context used for extraction**:

- Initial user request
- Full conversation history (all refinements and clarifications)
- Research goal from workflow plan
- Workflow description
- Agent selection reasoning (why this agent was chosen)
- Agent field schemas and descriptions
- Agent dependencies and expected outputs

**Example conversation flow**:

```
User: "I want to study AlphaFold"
[... conversation refines the query ...]
User: "Focus on accuracy improvements in the last 2 years"
Planner: "Generating workflow..."

Input Extraction: Uses ENTIRE conversation to extract:
- query: "AlphaFold accuracy improvements 2023-2025"
- max_results: 50 (user mentioned wanting comprehensive results)
- category: "Biochemistry" (inferred from AlphaFold context)
```

**Fallback Strategy** (when LLM extraction fails):

1. **Schema defaults** - Use `field.default` from agent schema
2. **Generic patterns** - Fields named "query" get initial user request
3. **Type-based defaults** - `string→""`, `int→0`, `array→[]`

### Step 5: Pre-Build Validation

```python
# ✓ Check all agents exist in registry (BLOCKING)
missing = builder.check_missing_agents(plan)
if missing:
    raise ValueError(f"Agents not found: {missing}")

# ✓ Identify fields needing LLM mappings (NON-BLOCKING)
unmapped = builder.identify_unmapped_fields(plan, filled_inputs)
```

### Step 6: Build Executable Workflow

```python
workflow = builder.build(plan, filled_inputs)
```

**For each agent:**

1. Create `WorkflowNode` with inputs
2. Build `io_map` using three-tier strategy:

```
Priority 1: Registry mapping (explicit)
   └─ Manual mappings in field_mappings.json

Priority 2: Registry mapping (LLM-approved)
   └─ LLM-generated, confidence > threshold, user-approved

Priority 3: Exact name match
   └─ Field name exists in previous agent's output schema

Priority 4: No mapping found
   └─ Log warning, field will be missing
```

1. **Validate io_map fields**:

```python
# Get available source fields from schema
source_field_names = {f.name for f in prev_agent.output_schema.fields}

# Check mapping references actual field
if source_field not in source_field_names:
    if debug:
        raise ValueError(f"Invalid mapping: '{source_field}' not in {source_field_names}")
    else:
        logger.warning(f"Skipping invalid mapping")
        continue

# Create JSONPath expression
io_map[field.name] = f"$.{prev_agent_id}.outputs.{source_field}"
```

4. Create edges: `START → agent1 → agent2 → END`

### Step 7: WorkflowFormat Output

Executable JSON with runtime data flow:

```json
{
  "nodes": [
    {
      "type": "research_agent",
      "input": {"fields": [{"query": "..."}]},
      "io_map": null
    },
    {
      "type": "synthesis_agent",
      "input": {"fields": [{"gap": "..."}]},
      "io_map": {
        "search_results": "$.research_agent.outputs.results"
      }
    }
  ],
  "edges": [
    {"from_node": "START", "to_node": "research_agent"},
    {"from_node": "research_agent", "to_node": "synthesis_agent"},
    {"from_node": "synthesis_agent", "to_node": "END"}
  ]
}
```

### Step 8: Runtime Execution (Future Work)

Orchestrator (LangGraph/Custom) will:

1. Execute `research_agent` with `input.fields`
2. Store outputs in runtime state
3. For `synthesis_agent`:
   - Read `io_map`: `"$.research_agent.outputs.results"`
   - Resolve JSONPath from runtime state
   - Inject as `search_results` input
4. Execute `synthesis_agent`
5. Return final results

## Validation Layers

The planner implements **4 validation layers** at different stages:

### 1. Pydantic Field Validation (Compile-Time)

**Location**: `structures.py:36-71`
**When**: Automatic when `WorkflowPlan` is created
**Validates**: `workflow_steps` reference actual agents
**Behavior**: Non-blocking (warnings only)

```python
@field_validator('workflow_steps')
@classmethod
def validate_steps_reference_agents(cls, steps: list[str], info):
    """Ensures each step mentions at least one agent from suggested_agents."""
    agent_identifiers = {
        agent.agent_id.lower(),
        agent.agent_name.lower(),
        agent.agent_id.replace('_', ' ').lower()
    }

    for step in steps:
        if not any(agent_id in step.lower() for agent_id in agent_identifiers):
            logger.warning(f"Step '{step}' doesn't reference any agent")

    return steps
```

### 2. Agent Existence Validation (Pre-Build)

**Location**: `workflow_builder.py:222-233`
**When**: Before building workflow
**Validates**: All agents in plan exist in registry
**Behavior**: Blocking (raises `ValueError`)

```python
def check_missing_agents(self, plan: WorkflowPlan) -> list[str]:
    """Check if all agents exist in registry."""
    missing = []
    for agent_suggestion in plan.suggested_agents:
        if not self.registry.get_agent(agent_suggestion.agent_id):
            missing.append(agent_suggestion.agent_id)
    return missing
```

### 3. io_map Field Validation (Build-Time)

**Location**: `workflow_builder.py:107-122`
**When**: During `build()` when creating runtime data flow
**Validates**: Mapped fields exist in source agent's output schema
**Behavior**:

- **Debug mode**: Blocking (raises `ValueError`)
- **Production mode**: Non-blocking (skips invalid mapping, logs error)

```python
# Get available source fields from schema
source_field_names = {f.name for f in prev_agent.output_schema.fields}

# Validate mapping references actual field
if source_field not in source_field_names:
    logger.error(f"Invalid mapping: {source_field} not in {prev_agent_id}.outputs")

    if self.debug:
        raise ValueError(
            f"Mapping references non-existent field: '{source_field}'. "
            f"Available: {source_field_names}"
        )
    else:
        logger.warning(f"Skipping invalid mapping")
        continue
```

### 4. Unmapped Field Detection (Build-Time)

**Location**: `workflow_builder.py:235-288`
**When**: During `build()` to identify missing mappings
**Validates**: All required fields have data sources
**Behavior**: Non-blocking (returns list for LLM generation)

```python
def identify_unmapped_fields(self, plan, filled_inputs) -> list[UnmappedFieldInfo]:
    """
    Identify fields that need LLM-based mapping.
    Returns list of unmapped fields for LLM generation.
    """
```

## Validation Summary Table

| Stage | Validation | Blocking? | Location |
|-------|-----------|-----------|----------|
| Plan Creation | workflow_steps reference agents | ⚠️ No (warnings) | structures.py:36-71 |
| Pre-Build | Agent existence | Yes | workflow_builder.py:222-233 |
| Build | io_map field existence | ⚠️ Debug only | workflow_builder.py:107-122 |
| Build | Unmapped field detection | ⚠️ No (warnings) | workflow_builder.py:235-288 |

## Configuration

### PlannerConfig

All planner behavior is configurable via `PlannerConfig`:

```python
from akd.planner.structures import PlannerConfig

config = PlannerConfig(
    model_name="gpt-4",                              # LLM model for planning
    temperature=0.3,                                  # Deterministic planning
    max_conversation_turns=10,                        # Max interactive turns
    field_mapping_confidence_threshold=0.8,           # Auto-approve threshold
    input_extraction_temperature=0.1                  # Very deterministic extraction
)

planner = LLMWorkflowPlanner(planner_config=config)
```

### WorkflowBuilder Debug Mode

```python
# Debug mode: Fail fast on invalid mappings
builder = WorkflowBuilder(debug=True)

# Production mode: Skip invalid mappings, log errors
builder = WorkflowBuilder(debug=False)
```

## Field Mapping System

### Three-Tier Strategy

The planner uses a three-tier strategy to map data between agents:

```python
def _build_field_mappings(agent, prev_agent, agent_id, prev_agent_id):
    """
    Priority 1: Registry mapping (explicit from field_mappings.json)
    Priority 2: Registry mapping (LLM-generated, approved)
    Priority 3: Exact name match
    Priority 4: No mapping - log warning
    """
```

### Field Mapping Registry

**Explicit mappings** (`akd/mapping/field_mappings.json`):

```json
{
  "research_agent->synthesis_agent": {
    "search_results": "results"
  }
}
```

**LLM-generated mappings** (`akd/mapping/llm_generated_mappings.json`):

```json
{
  "version": "1.0.0",
  "mappings": {
    "research_agent->synthesis_agent": {
      "mapping": {"search_results": "results"},
      "confidence": 0.95,
      "user_approved": true,
      "reasoning": {"search_results": "Semantic match based on descriptions"}
    }
  }
}
```

### Auto-Approval Threshold

LLM-generated mappings with confidence ≥ `field_mapping_confidence_threshold` are auto-approved:

```python
if result.overall_confidence >= confidence_threshold:
    # Auto-approve high confidence mappings
    approved_mapping = result.mappings
else:
    # Low confidence - require user approval (future PR)
    logger.warning(f"Low confidence mapping: {result.overall_confidence}")
```

## Usage Examples

### Interactive Planning Session

```python
from akd.planner.llm_planner import create_planner

# Create planner
planner = await create_planner()

# Start interactive session with initial request
initial_request = "I want to analyze the current state of the art in drug discovery"
session = await planner.plan_workflow(initial_request)

# Start the conversation
response = await session.start()

# Conversation loop
while not response.ready_to_generate:
    print(f"Planner: {response.message}")

    if response.question:
        print(f"Question: {response.question.question}")
        user_input = input("Your response: ")
    else:
        user_input = input("Continue: ")

    response = await session.respond(user_input)

# Generate workflow
workflow = await session.generate_workflow()
workflow.save_to_file("drug_discovery_workflow.json")
```

### Quick Planning (Non-Interactive)

```python
from akd.planner.llm_planner import quick_plan

# Quick plan with single query
session = await quick_plan("Find papers on AlphaFold and identify research gaps")
response = await session.start()

if response.workflow_plan:
    workflow = await session.generate_workflow()
    print(workflow.to_json())
```

### Custom Configuration

```python
from akd.planner.llm_planner import LLMWorkflowPlanner
from akd.planner.structures import PlannerConfig
from akd.planner.registry import get_agent_registry

# Custom planner config
planner_config = PlannerConfig(
    model_name="gpt-4",
    temperature=0.1,  # Very deterministic
    field_mapping_confidence_threshold=0.9  # High confidence required
)

# Custom registry (selective agents)
registry = get_agent_registry(config=AgentRegistryConfig(
    use_agents=["research_agent", "synthesis_agent"]
))

# Create planner with custom config
planner = LLMWorkflowPlanner(
    planner_config=planner_config,
    registry=registry,
    debug=True  # Fail fast on errors
)

session = await planner.plan_workflow("My research goal")
```

### CLI Demo

```bash
# Interactive mode
python scripts/demo_planner.py interactive

# Quick mode
python scripts/demo_planner.py quick "Find papers on protein folding"

# Quick mode with output file
python scripts/demo_planner.py quick "Find papers on AlphaFold" -o workflow.json

# Non-interactive (for CI/CD)
python scripts/demo_planner.py quick "Research goal" --non-interactive -o out.json

# Show available agents
python scripts/demo_planner.py agents

# Show sample workflow format
python scripts/demo_planner.py sample
```

## Testing

Run all planner tests:

```bash
pytest tests/planner/ -v -o addopts=""
```

Test specific modules:

```bash
# LLM planner
pytest tests/planner/test_llm_planner.py -v

# Field mapping
pytest tests/planner/test_field_mapping.py -v

# Registry
pytest tests/planner/test_registry.py -v

# Workflow builder
pytest tests/planner/test_workflow_builder.py -v
```

## Module Files

```
akd/planner/
├── __init__.py                    # Public API exports
├── config.py                      # Configuration models
├── structures.py                  # Data structures (WorkflowPlan, PlannerConfig)
├── registry.py                    # Agent registry and discovery
├── workflow_builder.py            # Builds executable WorkflowFormat
├── format_builder.py              # WorkflowFormat model and serialization
├── field_mapping_registry.py      # Field mapping storage
├── field_mapping_generator.py     # LLM-based field mapping
├── llm_planner.py                 # Main LLM planner implementation
└── README.md                      # This file

tests/planner/
├── test_llm_planner.py            # LLM planner tests
├── test_field_mapping.py          # Field mapping tests
├── test_registry.py               # Registry tests
└── test_workflow_builder.py       # Builder tests

scripts/
└── demo_planner.py                # CLI demo

akd/mapping/
├── field_mappings.json            # Explicit field mappings
└── llm_generated_mappings.json    # LLM-generated mappings with approval
```

## Future Enhancements

1. **Runtime execution** - Orchestrator to execute WorkflowFormat (LangGraph integration)
2. **User approval flow** - Interactive approval for low-confidence mappings
3. **Parallel agent execution** - Support for non-sequential workflows
4. **Conditional branching** - Agent selection based on runtime conditions
5. **Workflow validation** - Pre-execution validation of complete workflow
6. **Workflow templates** - Reusable workflow patterns
7. **Agent suggestions** - LLM suggests new agents based on gaps

## Related Documentation

- [Planner Roadmap](../../docs/planner-roadmap-updated.md)
- [PR #2: Builder + Field Mapping](../../docs/pr_02_builder_field_mapping.md)
- [Field Mapping Design](../../FIELD_MAPPING_DESIGN.md)
- [Node Template Architecture](../../docs/node-template.md)
