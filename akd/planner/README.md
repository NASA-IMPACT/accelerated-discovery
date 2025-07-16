# AKD Planner Module

The AKD Planner is an intelligent LLM-based research workflow planning system that automatically generates, validates, and executes multi-agent research workflows. It combines the power of reasoning LLMs with the AKD agent ecosystem to create efficient, scientifically rigorous research pipelines.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)  
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Development](#development)

## Overview

The AKD Planner provides intelligent workflow planning and execution for research tasks:

- **Intelligent Planning**: Uses LLMs to generate research workflows
- **Agent Discovery**: Automatically discovers and analyzes available research agents
- **Workflow Validation**: Validates plans for consistency, feasibility, and scientific rigor
- **Execution Orchestration**: Coordinates multi-agent workflow execution with LangGraph
- **Performance Optimization**: Implements caching, rate limiting, and parallel processing

### Key Features

- 🧠 **LLM-based Planning**: Leverages LLMs for intelligent workflow generation
- 🔄 **Multi-Agent Orchestration**: Coordinates diverse research agents seamlessly
- 📊 **Performance Monitoring**: Real-time metrics and health monitoring
- 🚀 **Scalable Architecture**: Designed for high-performance research workflows
- 📈 **Streaming Support**: Real-time execution updates and progress tracking
- 🔧 **Consolidated Structure**: Clean, organized codebase with eliminated redundancies

## Core Components

### 1. **PlannerOrchestrator** (`orchestrator.py`)

The main entry point providing a unified interface for all planner operations.

```python
from akd.planner import PlannerOrchestrator, PlannerRequest, PlannerConfig

# Create orchestrator
config = PlannerConfig(enable_caching=True)
orchestrator = PlannerOrchestrator(config)

# Generate and execute workflow
request = PlannerRequest(
    research_query="Find papers on quantum computing in drug discovery",
    execution_mode="plan_and_execute"
)

response = await orchestrator.plan_and_execute(request)
```

**Key Methods:**

- `plan_only()` - Generate workflow plan without execution
- `execute_plan()` - Execute an existing workflow plan
- `plan_and_execute()` - Generate and execute in one step
- `stream_execution()` - Stream real-time execution updates
- `health_check()` - Monitor system health
- `get_performance_metrics()` - Access performance data

### 2. **Core Services** (`core.py`)

#### PlannerServiceManager

Coordinates all planner services and manages shared resources.

```python
from akd.planner.core import PlannerServiceManager

service_manager = PlannerServiceManager(config)
metrics = service_manager.get_metrics()
```

#### LLMService

Centralized LLM management with rate limiting and retry logic.

```python
# Automatic rate limiting and retry
result = await service_manager.llm_service.invoke_with_retry(messages)

# Batch processing for parallel requests
results = await service_manager.llm_service.invoke_batch(batch_requests)
```

#### AgentRegistry

Manages agent discovery and caching with lazy loading.

```python
# Get available agents
agents = service_manager.agent_registry.get_all_agents()

# Register custom agent
service_manager.agent_registry.register_agent("CustomAgent", MyAgentClass)
```

#### ValidationService

Provides centralized validation logic for workflows.

```python
# Validate workflow structure
validation_result = service_manager.validation_service.validate_workflow_plan(plan)
```

### 3. **Research Planner** (`research_planner.py`)

High-level research planning with agent capability analysis.

```python
from akd.planner import ResearchPlanner, ResearchPlanningInput

planner = ResearchPlanner(config)

input_data = ResearchPlanningInput(
    research_query="Analyze COVID-19 treatment effectiveness",
    requirements={"focus_area": "clinical trials"},
    preferred_agents=["LiteratureSearchAgent", "DataExtractionAgent"]
)

result = await planner.execute_with_error_handling(input_data)
```

### 4. **Workflow Planner** (`workflow_planner.py`)

LLM-based workflow generation with reasoning traces.

```python
from akd.planner import WorkflowPlanner, WorkflowPlanningInput

planner = WorkflowPlanner(service_manager)

input_data = WorkflowPlanningInput(
    research_query="Find quantum computing applications",
    available_agents=agent_list,
    requirements={"max_duration": 300}
)

result = await planner.execute_with_error_handling(input_data)
```

### 5. **Agent Analyzer** (`agent_analyzer.py`)

Analyzes agent capabilities and compatibility for optimal workflow planning.

```python
from akd.planner import AgentAnalyzer

analyzer = AgentAnalyzer(config)

# Analyze agent capabilities
capabilities = await analyzer.analyze_agent_batch(agent_classes)

# Check compatibility between agents
compatibility = await analyzer.analyze_compatibility(agent1, agent2)
```

### 6. **LangGraph Builder** (`langgraph_builder.py`)

Builds and executes LangGraph workflows from research plans.

```python
from akd.planner import LangGraphWorkflowBuilder

builder = LangGraphWorkflowBuilder(config, service_manager)

# Build workflow
await builder.build_workflow(workflow_plan, agent_profiles)

# Execute workflow
result = await builder.execute_workflow(
    research_query="Find papers on AI",
    workflow_plan=plan,
    agent_profiles=profiles
)
```

### 7. **Supporting Components**

- **Agent Node Template** (`agent_node_template.py`): Wraps agents as NodeTemplate components
- **State Adapters** (`state_adapters.py`): Converts between different state representations  
- **Data Flow** (`data_flow.py`): Manages automatic data transformation between agents
- **LangGraph State** (`langgraph_state.py`): LangGraph-compatible state management

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PlannerOrchestrator                          │
│                   (Main Entry Point)                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                  PlannerServiceManager                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │ LLMService  │ │AgentRegistry│ │ValidationSvc│ │ Utilities│   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                   Planning Components                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ResearchPlanner  │ │WorkflowPlanner  │ │ AgentAnalyzer   │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Components                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │LangGraphBuilder │ │ StateAdapters   │ │   DataFlow      │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Planning Flow

```
Research Query → Agent Discovery → Capability Analysis → Workflow Planning → Validation → Execution
       ↓               ↓                ↓                  ↓             ↓           ↓
   Parse intent   Find agents    Analyze agents      Generate plan   Check plan   Execute nodes
   Extract reqs   Load schemas   Check compatibility  Create graph    Fix issues   Monitor progress
   Add context    Cache info     Score connections   Add reasoning   Validate     Stream updates
```

### Data Flow

```
Input Request → Planning State → Workflow Plan → Execution State → Results
     ↓              ↓              ↓               ↓                 ↓
PlannerRequest  WorkflowInput  WorkflowOutput  PlannerState   PlannerResponse
```

## Usage Guide

### Basic Usage

#### 1. Simple Planning

```python
from akd.planner import PlannerOrchestrator, PlannerRequest

# Create orchestrator
orchestrator = PlannerOrchestrator()

# Create request
request = PlannerRequest(
    research_query="Find recent papers on machine learning in healthcare",
    execution_mode="plan_only"
)

# Generate plan
response = await orchestrator.plan_only(request)

print(f"Generated plan with {len(response.workflow_plan['nodes'])} nodes")
print(f"Confidence: {response.confidence_score}")
print(f"Estimated duration: {response.estimated_duration}s")
```

#### 2. Plan and Execute

```python
request = PlannerRequest(
    research_query="Analyze effectiveness of COVID-19 treatments",
    requirements={
        "focus_area": "clinical trials",
        "publication_years": [2023, 2024],
        "max_papers": 50
    },
    execution_mode="plan_and_execute"
)

response = await orchestrator.plan_and_execute(request)

# Check execution results
if response.execution_result:
    print(f"Execution status: {response.execution_result['status']}")
    print(f"Final results: {response.execution_result['results']}")
```

#### 3. Streaming Execution

```python
# Execute with real-time updates
async for event in orchestrator.stream_execution(workflow_plan, session_id):
    print(f"Event: {event}")
    # Handle progress updates, node completion, errors, etc.
```

### Advanced Usage

#### 1. Custom Configuration

```python
from akd.planner import PlannerConfig

config = PlannerConfig(
    # LLM settings
    planning_model="o3-mini",
    enable_reasoning_traces=True,
    
    # Performance settings
    enable_caching=True,
    max_parallel_analysis=5,
    
    # Agent settings
    auto_discover_agents=True,
    agent_packages=["akd.agents", "my_custom_agents"],
    
    # Execution settings
    enable_checkpointing=True,
    checkpoint_storage="sqlite"
)

orchestrator = PlannerOrchestrator(config)
```

#### 2. Agent Analysis

```python
# Analyze available agents
agent_capabilities = await orchestrator.analyze_agents()

for capability in agent_capabilities:
    print(f"Agent: {capability.agent_name}")
    print(f"Domain: {capability.domain}")
    print(f"Capabilities: {capability.capabilities}")
    print(f"Confidence: {capability.confidence_score}")
```

#### 3. Workflow Validation

```python
# Validate workflow before execution
validation_result = await orchestrator.validate_workflow(workflow_plan)

if not validation_result['valid']:
    print("Validation errors:")
    for error in validation_result['errors']:
        print(f"  - {error}")
```

#### 4. Performance Monitoring

```python
# Get performance metrics
metrics = orchestrator.get_performance_metrics()

print(f"Plans generated: {metrics['plans_generated']}")
print(f"Cache hit rate: {metrics['cache_hits'] / metrics['llm_calls']:.2%}")
print(f"Error rate: {metrics['errors'] / metrics['orchestrator_operations']:.2%}")
```

#### 5. Health Monitoring

```python
# Check system health
health_status = await orchestrator.health_check()

print(f"Overall status: {health_status['orchestrator']}")
print("Component health:")
for component, status in health_status['components'].items():
    print(f"  {component}: {status}")
```

### Error Handling

```python
from akd.planner.core import PlannerError, PlanningError, ExecutionError

try:
    response = await orchestrator.plan_and_execute(request)
except PlanningError as e:
    print(f"Planning failed: {e}")
    # Handle planning-specific errors
except ExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle execution-specific errors
except PlannerError as e:
    print(f"General planner error: {e}")
    # Handle other planner errors
```

## Configuration

### Configuration Options

```python
class PlannerConfig:
    # Agent Discovery
    auto_discover_agents: bool = True
    agent_packages: List[str] = ["akd.agents", "akd.nodes"]
    
    # LLM Settings
    planning_model: str = "gpt-4o-mini"
    enable_reasoning_traces: bool = True
    reasoning_model_preference: List[str] = ["o3-mini", "o3", "o1-preview"]
    
    # Performance
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_parallel_analysis: int = 10
    
    # Execution
    execution_strategy: str = "langgraph"
    enable_checkpointing: bool = True
    checkpoint_storage: str = "memory"  # memory, sqlite, postgres
    
    # Monitoring
    enable_monitoring: bool = True
    enable_streaming: bool = True
    
    # Workflow Settings
    max_workflow_nodes: int = 20
    compatibility_threshold: float = 0.7
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Database (for checkpointing)
export POSTGRES_CHECKPOINT_URL="postgresql://user:pass@localhost/db"

# Performance
export AKD_PLANNER_CACHE_TTL=3600
export AKD_PLANNER_MAX_PARALLEL=5
```

## Examples

### Example 1: Literature Review Planning

```python
import asyncio
from akd.planner import PlannerOrchestrator, PlannerRequest

async def literature_review_example():
    orchestrator = PlannerOrchestrator()
    
    request = PlannerRequest(
        research_query="Conduct systematic review of quantum computing applications in drug discovery",
        requirements={
            "review_type": "systematic",
            "search_databases": ["PubMed", "arXiv", "IEEE"],
            "publication_years": [2020, 2024],
            "quality_threshold": 0.8
        },
        context={
            "constraints": {
                "max_papers": 100,
                "time_limit": 1800  # 30 minutes
            }
        }
    )
    
    response = await orchestrator.plan_and_execute(request)
    
    print(f"Literature review completed:")
    print(f"- Papers analyzed: {response.execution_result.get('papers_count', 0)}")
    print(f"- Confidence: {response.confidence_score:.2f}")
    print(f"- Duration: {response.estimated_duration}s")

asyncio.run(literature_review_example())
```

### Example 2: Data Extraction Pipeline

```python
async def data_extraction_example():
    orchestrator = PlannerOrchestrator()
    
    request = PlannerRequest(
        research_query="Extract treatment effectiveness data from clinical trial papers",
        requirements={
            "data_types": ["efficacy", "safety", "dosage"],
            "study_types": ["RCT", "meta-analysis"],
            "extraction_format": "structured"
        },
        preferred_agents=["LiteratureSearchAgent", "DataExtractionAgent", "ValidationAgent"]
    )
    
    # Stream execution for real-time updates
    async for event in orchestrator.stream_execution(
        (await orchestrator.plan_only(request)).workflow_plan,
        session_id="extraction_session"
    ):
        if event.get("event", {}).get("type") == "node_complete":
            node_id = event["event"]["node_id"]
            print(f"Completed node: {node_id}")

asyncio.run(data_extraction_example())
```

### Example 3: Custom Agent Integration

```python
from akd.agents._base import BaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema

class CustomResearchAgent(BaseAgent):
    class InputSchema(InputSchema):
        research_topic: str
        analysis_depth: str = "standard"
    
    class OutputSchema(OutputSchema):
        findings: List[str]
        confidence: float
    
    async def _arun(self, params: InputSchema) -> OutputSchema:
        # Custom research logic
        return self.OutputSchema(
            findings=["Custom finding 1", "Custom finding 2"],
            confidence=0.85
        )

# Register custom agent
orchestrator = PlannerOrchestrator()
orchestrator.register_agent("CustomResearchAgent", CustomResearchAgent)

# Use in planning
request = PlannerRequest(
    research_query="Analyze market trends using custom methodology",
    preferred_agents=["CustomResearchAgent"]
)

response = await orchestrator.plan_and_execute(request)
```

## API Reference

### Core Classes

#### PlannerOrchestrator

```python
class PlannerOrchestrator:
    def __init__(self, config: Optional[PlannerConfig] = None)
    
    async def plan_only(self, request: PlannerRequest) -> PlannerResponse
    async def execute_plan(self, workflow_plan: Dict, session_id: str) -> Dict
    async def plan_and_execute(self, request: PlannerRequest) -> PlannerResponse
    async def stream_execution(self, workflow_plan: Dict, session_id: str) -> AsyncIterator
    
    async def analyze_agents(self, agent_names: Optional[List[str]] = None) -> List[AgentCapability]
    async def validate_workflow(self, workflow_plan: Dict) -> Dict
    async def generate_alternatives(self, request: PlannerRequest, num_alternatives: int = 2) -> List[Dict]
    async def get_suggestions(self, workflow_plan: Dict) -> List[str]
    
    def get_performance_metrics(self) -> Dict
    def reset_metrics(self)
    def clear_caches(self)
    async def health_check(self) -> Dict
```

#### PlannerRequest

```python
class PlannerRequest:
    research_query: str
    requirements: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    preferred_agents: Optional[List[str]] = None
    session_id: Optional[str] = None
    execution_mode: str = "plan_and_execute"  # plan_only, execute_only, plan_and_execute
```

#### PlannerResponse

```python
class PlannerResponse:
    workflow_plan: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    reasoning_trace: str
    confidence_score: float
    estimated_duration: int
    available_agents: List[Dict[str, Any]]
    agent_capabilities: List[Dict[str, Any]]
    planning_metadata: Dict[str, Any]
    execution_ready: bool
    session_id: str
```

### Service Classes

#### PlannerServiceManager

```python
class PlannerServiceManager:
    def __init__(self, config: PlannerConfig)
    
    @property
    def llm_service: LLMService
    @property
    def agent_registry: AgentRegistry
    @property
    def validation_service: ValidationService
    
    def get_metrics(self) -> Dict[str, Any]
    def reset_metrics(self)
```

#### LLMService

```python
class LLMService:
    def __init__(self, config: PlannerConfig)
    
    async def invoke_with_retry(self, messages: List, max_retries: int = 3) -> Dict
    async def invoke_batch(self, batch_requests: List[List]) -> List[Dict]
```

#### AgentRegistry

```python
class AgentRegistry:
    def __init__(self, config: PlannerConfig)
    
    def register_agent(self, name: str, agent_class: Type[BaseAgent])
    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]
    def get_all_agents(self) -> Dict[str, Type[BaseAgent]]
    def get_agent_info(self, name: str) -> Dict[str, Any]
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd akd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running Tests

```bash
# Run all tests
pytest tests/planner/ -v
```

<!-- ### Adding New Components

1. **Create Component**: Follow the `BasePlannerComponent` pattern
2. **Add Tests**: Create comprehensive test coverage
3. **Update Documentation**: Update README and docstrings
4. **Add to **init**.py**: Export new components

```python
# Example new component
from akd.planner.core import BasePlannerComponent

class MyNewComponent(BasePlannerComponent):
    async def _execute(self, params):
        # Implementation
        return result
``` -->

<!-- ## Troubleshooting

### Common Issues

#### 1. Agent Discovery Failures

**Problem**: Agents not being discovered automatically

**Solution**:
```python
# Check agent packages configuration
config = PlannerConfig(
    agent_packages=["akd.agents", "your.custom.agents"]
)

# Manually register agents
orchestrator.register_agent("MyAgent", MyAgentClass)
```

#### 2. LLM Rate Limiting

**Problem**: API rate limit errors

**Solution**:
```python
# Reduce concurrent requests
config = PlannerConfig(max_parallel_analysis=3)

# Enable rate limiting
orchestrator = PlannerOrchestrator(config)
```

#### 3. Planning Failures

**Problem**: Workflow planning returns errors

**Solution**:
```python
# Check LLM service health
health = await orchestrator.health_check()
print(health['services']['llm'])

# Use fallback planning
try:
    response = await orchestrator.plan_only(request)
except PlanningError as e:
    print(f"Planning failed: {e}")
    # Check reasoning trace for details
```

#### 4. Performance Issues

**Problem**: Slow planning or execution

**Solution**:
```python
# Enable caching
config = PlannerConfig(enable_caching=True)

# Monitor performance
metrics = orchestrator.get_performance_metrics()
print(f"Cache hit rate: {metrics['cache_hits'] / metrics['llm_calls']:.2%}")

# Clear caches if needed
orchestrator.clear_caches()
```

### Debugging

#### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use loguru for better formatting
from loguru import logger
logger.add("planner_debug.log", level="DEBUG")
```

#### 2. Check System Health

```python
health_status = await orchestrator.health_check()
print(f"System health: {health_status}")
```

#### 3. Analyze Metrics

```python
metrics = orchestrator.get_performance_metrics()
print(f"Metrics: {metrics}")
```

#### 4. Validate Workflow

```python
validation = await orchestrator.validate_workflow(workflow_plan)
if not validation['valid']:
    print(f"Validation errors: {validation['errors']}")
```

### Performance Optimization

#### 1. Caching Strategy

```python
# Enable intelligent caching
config = PlannerConfig(
    enable_caching=True,
    cache_ttl_hours=24
)

# Monitor cache performance
metrics = orchestrator.get_performance_metrics()
cache_hit_rate = metrics['cache_hits'] / max(metrics['llm_calls'], 1)
```

#### 2. Parallel Processing

```python
# Optimize parallel execution
config = PlannerConfig(max_parallel_analysis=10)

# Use batch processing for alternatives
alternatives = await orchestrator.generate_alternatives(request, num_alternatives=5)
```

#### 3. Resource Management

```python
# Monitor resource usage
health = await orchestrator.health_check()

# Reset metrics periodically
orchestrator.reset_metrics()

# Clear caches when needed
orchestrator.clear_caches()
```

## Project Structure

### Planner Module Structure
```
akd/planner/           # Consolidated planner module
├── __init__.py        # Clean, organized exports
├── README.md          # Comprehensive documentation
├── core.py            # Core services and utilities
├── orchestrator.py    # Main entry point
├── research_planner.py # High-level research planning
├── workflow_planner.py # LLM-based workflow generation
├── agent_analyzer.py  # Agent capability analysis
├── langgraph_builder.py # LangGraph execution
├── langgraph_state.py # State management
├── agent_node_template.py # Agent-NodeTemplate wrapper
├── state_adapters.py  # State conversion utilities
└── data_flow.py       # Automatic data flow management
```

### Consolidation Benefits
- **Eliminated Redundancy**: Removed duplicate workflow planner implementations
- **Centralized Utilities**: Common functions consolidated in `core.py`
- **Clean Structure**: Logical organization with clear separation of concerns
- **Improved Maintainability**: Reduced code duplication and cleaner imports
- **Enhanced Performance**: Consolidated caching and service management

This framework prioritizes scientific integrity, human control, and reproducible research over automation convenience. -->