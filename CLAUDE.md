# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv sync

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and configurations
```

### Running the Framework
The framework uses notebooks for examples and testing:
- `notebooks/` - Contains end-to-end examples
- Various `.ipynb` files in root - Testing and demonstration notebooks
- `scripts/run_lit_agent.py` - Literature agent runner script

### Testing
No formal test suite is currently configured. Testing is primarily done through:
- Example scripts in `examples/` directory
- Jupyter notebooks for interactive testing
- Individual test files like `scripts/test_journal_validator.py`

## Architecture Overview

### Core Philosophy
This is a **human-centric Multi-Agent System (MAS) framework** for scientific discovery built on these principles:
- **Human-in-the-loop control** - Researchers direct the discovery process
- **Scientific integrity** - Deep attribution, conflicting evidence identification, rigorous validation
- **Transparent & reproducible research** - Complete workflow tracking and shareability
- **Framework agnostic** - Core logic decoupled from orchestration engines

### Key Design Patterns

#### Planner-Orchestrator Pattern
- **Global Context**: Maintains overall research project state
- **Local Context**: Need-to-know subsets for individual agents
- **Specialized Agents**: Literature search, data extraction, relevancy checking, etc.
- **Multi-Agent RAG**: Literature, Data, Code search agents, Gap agents, conflict agents

#### NodeTemplate Architecture
All functional components MUST implement the `AbstractNodeTemplate` class (`akd/nodes/templates.py`):

```python
class NodeTemplate:
    - State: Well-defined, typed dictionary for node's internal memory
    - Input Guardrails: Validation functions executed on input state
    - Output Guardrails: Validation functions on node output
    - Node Supervisor: LLM agent or custom logic for internal orchestration
    - Tool Subset: Explicit list of permitted tools (principle of least privilege)
```

The core logic remains framework-agnostic and can be wrapped for different orchestration engines:
```python
from akd import NodeTemplate

node_t = NodeTemplate(
    supervisor=ReActLLMSupervisor(...)  # LLM-based or custom logic
    input_guardrails=[],  # Input validation rules
    output_guardrails=[],  # Output validation rules
)

node_lg = node.as_langgraph_node(...)
```

See `docs/node-template.md` for full details on node templates.

### Core Components

#### Base Classes (`akd/_base.py`)
- `AbstractBase`: Foundation for all agents and tools with schema validation
- `UnrestrictedAbstractBase`: Flexible version without strict schema enforcement
- `InputSchema`: Base for input schemas with required docstrings
- `OutputSchema`: Base for output schemas with required docstrings
- [Alternatively, `IOSchema`: Base for input/output schemas with required docstrings]

#### Agent System (`akd/agents/`)
- `BaseAgent`: Abstract base for all agents
- `LangBaseAgent`: LangChain-based agents with ChatOpenAI integration
- `InstructorBaseAgent`: Instructor-based agents for structured outputs
- Specialized agents: `akd.agents.extraction.EstimationExtractionAgent`, `akd.agents.query.QueryAgent`, `akd.agents.FollowUpQueryAgent`, `akd.agents.litsearch.py`

#### Node System (`akd/nodes/`)
- `templates.py`: `AbstractNodeTemplate` implementations for workflow components for node template
- `supervisor.py`: Various supervisor types (LLM, ReAct, Manual)
- `states.py`: State management (GlobalState, NodeTemplateState, SupervisorState)

#### Tool System (`akd/tools/`)
- `_base.py`: `BaseTool` foundation
- `scrapers/`: Web and PDF content extraction
- `search.py`: Search functionality (consists of `SearxNGSearchTool`, `SemanticScholarSearchTool`)
- `relevancy.py`: Content relevance checking
- `source_validator.py`: Source validation

#### Configuration (`akd/configs/`)
- `project.py`: Main configuration management
- `lit_config.py`: Literature search configuration
- `storm_config.py`: STORM workflow configuration
- `prompts.py`: System prompts and templates

### Key Data Structures (`akd/structures.py`)
- `SearchResultItem`: Search results with metadata
- `ResearchData`: Research dataset information
- `ExtractionSchema`: Information extraction base
- `SingleEstimation`: Research estimation extraction
- `ToolSearchResult`: Tool search and execution results

### State Management
- **GlobalState**: System-wide state containing all node states
- **NodeTemplateState**: Per-node state with supervisor state and guardrails
- **SupervisorState**: Tool calls, steps, and execution state

### Scientific Guardrails
The framework implements deep guardrails for scientific trust:
- **Deep Attribution**: Claims traceable to specific source sentences
- **Conflicting Literature Identification**: Actively surfaces contradictory evidence
- **Agentic RAG**: Gap Agent finds missing information, Conflict Agent finds contradictions
- **Quality Validation**: Prioritizes refereed journals and validated sources

### Golden Rules (from Design Philosophy)
1. **G-0**: Propose research plans but require human approval before execution
2. **G-1**: Use only vetted sources, flag non-refereed sources
3. **G-2**: Present conflicting evidence, avoid confirmation bias
4. **G-3**: Maintain sandboxed local contexts between agents
5. **G-4**: Require human approval for workflow modifications
6. **G-5**: Label all AI-generated content, require human validation
7. **G-6**: Maintain complete transparency and inspectability
8. **G-7**: Implement all components using standardized NodeTemplate

## Development Guidelines

### Adding New Components
1. **Always use BaseAgent or Base Tool** for functional agents and tools
    - Each tool/agent component has 4 parts to it for implementation:
        - Input Schema (inherited from `akd._base.InputSchema`)
        - Output Schema (inherited from `akd._base.OutputSChema)
        - Config schema (for tool `akd.tools._base.BaseToolConfig`, and for agent `akd.agents._base.BaseAgentConfig`)
        - Implementation of `_arun` method that takes in input schema and gives out output schema.
2. **Implement schema validation** with proper input/output schemas
3. **Maintain clear attribution** for all sources
4. **Design for transparency** - every step should be inspectable
5. **Enable reproducibility** through complete state capture

### Configuration Management
- Global configuration in `akd/configs/project.py`
- Agent-specific configs in respective files
- Environment variables for API keys and sensitive data
- TOML files for structured configuration (e.g., `config/lit_agent.toml`)

### Memory and State
- Use `GlobalState` for system-wide state management
- Implement proper state transitions through NodeTemplate
- Maintain message history in supervisor states
- Track tool calls and execution steps

### Integration Points
- **LangGraph**: Use `to_langgraph_node()` method for integration
- **LangChain**: Agents support LangChain tool binding
- **Instructor**: Structured output generation support
- **OpenAI**: Primary LLM integration

## Project Structure
```
akd/                    # Core framework package
├── agents/            # Specialized research agents
│   ├── extraction.py  # Data extraction agents
│   ├── litsearch.py   # Literature search agents
│   ├── query.py       # Query processing agents
│   └── relevancy.py   # Relevance checking agents
├── configs/           # Configuration management
├── nodes/             # NodeTemplate implementations
├── tools/             # Research tools and utilities
│   ├── scrapers/      # Content extraction tools
│   └── search.py      # Search functionality
└── structures.py      # Core data structures

examples/              # Usage examples
scripts/               # Utility scripts
docs/                  # Documentation
config/                # Configuration files
```

This framework prioritizes scientific integrity, human control, and reproducible research over automation convenience.
