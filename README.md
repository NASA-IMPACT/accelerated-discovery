# Accelerated Discovery Framework

A **human-centric MAS Framework** for scientific discovery that empowers researchers while maintaining scientific integrity, transparency, and reproducibility.

## Core Philosophy

This framework is built on the principle that **human researchers should direct the discovery process**. AI agents are powerful tools to augment, not replace, human intellect and intuition. We prioritize:

- **Human-in-the-loop control** - The researcher has final say on workflow, parameters, and interpretation
- **Scientific integrity** - Deep attribution, conflicting evidence identification, and rigorous validation
- **Transparent & reproducible research** - Every workflow is a shareable, inspectable artifact
- **Open collaboration** - Community-driven framework for shared scientific advancement

## Architecture

The system implements a **Planner-Orchestrator** pattern with standardized `NodeTemplate` components for maximum flexibility and scientific rigor:

### Core Design Patterns

- **NodeTemplate Architecture**: All functional components implement the standardized `AbstractNodeTemplate` (`akd.nodes.templates`) with:
  - Well-defined state management
  - Input/output guardrails for validation
  - Tool subset isolation (principle of least privilege)
  - Framework-agnostic design that can wrap into any orchestration engine

- **Context Management**:
  - **Global Context**: Maintains overall research project state
  - **Local Context**: Sandboxed, need-to-know subsets for individual agents
  - Prevents context bleeding between agents

- **Human-in-the-Loop Control**: Researchers maintain control over:
  - Workflow approval and modifications
  - Parameter tuning for any component
  - Branching and merging decisions

- **Multi-Agent Coordination**: Specialized agents work together with conflict detection and gap identification

### Human-in-the-Loop Control Points

The framework ensures researchers maintain control throughout the discovery process:

- **Plan Approval**: Initial research plans and any significant modifications require explicit human approval
- **Parameter Control**: Researchers can inspect and adjust parameters for any NodeTemplate component
- **Workflow Direction**: AI proposes next steps but humans direct the overall research strategy
- **Quality Gates**: Human validation required before accepting AI-generated analyses or conclusions
- **Branching Decisions**: Research workflow branching and merging decisions are human-directed

**Golden Rule**: When in doubt about research direction, data validity, or result interpretation, the system defers to human researcher guidance.

For comprehensive design principles, see [Design Philosophy](docs/design_philosophy.md).

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager (recommended)

### Installation

```bash
# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv sync

# For development (includes testing tools)
uv sync --extra dev

# For local development (includes marimo and other local tools)
uv sync --extra dev --extra local

# For ML development (includes pandas, sentence-transformers, docling, deepeval)
uv sync --extra ml

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and configurations

```

### Basic Usage

Refer to the [notebooks](notebooks) for examples.

## Workflow Planning System

The framework includes an LLM-based workflow planner that converts natural language research goals into executable workflows:

### Interactive Planning

```python
from akd.planner.llm_planner import create_planner

planner = await create_planner()
session = await planner.plan_workflow("Find papers on AlphaFold and identify research gaps")
response = await session.start()

while not response.ready_to_generate:
    user_input = input(f"{response.message}\nYour response: ")
    response = await session.respond(user_input)

workflow = await session.generate_workflow()
workflow.save_to_file("research_workflow.json")
```

### Automated Planning

```bash
# Interactive session
python scripts/demo_planner.py interactive

# Automated mode (CI/CD, batch processing)
python scripts/demo_planner.py automated "Research goal" --quiet -o workflow.json

# Quick planning
python scripts/demo_planner.py quick "Find papers on protein folding"
```

See [Planner Documentation](akd/planner/README.md) for comprehensive usage and deployment guides.

## Core Tools & Agents

### Search Infrastructure
- **Search Tools**: SearxNG (web), Semantic Scholar (academic), code repositories
- **Search Agents**: Deep search with iterative refinement, controlled search workflows, query processing and refinement
- **Relevancy Filtering**: Content assessment, link validation, context-aware filtering

### Content Extraction
- **Document Processing**: PDF extraction (PyPaperBot), advanced document parsing (Docling)
- **Web Scraping**: Multi-source content extraction with validation
- **Quality Control**: Source validation, credibility assessment, attribution tracking

## Scientific Guardrails

The framework implements deep guardrails specifically designed for scientific research integrity:

### Deep Attribution & Validation
- **Traceable Claims**: All claims traceable to specific source sentences and data points
- **Source Quality**: Prioritizes refereed journals and validated data repositories
- **Attribution Chain**: Complete attribution from final claims back to original sources
- **Quality Validation**: Multi-level validation of source credibility and relevance

### Conflict Detection & Gap Analysis
- **Agentic RAG**: Multi-agent approach to comprehensive information retrieval
  - **Gap Agent**: Actively identifies missing information and research gaps
  - **Conflict Agent**: Specifically searches for contradictory evidence and conflicting findings
- **Bias Prevention**: Deliberately surfaces contradictory evidence to prevent confirmation bias
- **Evidence Balance**: Ensures both supporting and conflicting evidence is presented

### Transparent Research Process
- **Stateful Execution**: Complete workflow state capture for reproducibility
- **Shareable Artifacts**: Research graphs that others can inspect, validate, and extend
- **Human Validation**: All AI-generated content clearly labeled and requires human approval
- **Complete Transparency**: Every step of the research process is inspectable and documented

## Key Features

- **Human-in-the-Loop Control**: Researchers direct the discovery process with AI augmentation
- **Framework Agnostic**: Core logic decoupled from orchestration engines for maximum flexibility
- **Reproducible Research**: Complete workflow capture enables true reproducibility and sharing
- **Community-Driven**: Open framework designed for collaborative scientific advancement

## Project Structure

```
akd/                    # Core framework
├── agents/            # Specialized research agents
├── planner/           # LLM workflow planner and builder
├── nodes/             # NodeTemplate implementations
├── tools/             # Research tools and scrapers
├── configs/           # Configuration and prompts
└── mapping/           # Field mapping definitions

examples/              # Usage examples
scripts/               # Utility scripts and demos
tests/                 # Comprehensive test suite
```

## Contributing

This is an open, community-driven framework. See our [design philosophy](docs/design_philosophy.md) for development and design guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
