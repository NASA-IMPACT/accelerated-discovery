# Accelerated Discovery Framework

A **human-centric MAS Framework** for scientific discovery that empowers researchers while maintaining scientific integrity, transparency, and reproducibility.

## Core Philosophy

This framework is built on the principle that **human researchers should direct the discovery process**. AI agents are powerful tools to augment, not replace, human intellect and intuition. We prioritize:

- **Human-in-the-loop control** - The researcher has final say on workflow, parameters, and interpretation
- **Scientific integrity** - Deep attribution, conflicting evidence identification, and rigorous validation
- **Transparent & reproducible research** - Every workflow is a shareable, inspectable artifact
- **Open collaboration** - Community-driven framework for shared scientific advancement

## Architecture

The system uses a **Planner-Orchestrator** pattern with standardized `NodeTemplate` components:

- **Global Context**: Maintains overall research project state
- **Local Context**: Need-to-know subsets for individual agents
- **Specialized Agents**: Literature search, data extraction, relevancy checking, etc.
- **Multi-Agent RAG**: Literature, Data, Code search agents, Gap agents, conflict agents, and quality validators, Science guardrails, etc.
- **Framework Agnostic**: Core logic decoupled from orchestration and execution of individual agents and tools.

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

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and configurations

```

### Basic Usage

Refer to the [notebooks](notebooks) for examples.

### Granite Guardian Setup

For enhanced content validation and safety guardrails, see the [Granite Guardian Setup Guide](docs/granite-guardian-setup.md).

## Key Features


- **Deep Attribution**: claims traceable to specific source material, down to the sentences that were combined to make the claim (via factreasoner).
- **Conflict Detection**: Actively identifies contradictory evidence
- **Shareable Workflows**: Complete research (execution) graphs that others can inspect and extend, not just summarized end products.
- **Scientific Guardrails**: Guardrails that go beyond the generic LLM literature, that explicity designed to support scientific research.
- **Stateful Execution**: The workflow maintains a persistent, stateful context of the research journey, enabling a branching, reversible and iterative process that supports a researcher's natural methodology rather than imposing a rigid, automated sequence that current agentic systems embrace.

## Project Structure

```
akd/                    # Core framework
├── agents/            # Specialized research agents
├── nodes/             # NodeTemplate implementations  
├── tools/             # Research tools and scrapers
└── configs/           # Configuration management

examples/              # Usage examples
scripts/               # Utility scripts
```

## Contributing

This is an open, community-driven framework. See our [design philosophy](docs/design_philosophy.md) for development and design guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
