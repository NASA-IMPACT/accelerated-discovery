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

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and configurations

```

### Basic Usage

Refer to the [notebooks](notebooks) for examples.

## Core Tools & Agents

The framework provides a comprehensive suite of specialized tools and agents for scientific research:

### Search Tools
- **SearxNGSearchTool** (`akd.tools.search.searxng_search`) - General web search with privacy focus
- **SemanticScholarSearchTool** (`akd.tools.search.semantic_scholar_search`) - Academic paper search and discovery
- **Code Search Tool** (`akd.tools.code_search`) - Code repository search and analysis

### Search Agents
- **Deep Search Agent** (`akd.agents.search.deep_search`) - Advanced multi-step literature search with iterative refinement
- **Controlled Search Agent** (`akd.agents.search.controlled`) - Structured search workflow with quality controls
- **Query Agent & FollowUp Query Agent** (`akd.agents.query`) - Query processing, refinement, and follow-up generation
- **Relevancy Agent** (`akd.agents.relevancy`) - Content relevance assessment and filtering

### Extraction & Processing
- **PyPaperBot Scraper** (`akd.tools.scrapers.pypaperbot`) - PDF content extraction and processing
- **DoclingScraper** (`akd.tools.scrapers`) - Advanced document processing and text extraction
- **Web Scrapers** (`akd.tools.scrapers.web_scrapers`) - Content extraction from web sources

### Specialized Components
- **Source Validator** (`akd.tools.source_validator`) - Source quality validation and verification
- **Relevancy Checker** (`akd.tools.relevancy`) - Content relevance validation against research context
- **Link Relevancy Assessor** (`akd.tools.link_relevancy_assessor`) - URL relevance assessment and filtering

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
