# Accelerated Knowledge Discovery Core (akd-core)

A human-centric multi-agent system (MAS) framework for scientific discovery — providing base classes, streaming, human-in-the-loop (HITL), guardrails, and out-of-box agents and tools.

## Ecosystem

akd-core is the foundation layer that drives the entire AKD ecosystem:

- **akd-core** (this repo) — base classes, streaming infrastructure, HITL, guardrails, and out-of-box agents/tools for scientific discovery
- **[akd-framework](https://github.com/NASA-IMPACT/akd-framework/)** — the AKD backend application, built on akd-core
- **[akd-ext](https://github.com/NASA-IMPACT/akd-ext)** — community extensions that use akd-core's base agents and tools to build domain-specific capabilities

akd-core is standalone and pip-installable. Everything downstream inherits its streaming, HITL, and guardrail infrastructure.

## Core Philosophy

- **Human-in-the-loop control** — researchers direct the discovery process; AI augments, never replaces
- **Scientific integrity** — deep attribution, evidence validation, and rigorous guardrails
- **Transparent and reproducible** — every workflow is a shareable, inspectable artifact
- **Open collaboration** — community-driven framework for shared scientific advancement

See [Design Philosophy](docs/design_philosophy.md) for the full set of principles and golden rules.

## What You Get

- **Async-first** — all agents and tools implement `async def _arun()` with full `astream()` support
- **Streaming-native** — 11 typed event types covering tokens, reasoning, tool calls, and HITL
- **Type-safe** — Pydantic v2 schemas with required docstrings for all inputs and outputs
- **Composable** — tools combine via Composite patterns (search, resolvers, guardrails)
- **HITL built-in** — pause, save state, get human input, resume seamlessly
- **Guardrails** — pluggable safety layer with decorator API
- **LLM-agnostic** — works with any provider via LiteLLM (OpenAI, Anthropic, Ollama, etc.)

```python
from akd.agents import BaseAgent

agent = BaseAgent(config={"model_name": "gpt-4o-mini"})
async for event in agent.astream(input_params):
    match event.event_type:
        case "streaming": print(event.token, end="")
        case "tool_calling": print(f"Calling {event.tool_name}...")
        case "human_input_required": response = input(event.human_prompt)
        case "completed": result = event.output
```

## Streaming

Everything in akd-core is a stream of typed events. Agents emit `StreamEvent` objects as they execute:

| Event | Description |
|-------|-------------|
| `STARTING` | Agent/tool begins execution |
| `RUNNING` | Progress update |
| `STREAMING` | Raw LLM tokens as they arrive |
| `THINKING` | Reasoning tokens (Claude extended thinking, o1/o3) |
| `PARTIAL` | Partial structured output as it streams |
| `TOOL_CALLING` | Agent invokes a tool |
| `TOOL_RESULT` | Tool returns its result |
| `HUMAN_INPUT_REQUIRED` | Agent needs human input — execution pauses |
| `HUMAN_RESPONSE` | Resumed with human input |
| `COMPLETED` | Execution finished successfully |
| `FAILED` | Execution failed with error details |

Each event carries typed data (e.g., `CompletedEventData[T]` includes the output, `FailedEventData` includes the error) and a `run_context` for execution state.

## Human-in-the-Loop

HITL is a first-class concept, not an afterthought. The `HumanTool` enables any agent to pause execution, request human input, and resume:

1. Agent calls `HumanTool` during its tool loop
2. Framework emits `HUMAN_INPUT_REQUIRED` event with the question and full message history
3. Caller saves state and collects human response
4. Resume with `RunContext(messages=saved_history, human_response=HumanResponse(...))`
5. Agent continues exactly where it left off

This works across any transport — REST APIs, WebSockets, CLI — because the pause/resume is state-based, not connection-based.

## Out-of-Box Agents

| Category | Agent | Description |
|----------|-------|-------------|
| **Research** | `DeepLitSearchAgent` | Multi-agent deep literature search with triage, clarification, and synthesis |
| | `ControlledSearchAgent` | Controlled search with configurable parameters |
| | `AspectSearchAgent` | Interview-pattern multi-aspect search |
| | `CodeSearchAgent` | Code repository search |
| | `QuestionAnsweringAgent` | QA over retrieved content |
| **Analysis** | `GapAgent` | Research gap identification via knowledge graphs |
| | `EstimationExtractionAgent` | Intent-based data extraction |
| | `StormAgent` | Structured narrative generation |
| **Utility** | `IntentAgent` | User intent classification |
| | `QueryAgent` | Query reformulation and refinement |
| | `FollowUpQueryAgent` | Follow-up query generation |
| | `RelevancyAgent` | Binary relevance classification |
| | `MultiRubricRelevancyAgent` | Multi-dimensional relevance scoring |
| **Base** | `BaseAgent` | Core agent with streaming, tool calling, HITL, message trimming |
| | `LiteLLMInstructorBaseAgent` | Structured Pydantic output via Instructor |

## Out-of-Box Tools

| Category | Tool | Description |
|----------|------|-------------|
| **Search** | `SearxNGSearchTool` | Web search via SearxNG |
| | `SerperSearchTool` | Web search via Serper API |
| | `SemanticScholarSearchTool` | Academic paper search |
| | `CompositeSearchTool` | Multi-source search (combines backends) |
| | `SearchPipeline` | Full pipeline: search + resolve + scrape |
| **Scraping** | `WebScraper` | Web content extraction |
| | `PDFScraper` | PDF content extraction |
| | `DoclingScraper` | Advanced document parsing (tables, structure) |
| **Resolvers** | `CrossRefDoiResolver` | DOI resolution via CrossRef |
| | `ArxivResolver` | arXiv paper lookup |
| | `ADSResolver` | NASA ADS paper lookup |
| | `UnpaywallResolver` | Open access paper lookup |
| | `CompositeResolver` | Chain multiple resolvers |
| **Evaluation** | `RelevancyTool` | Content relevance scoring |
| | `RerankerTool` | Result reranking |
| | `SourceValidator` | Source credibility assessment |
| **Special** | `HumanTool` | Human-in-the-loop interaction |
| | `OutputTool` | Structured output capture |

## Guardrails

akd-core includes a pluggable guardrail system with a unified `GuardrailProtocol` interface:

**Providers:**
- `GraniteGuardianTool` — IBM Granite Guardian model (local or cloud)
- `RiskAgent` — LLM-based risk assessment with configurable criteria
- `CompositeGuardrail` — chain multiple providers (AND, OR, CONSENSUS modes)

**Risk categories:** Granite built-in categories, Atlas dynamic taxonomy, and science-specific risks (misinformation, bias, attribution).

```python
from akd.guardrails import guardrail
from akd.guardrails.providers import GraniteGuardianTool

@guardrail(input_guardrail=GraniteGuardianTool(), fail_on_input_risk=True)
class SafeAgent(BaseAgent):
    ...
```

## Workflow Planner

The planner converts natural language research goals into executable multi-agent workflows:

```python
from akd.planner.llm_planner import create_planner

planner = await create_planner()
session = await planner.plan_workflow("Find papers on AlphaFold and identify research gaps")
response = await session.start()

while not response.ready_to_generate:
    user_input = input(f"{response.message}\nYour response: ")
    response = await session.respond(user_input)

workflow = await session.generate_workflow()
```

The planner uses an `AgentRegistry` with auto-discovery, field mapping between agent inputs/outputs, and generates executable `WorkflowFormat` definitions.

## Extending akd-core

akd-core is designed to be extended. Every agent and tool follows a consistent 4-part pattern:

1. **InputSchema** — Pydantic model defining what goes in (requires docstring)
2. **OutputSchema** — Pydantic model defining what comes out (requires docstring)
3. **Config** — `BaseAgentConfig` or `BaseToolConfig` with settings
4. **Implementation** — subclass `BaseAgent[In, Out]` or `BaseTool[In, Out]`, implement `_arun()`

```python
from akd._base import InputSchema, OutputSchema
from akd.agents._base import AKDAgent, BaseAgentConfig

class MyInput(InputSchema):
    """What goes in."""
    query: str

class MyOutput(OutputSchema):
    """What comes out."""
    answer: str

class MyAgent(AKDAgent[MyInput, MyOutput]):
    input_schema = MyInput
    output_schema = MyOutput

    async def _arun(self, params: MyInput, run_context=None, **kwargs) -> MyOutput:
        ...  # your logic here
```

`AKDAgent` is the default batteries-included agent — it comes with LiteLLM + Instructor, ReAct tool calling, HITL, streaming, and output routing. Your agent inherits all of it. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide with tool examples, guardrail integration, and planner registration.

This is exactly how [akd-ext](https://github.com/NASA-IMPACT/akd-ext) builds on akd-core — importing base classes and creating domain-specific agents and tools.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager

### Installation

**As a dependency** (for akd-ext, akd-framework, or your own project):

```bash
uv pip install "akd @ git+https://github.com/NASA-IMPACT/accelerated-discovery.git@develop"
```

Or add to your `pyproject.toml`:

```toml
dependencies = [
    "akd @ git+https://github.com/NASA-IMPACT/accelerated-discovery.git@develop",
]
```

**For local development:**

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

# For ML features (includes sentence-transformers, docling, deepeval)
uv sync --extra ml

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Usage

See the [notebooks](notebooks) directory for examples.

## Project Structure

```
akd/
  _base/         # AbstractBase, schemas, streaming, HITL, tool calling, sessions
  agents/        # Out-of-box agents (search, analysis, utility)
  tools/         # Out-of-box tools (search, scraping, resolvers, evaluation)
  guardrails/    # GuardrailProtocol, providers, risk categories, decorators
  planner/       # LLM planner, agent registry, workflow builder
  configs/       # Project configuration and prompts

docs/            # Design philosophy and specs
notebooks/       # Usage examples (Jupyter, Marimo)
scripts/         # Utility scripts and demos
tests/           # Test suite (mirrors akd/ structure)
```

## Roadmap

These features are part of the design vision but not yet fully implemented:

- **Conflict Agent** — a dedicated agent that specifically searches for contradictory evidence and conflicting findings across sources. Currently, conflict detection is a design principle (see [Design Philosophy](docs/design_philosophy.md)) but lacks a standalone agent implementation.
- **Full Attribution Chain** — end-to-end traceability from final claims back to specific source sentences. Partial support exists today: `GapAgent` provides `attributed_source_answers` and the guardrail system includes an `ATTRIBUTION` risk category.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style guide, branch conventions, and how to create agents and tools.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
