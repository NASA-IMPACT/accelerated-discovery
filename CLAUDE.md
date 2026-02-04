# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Commands

```bash
# Setup
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev          # install with test deps

# Run anything through uv (never bare python/pytest)
uv run pytest                         # full suite (parallel, coverage)
uv run pytest tests/agents/base/ -v   # specific directory
uv run pytest -m unit                 # by marker: unit, integration, slow, requires_ml
uv run pytest -k "test_human"         # by name pattern
uv run python scripts/run_lit_agent.py
```

## Python Style

- **Python 3.12+** — use modern syntax everywhere:
  - `str | None` not `Optional[str]`, `list[str]` not `List[str]`
  - Generic classes: `class Memory[T](BaseModel):` not `class Memory(BaseModel, Generic[T]):`
  - `type Alias = ...` for type aliases where appropriate
- All async: agents and tools implement `async def _arun()`
- Pydantic v2 for all models

## Architecture

Human-centric Multi-Agent System (MAS) framework for scientific discovery.
See `docs/design_philosophy.md` for full design principles.

### Base System (`akd/_base/`)

Everything inherits from `AbstractBase` or `UnrestrictedAbstractBase`. See `docs/specs/AKD_BASE.md` for the full reference.

**Key exports from `akd._base`:**

| Class | Purpose |
|-------|---------|
| `AbstractBase` | Strict base with schema validation (agents, tools) |
| `InputSchema` / `OutputSchema` / `IOSchema` | Typed schemas with required docstrings |
| `BaseConfig` | Configuration base |
| `StreamEvent` / `StreamingMixin` | Streaming event system |
| `ToolCall` / `ToolResult` / `ToolCallingMixin` | Tool calling infrastructure |
| `RunContext` | Execution context passed to agents (in `_base/structures.py`) |
| `HumanResponse` | Human reply for HITL resumption (in `_base/structures.py`) |
| `Memory` | Message storage with session lifecycle |

### Adding Agents or Tools

Every agent/tool has 4 parts:

```python
from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig

class MyInput(InputSchema):
    """What goes in."""
    query: str

class MyOutput(OutputSchema):
    """What comes out."""
    answer: str

class MyConfig(BaseAgentConfig):
    model: str = "gpt-4o-mini"

class MyAgent(BaseAgent[MyInput, MyOutput]):
    config_schema = MyConfig
    input_schema = MyInput
    output_schema = MyOutput

    async def _arun(self, params: MyInput, run_context: RunContext | None = None, **kwargs) -> MyOutput:
        ...
```

Same pattern for tools — inherit `BaseTool[InSchema, OutSchema]` with `BaseToolConfig`.

### Streaming & Human-in-the-Loop

Agents support streaming via `astream()` which yields typed `StreamEvent` subclasses.
The `HumanTool` enables HITL: the tool loop detects it via `isinstance` (not by name) and emits `HumanInputRequiredEvent`.

See specs:
- `docs/specs/STREAMING.md` — event types, streaming mixin
- `docs/specs/TOOL_CALLING.md` — tool loop, ToolCall/ToolResult, HumanTool interception
- `docs/specs/HUMAN_INTERRUPT.md` — pause/resume lifecycle
- `docs/specs/AGENT_MEMORY.md` — memory sessions, message management

### Key Data Structures

**`akd/_base/structures.py`** (internal):
- `RunContext` — execution context (messages, human_response, run_id, extra keys)
- `HumanResponse` — human's reply to HITL event (extends ToolResult)

**`akd/structures.py`** (public, also re-exports `HumanResponse`):
- `SearchResult` / `SearchResultItem` — search results with metadata
- `ExtractionSchema` / `SingleEstimation` — extraction output schemas
- `HumanResponse` — re-exported for backend convenience

## Project Structure

```
akd/
├── _base/             # Base classes, streaming, tool calling, memory, structures
├── agents/            # Agent implementations
│   ├── _base.py       #   BaseAgent, InstructorBaseAgent, LiteLLMInstructorBaseAgent
│   ├── search/        #   SearchAgent, DeepLitSearchAgent, ControlledSearchAgent, AspectSearchAgent
│   ├── gap_analysis/  #   GapAgent
│   ├── extraction.py  #   EstimationExtractionAgent
│   ├── query.py       #   QueryAgent, FollowUpQueryAgent
│   ├── relevancy.py   #   Relevancy checking agents
│   ├── storm/         #   STORM workflow agent
│   └── intents.py     #   Intent detection
├── tools/             # Tool implementations
│   ├── _base.py       #   BaseTool, BaseToolConfig
│   ├── human.py       #   HumanTool (HITL)
│   ├── search/        #   SearxNG, Serper, SemanticScholar, CodeSearch, Composite, Pipeline
│   ├── scrapers/      #   Web, PDF, Crawl4AI, PyPaperBot, Docling scrapers
│   ├── resolvers/     #   DOI, Arxiv, ADS, Unpaywall resolvers
│   ├── reranker.py    #   CrossEncoder, NoOp rerankers
│   ├── relevancy.py   #   Relevancy checker
│   └── source_validator.py
├── configs/           # Configuration (project, prompts, lit, storm)
├── guardrails/        # Safety and validation guardrails
├── mapping/           # Agent/tool registry and field mapping
├── planner/           # Workflow planning
└── structures.py      # Public data structures

tests/                 # Mirrors akd/ structure — pytest + asyncio + xdist
docs/specs/            # Detailed specifications (see links above)
```
