# Contributing to akd-core

## Getting Started

```bash
# Clone the repo
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

> **Always run commands through `uv run`** — never use bare `python` or `pytest`. This ensures the correct virtual environment and dependencies are used.
>
> ```bash
> uv run pytest                          # not: pytest
> uv run python scripts/run_lit_agent.py # not: python scripts/...
> ```

## Branch Naming

All branches follow the pattern `<type>/<short-description>` and target `develop` (not `main`).

| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `bugfix/` | Bug fixes |
| `fix/` | Targeted fixes (typos, small corrections) |
| `refactor/` | Code restructuring without behavior change |
| `enhance/` | Improvements to existing features |
| `docs/` | Documentation only |

Examples: `feature/conflict-agent`, `bugfix/granite-conf-level`, `enhance/baseagentconfig`

## Commit Messages

Use lowercase `type: description` format. Keep the first line under 72 characters.

```
fix: updated config description code
refactor: cleaner public/private apis for output routing mixin
feature: add as_function to akd BaseTool
```

## Python Style

### Python 3.12+ Modern Syntax

Always use modern type syntax — never import from `typing` for things that have built-in equivalents:

```python
# Do this
name: str | None = None
items: list[str] = []
mapping: dict[str, int] = {}
type SearchResults = list[SearchResultItem]

# Not this
from typing import Optional, List, Dict
name: Optional[str] = None
items: List[str] = []
```

### General Style

- **Line length**: 120 characters (configured in ruff)
- **Quotes**: double quotes (`"`, not `'`)
- **Line endings**: LF (Unix-style)
- **Pydantic v2** for all models
- **All async**: agents and tools implement `async def _arun()`

### Variable Naming

Be descriptive and intentional with names. Code is read far more than it is written.

- No single-letter variables outside of loop counters or comprehensions
- No cryptic abbreviations — `search_results` not `sr`, `document_content` not `doc_c`
- Boolean variables should read as conditions: `is_valid`, `has_results`, `should_retry`
- Use consistent terminology from the codebase — `run_context` not `ctx`, `input_schema` not `in_schema`

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. The following hooks are configured:

- **trailing-whitespace** — removes trailing whitespace
- **end-of-file-fixer** — ensures files end with a newline
- **check-yaml** — validates YAML syntax
- **debug-statements** — catches leftover `breakpoint()` / `pdb` calls
- **isort** — sorts imports (black-compatible profile)
- **add-trailing-comma** — adds trailing commas (Python 3.6+)
- **ruff-check** — linting with auto-fix
- **ruff-format** — code formatting

To run hooks manually against all files:

```bash
pre-commit run --all-files
```

Never skip hooks with `--no-verify`. If a hook fails, fix the issue — don't bypass it.

## Testing

```bash
uv run pytest                         # full suite (parallel, with coverage)
uv run pytest tests/agents/base/ -v   # specific directory
uv run pytest -m unit                 # by marker
uv run pytest -k "test_human"         # by name pattern
```

**Markers:** `unit`, `integration`, `slow`, `requires_ml`

- Tests live in `tests/` and mirror the `akd/` directory structure
- Use `pytest-asyncio` (auto mode) — just write `async def test_...()` and it works
- Coverage is collected automatically via `pytest-cov`
- Tests run in parallel via `pytest-xdist`

## Creating Agents and Tools

akd-core uses a consistent **4-part pattern** for all agents and tools. Understanding this pattern is the key to extending the framework.

### The Public vs Private API

This distinction is important:

- **`_arun()`** — the **private** method you **override** when creating an agent or tool. This is where your implementation logic goes.
- **`arun()`** — the **public** method callers **invoke** to run your agent or tool. It handles the full lifecycle: sessions, streaming events, guardrails, error handling, and then calls your `_arun()` internally.
- **`astream()`** — the **public** async generator callers use for streaming. Yields `StreamEvent` objects.

**Never call `_arun()` directly.** Always use `arun()` or `astream()`.

### Creating an Agent

Every agent needs four parts: InputSchema, OutputSchema, Config (optional), and the Agent class.

```python
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig


class SummaryInput(InputSchema):
    """Input for the summarization agent."""

    text: str = Field(..., description="The text to summarize")
    max_length: int = Field(default=200, description="Maximum summary length in words")


class SummaryOutput(OutputSchema):
    """Output from the summarization agent."""

    summary: str = Field(..., description="The generated summary")
    word_count: int = Field(..., description="Number of words in the summary")


class SummaryConfig(BaseAgentConfig):
    """Configuration for the summarization agent."""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3


class SummaryAgent(BaseAgent[SummaryInput, SummaryOutput]):
    """Summarizes text to a target length."""

    config_schema = SummaryConfig
    input_schema = SummaryInput
    output_schema = SummaryOutput

    async def _arun(self, params: SummaryInput, run_context=None, **kwargs) -> SummaryOutput:
        # Your implementation here — this is called by the public arun()/astream()
        ...
```

**Using it:**

```python
agent = SummaryAgent(config={"model_name": "gpt-4o-mini"})

# Simple invocation
result = await agent.arun(SummaryInput(text="...", max_length=100))

# Streaming
async for event in agent.astream(SummaryInput(text="...")):
    if event.event_type == "streaming":
        print(event.token, end="")
    elif event.event_type == "completed":
        print(event.output.summary)
```

**Key points:**
- `InputSchema` and `OutputSchema` require docstrings — schema validation will fail without them
- Use Python 3.12+ generic syntax: `BaseAgent[SummaryInput, SummaryOutput]`
- Streaming, HITL, tool calling, and message trimming come for free from `BaseAgent`
- For structured LLM output, use `LiteLLMInstructorBaseAgent` instead of `BaseAgent`

### Creating a Tool

Tools follow the same pattern but inherit from `BaseTool`:

```python
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig


class FetchInput(InputSchema):
    """Input for fetching a URL."""

    url: str = Field(..., description="The URL to fetch")


class FetchOutput(OutputSchema):
    """Output from fetching a URL."""

    content: str = Field(..., description="The fetched content")
    status_code: int = Field(..., description="HTTP status code")


class FetchToolConfig(BaseToolConfig):
    """Configuration for the fetch tool."""

    timeout: int = 30


class FetchTool(BaseTool[FetchInput, FetchOutput]):
    """Fetches content from a URL."""

    config_schema = FetchToolConfig
    input_schema = FetchInput
    output_schema = FetchOutput

    async def _arun(self, params: FetchInput, run_context=None, **kwargs) -> FetchOutput:
        # Your implementation here
        ...
```

**Using it:**

```python
tool = FetchTool()

# Direct invocation
result = await tool.arun(FetchInput(url="https://example.com"))

# As a function (for LLM tool calling)
fn = tool.as_function()
result = await fn(url="https://example.com")

# Get tool definition for LLM function calling
definition = tool.as_tool_definition()
```

**Composition:** Tools can be composed using the Composite pattern. See `CompositeSearchTool`, `CompositeResolver`, and `CompositeGuardrail` for examples of combining multiple tools into one.

### Adding Guardrails to Your Agent

Use the decorator API to add input/output guardrails:

```python
from akd.guardrails import guardrail
from akd.guardrails.providers import GraniteGuardianTool

@guardrail(
    input_guardrail=GraniteGuardianTool(),
    fail_on_input_risk=True,
)
class SafeSummaryAgent(SummaryAgent):
    ...
```

Or apply at runtime:

```python
from akd.guardrails import apply_guardrails

agent = SummaryAgent()
guarded = apply_guardrails(agent, input_guardrail=GraniteGuardianTool())
```

### Registering with the Planner

To make your agent discoverable by the workflow planner, add it to the agent registry configuration. The planner's `AgentRegistry` can auto-discover agents — your agent just needs to follow the standard 4-part pattern with proper schema docstrings, and it can be registered for workflow planning.

### How akd-ext Extends akd-core

[akd-ext](https://github.com/NASA-IMPACT/akd-ext) is the canonical example of extending akd-core. It imports base classes from akd-core and builds domain-specific agents and tools on top:

```python
# In akd-ext, you import from akd-core
from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig

# Then create your domain-specific extensions
class DomainSpecificAgent(BaseAgent[MyInput, MyOutput]):
    ...
```

The same pattern works for any project that depends on akd-core.

## PR Process

1. Branch from `develop` using the naming convention above
2. Make your changes, ensuring pre-commit hooks pass
3. Write tests that mirror the `akd/` directory structure
4. Run `uv run pytest` to verify everything passes
5. Open a PR targeting `develop`

## Further Reading

For deeper dives into framework internals, see the specs in `docs/specs/`:

- **AKD_BASE.md** — base class system, AbstractBase, schema validation
- **STREAMING.md** — event types, StreamingMixin, event data models
- **TOOL_CALLING.md** — tool loop, ToolCall/ToolResult, function conversion
- **HUMAN_INTERRUPT.md** — pause/resume lifecycle, HumanTool, HumanResponse
- **AGENT_MEMORY.md** — memory sessions, message management
- **AKD_GUARDRAILS.md** — guardrail protocol, providers, risk categories
