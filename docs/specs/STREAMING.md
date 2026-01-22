# Streaming Architecture

Real-time progress and partial output streaming for AKD agents.

## Overview

The streaming system enables agents to emit events during execution, providing:

- **Progress visibility**: Track multi-step pipelines in real-time
- **Partial outputs**: Access structured results as they build progressively
- **Reasoning transparency**: Observe LLM thinking tokens (Claude extended thinking, o1)
- **Error handling**: Immediate failure notification with context

Streaming follows a simple pattern: call `agent.astream()` instead of `agent.arun()` and iterate over events.

```python
async for event in agent.astream(input_data):
    match event.event_type:
        case StreamEventType.RUNNING:
            print(f"Step: {event.data.get('step')}")
        case StreamEventType.PARTIAL:
            print(f"Partial: {event.partial_output}")
        case StreamEventType.COMPLETED:
            result = event.output
```

---

## Core Infrastructure

### StreamEventType

Enum defining all event types (`akd/_base/streaming.py`):

| Event Type | Purpose | When Emitted |
|------------|---------|--------------|
| `STARTING` | Lifecycle | Beginning of execution |
| `RUNNING` | Progress | Each pipeline step or substep |
| `STREAMING` | Raw tokens | LLM token-by-token output |
| `THINKING` | Reasoning | Claude extended thinking, o1 reasoning tokens |
| `PARTIAL` | Structured | Validated partial model as JSON builds |
| `COMPLETED` | Lifecycle | Successful completion with output |
| `FAILED` | Lifecycle | Error with details |
| `TOOL_CALLING` | Tools | Tool invocation (Stage 2) |
| `TOOL_RESULT` | Tools | Tool completion (Stage 2) |

### StreamEvent

CloudEvents-inspired envelope for all streaming events:

```python
class StreamEvent(BaseModel):
    # Core envelope
    event_type: StreamEventType
    timestamp: datetime
    event_id: str           # Unique ID (auto-generated)
    source: str | None      # Class name that generated event
    message: str | None     # Human-readable description

    # Flexible payload
    data: dict[str, Any]    # Event-specific data

    # Execution context
    context: dict[str, Any] # node_id, query, run_id, etc.
```

**Convenience properties** provide typed access to common data fields:

```python
event.output           # COMPLETED: final output
event.error            # FAILED: error message
event.token            # STREAMING: raw token string
event.thinking_content # THINKING: reasoning content
event.partial_output   # PARTIAL: partial model instance
event.tool_name        # TOOL_CALLING/TOOL_RESULT: tool name
event.tool_input       # TOOL_CALLING: tool parameters
event.tool_output      # TOOL_RESULT: tool return value
```

### StreamingMixin

Base mixin that adds `astream()` to any agent (`akd/_base/streaming.py`):

```python
class StreamingMixin:
    async def astream(
        self,
        params: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Public streaming API with input/output validation."""
        params = self._validate_input(params)

        async for event in self._astream(params, context, **kwargs):
            if event.event_type == StreamEventType.COMPLETED:
                output = self._validate_output(event.data.get("output"))
                yield StreamEvent(..., data={"output": output})
            else:
                yield event

    async def _astream(self, params, context, **kwargs):
        """Override this for custom streaming logic."""
        # Default: STARTING -> RUNNING -> _arun() -> COMPLETED/FAILED
        ...
```

Key design:
- `astream()` handles validation, `_astream()` contains streaming logic
- Input validation happens before any events are yielded
- Output validation happens on COMPLETED event before yielding
- Default `_astream()` wraps `_arun()` for backward compatibility

---

## Event Types in Detail

### STARTING

Emitted once at execution start:

```python
StreamEvent(
    event_type=StreamEventType.STARTING,
    source="DeepLitSearchAgent",
    message="Starting deep literature search: climate change...",
    data={"query": "climate change research"},
    context={"run_id": "abc123", "query": "climate change research"},
)
```

### RUNNING

Emitted for pipeline progress. Use `data["step"]` for step tracking:

```python
StreamEvent(
    event_type=StreamEventType.RUNNING,
    source="DeepLitSearchAgent",
    message="Starting research iteration 2/5",
    data={
        "step": "research.iteration",
        "substep": "iteration_2",
        "step_index": 4,
        "total_steps": 5,
        "iteration": 2,
        "max_iterations": 5,
        "current_results_count": 15,
    },
    context={"run_id": "abc123"},
)
```

Common step patterns:
- `step="triage"` - Query analysis
- `step="research.queries"` - Query generation
- `step="research.iteration"` - Main iteration loop
- `step="research.search"` - Search execution
- `step="research.evaluate"` - Quality evaluation
- `step="synthesis"` - Report/answer generation

### STREAMING

Raw LLM tokens as they arrive (batched for efficiency):

```python
StreamEvent(
    event_type=StreamEventType.STREAMING,
    source="LiteLLMInstructorBaseAgent",
    data={"token": "The research shows"},
    context={"run_id": "abc123"},
)
```

Access via `event.token`.

### THINKING

Reasoning tokens from Claude extended thinking or o1:

```python
StreamEvent(
    event_type=StreamEventType.THINKING,
    source="LiteLLMInstructorBaseAgent",
    message="Reasoning...",
    data={"thinking_content": "Let me analyze the query structure..."},
    context={"run_id": "abc123"},
)
```

Access via `event.thinking_content`.

### PARTIAL

Validated partial output as JSON builds:

```python
StreamEvent(
    event_type=StreamEventType.PARTIAL,
    source="DeepLitSearchAgent",
    message="Research results available",
    data={
        "partial_output": PartialModel[LitSearchAgentOutputSchema](
            results=[...],
            extra={"key_findings": [...]},
        )
    },
    context={"run_id": "abc123"},
)
```

Access via `event.partial_output`. The partial model has all fields Optional.

### COMPLETED

Successful completion with validated output:

```python
StreamEvent(
    event_type=StreamEventType.COMPLETED,
    source="DeepLitSearchAgent",
    message="Deep literature search completed",
    data={"output": LitSearchAgentOutputSchema(...)},
    context={"run_id": "abc123"},
)
```

Access via `event.output`.

### FAILED

Error with details:

```python
StreamEvent(
    event_type=StreamEventType.FAILED,
    source="DeepLitSearchAgent",
    message="Failed: Connection timeout",
    data={"error": "Connection timeout", "error_type": "TimeoutError"},
    context={"run_id": "abc123"},
)
```

Access via `event.error`. The original exception is re-raised after this event.

---

## Adding Streaming to New Agents

### Step 1: Override `_astream()`

```python
from akd._base.streaming import StreamEvent, StreamEventType

class MyAgent(LiteLLMInstructorBaseAgent):

    async def _astream(
        self,
        params: MyInputSchema,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution with progress events."""
        class_name = self.__class__.__name__

        # Setup context with auto-generated run_id
        run_context = context.copy() if context else {}
        if "run_id" not in run_context:
            run_context["run_id"] = uuid.uuid4().hex[:8]

        # STARTING event
        yield StreamEvent(
            event_type=StreamEventType.STARTING,
            source=class_name,
            message=f"Starting {class_name}",
            context=run_context,
        )

        try:
            # Your pipeline steps here...
            yield StreamEvent(
                event_type=StreamEventType.RUNNING,
                source=class_name,
                message="Processing step 1",
                data={"step": "step1"},
                context=run_context,
            )

            result = await self._do_work(params)

            # COMPLETED event
            yield StreamEvent(
                event_type=StreamEventType.COMPLETED,
                source=class_name,
                message=f"Completed {class_name}",
                data={"output": result},
                context=run_context,
            )

        except Exception as e:
            # FAILED event
            yield StreamEvent(
                event_type=StreamEventType.FAILED,
                source=class_name,
                message=f"Failed: {e!s}",
                data={"error": str(e), "error_type": type(e).__name__},
                context=run_context,
            )
            raise  # Re-raise after yielding FAILED
```

### Step 2: Implement `_arun()` via `_astream()`

For consistency, delegate `_arun()` to `_astream()`:

```python
async def _arun(
    self,
    params: MyInputSchema,
    **kwargs: Any,
) -> MyOutputSchema:
    """Run by collecting output from _astream()."""
    output = None
    async for event in self._astream(params, **kwargs):
        if event.event_type == StreamEventType.COMPLETED:
            output = event.output

    if output is None:
        raise RuntimeError("No output received from _astream()")
    return output
```

### Step 3: Helper for RUNNING Events

Create a helper method for consistent step events:

```python
def _emit_step_event(
    self,
    step: str,
    message: str,
    run_context: dict[str, Any],
    step_index: int | None = None,
    total_steps: int | None = None,
    substep: str | None = None,
    **data_kwargs: Any,
) -> StreamEvent:
    """Create a RUNNING event for a pipeline step."""
    data = {"step": step, **data_kwargs}
    if step_index is not None:
        data["step_index"] = step_index
    if total_steps is not None:
        data["total_steps"] = total_steps
    if substep is not None:
        data["substep"] = substep

    return StreamEvent(
        event_type=StreamEventType.RUNNING,
        source=self.__class__.__name__,
        message=message,
        data=data,
        context=run_context,
    )
```

---

## PartialModel for Progressive Output

`PartialModel[T]` creates a version of any Pydantic model where all fields are Optional (`akd/utils.py`):

```python
from akd.utils import PartialModel
from akd.agents.search._base import LitSearchAgentOutputSchema

# Create partial with only some fields populated
partial = PartialModel[LitSearchAgentOutputSchema](
    results=[...],           # Available
    extra={"key_findings": [...]},
    # answer=None (implicit)
    # report=None (implicit)
)

# Serialize for frontend
partial.model_dump()
# {'answer': None, 'report': None, 'results': [...], 'extra': {...}}
```

### Using PartialModel in Streaming

Emit PARTIAL events as pipeline produces partial results:

```python
# After search completes but before synthesis
yield StreamEvent(
    event_type=StreamEventType.PARTIAL,
    source=class_name,
    message="Search results available",
    data={
        "partial_output": PartialModel[LitSearchAgentOutputSchema](
            results=all_results,
            extra={"iterations_performed": iteration},
        ),
    },
    context=run_context,
)

# After report generation
yield StreamEvent(
    event_type=StreamEventType.PARTIAL,
    source=class_name,
    message="Report generated",
    data={
        "partial_output": PartialModel[LitSearchAgentOutputSchema](
            results=all_results,
            report=detailed_report,
            extra={"iterations_performed": iteration},
        ),
    },
    context=run_context,
)
```

### LLM Streaming with PartialModel

`LiteLLMInstructorBaseAgent._stream_llm_response()` validates partial JSON as it streams:

```python
async def _stream_llm_response(self, messages, response_model, token_batch_size=10):
    PartialResponseModel = PartialModel[response_model]
    accumulated = ""

    async for chunk in response:
        accumulated += delta.content

        # Try to validate as partial
        parsed = self._try_parse_json(accumulated)
        if parsed:
            try:
                partial = PartialResponseModel.model_validate(parsed)
                yield {"type": StreamEventType.PARTIAL, "partial": partial}
            except Exception:
                pass  # Skip invalid partials
```

---

## Usage Examples

### DeepLitSearchAgent - Multi-Step Pipeline

```python
from akd.agents.search.deep_search import DeepLitSearchAgent, LitSearchAgentInputSchema

agent = DeepLitSearchAgent()
input_data = LitSearchAgentInputSchema(query="climate change mitigation strategies")

async for event in agent.astream(input_data):
    match event.event_type:
        case StreamEventType.STARTING:
            print(f"Starting: {event.message}")

        case StreamEventType.RUNNING:
            step = event.data.get("step", "")
            if step.startswith("research.iteration"):
                iteration = event.data.get("iteration", 0)
                print(f"Iteration {iteration}: {event.message}")
            else:
                print(f"  {event.message}")

        case StreamEventType.PARTIAL:
            partial = event.partial_output
            if partial.results:
                print(f"  Results so far: {len(partial.results)}")

        case StreamEventType.COMPLETED:
            result = event.output
            print(f"Complete: {len(result.results)} results")
            print(f"Answer: {result.answer[:200]}...")

        case StreamEventType.FAILED:
            print(f"Error: {event.error}")
```

### ControlledSearchAgent - Iteration-Based

```python
from akd.agents.search.controlled import ControlledSearchAgent

agent = ControlledSearchAgent()

async for event in agent.astream({"query": "machine learning optimization"}):
    match event.event_type:
        case StreamEventType.RUNNING:
            step = event.data.get("step", "")

            if step == "iteration":
                step_index = event.data.get("step_index", 0)
                total = event.data.get("total_steps", 5)
                results = event.data.get("results_so_far", 0)
                print(f"[{step_index}/{total}] Results: {results}")

            elif step == "iteration.evaluate":
                positive = event.data.get("positive_rubrics", 0)
                strong = event.data.get("strong_rubrics", [])
                print(f"  Quality: {positive}/6 - Strong: {strong}")
```

### Base Agent LLM Streaming

```python
from akd.agents._base import LiteLLMInstructorBaseAgent

class MyAgent(LiteLLMInstructorBaseAgent):
    ...

async for event in agent.astream(input_data, token_batch_size=20):
    match event.event_type:
        case StreamEventType.STREAMING:
            print(event.token, end="", flush=True)

        case StreamEventType.THINKING:
            print(f"\n[Thinking] {event.thinking_content}")

        case StreamEventType.PARTIAL:
            # Partial structured output
            partial = event.partial_output
            if partial and hasattr(partial, 'answer') and partial.answer:
                print(f"\nPartial answer: {partial.answer[:100]}...")
```

---

## Testing Streaming

### Basic Event Collection

```python
import pytest

@pytest.mark.asyncio
async def test_agent_streaming():
    agent = DeepLitSearchAgent()
    events = []

    async for event in agent.astream({"query": "test query"}):
        events.append(event)

    # Verify event sequence
    assert events[0].event_type == StreamEventType.STARTING
    assert events[-1].event_type == StreamEventType.COMPLETED

    # Check for expected steps
    running_steps = [e.data.get("step") for e in events if e.event_type == StreamEventType.RUNNING]
    assert "triage" in running_steps
    assert "research.iteration" in running_steps
```

### Event Type Assertions

```python
def assert_streaming_contract(events: list[StreamEvent]):
    """Verify streaming contract is followed."""
    event_types = [e.event_type for e in events]

    # Must start with STARTING
    assert event_types[0] == StreamEventType.STARTING

    # Must end with COMPLETED or FAILED
    assert event_types[-1] in (StreamEventType.COMPLETED, StreamEventType.FAILED)

    # COMPLETED must have output
    if event_types[-1] == StreamEventType.COMPLETED:
        assert events[-1].output is not None

    # FAILED must have error
    if event_types[-1] == StreamEventType.FAILED:
        assert events[-1].error is not None
```

### Mock Event Consumer

```python
class StreamEventCollector:
    """Collect and categorize stream events for testing."""

    def __init__(self):
        self.events = []
        self.by_type = {}

    async def collect(self, stream: AsyncIterator[StreamEvent]):
        async for event in stream:
            self.events.append(event)
            self.by_type.setdefault(event.event_type, []).append(event)
        return self

    @property
    def output(self):
        completed = self.by_type.get(StreamEventType.COMPLETED, [])
        return completed[0].output if completed else None

    @property
    def partials(self):
        return [e.partial_output for e in self.by_type.get(StreamEventType.PARTIAL, [])]

# Usage
collector = StreamEventCollector()
await collector.collect(agent.astream(input_data))
assert collector.output is not None
assert len(collector.partials) >= 1
```

---

## Best Practices

1. **Always yield STARTING first** - Consumers expect it for initialization
2. **Always yield COMPLETED or FAILED last** - Never leave streams hanging
3. **Re-raise after FAILED** - Allow error propagation after event emission
4. **Use consistent step naming** - `parent.child` pattern (e.g., `research.search`)
5. **Include context in all events** - Enables filtering and correlation
6. **Batch tokens for efficiency** - Default `token_batch_size=10` balances latency/overhead
7. **Emit PARTIAL progressively** - Let consumers see results as they become available
8. **Delegate `_arun()` to `_astream()`** - Single source of truth for execution logic
