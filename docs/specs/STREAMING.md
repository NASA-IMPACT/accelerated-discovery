# Streaming Architecture

Real-time progress and partial output streaming for AKD agents.

## Overview

The streaming system enables agents to emit typed events during execution, providing:

- **Progress visibility**: Track multi-step pipelines in real-time
- **Partial outputs**: Access structured results as they build progressively
- **Reasoning transparency**: Observe LLM thinking tokens (Claude extended thinking, o1)
- **Tool calling**: Track tool invocations and results
- **Human interaction**: Pause for human input and resume with responses
- **Error handling**: Immediate failure notification with context

Streaming follows a simple pattern: call `agent.astream()` instead of `agent.arun()` and iterate over events.

```python
async for event in agent.astream(input_data):
    match event.event_type:
        case StreamEventType.RUNNING:
            print(f"Step: {event.message}")
        case StreamEventType.PARTIAL:
            print(f"Partial: {event.partial_output}")
        case StreamEventType.COMPLETED:
            result = event.output
```

Or use `isinstance` checks with typed event subclasses:

```python
async for event in agent.astream(input_data):
    if isinstance(event, CompletedEvent):
        print(event.data.output)       # Fully typed access
    elif isinstance(event, StreamingTokenEvent):
        print(event.data.token, end="")
    elif isinstance(event, HumanInputRequiredEvent):
        print(f"Question: {event.data.human_input.question}")
```

---

## Core Infrastructure

### StreamEventType

Enum defining all event types (`akd/_base/streaming.py`):

| Event Type | Purpose | When Emitted |
|------------|---------|--------------|
| `STARTING` | Lifecycle | Beginning of execution |
| `RUNNING` | Progress | Each pipeline step or substep |
| `STREAMING` | Raw tokens | LLM token-by-token output (batched) |
| `THINKING` | Reasoning | Claude extended thinking, o1 reasoning tokens |
| `PARTIAL` | Structured | Validated partial model as JSON builds |
| `COMPLETED` | Lifecycle | Successful completion with output |
| `FAILED` | Lifecycle | Error with details |
| `TOOL_CALLING` | Tools | Tool invocation |
| `TOOL_RESULT` | Tools | Tool completion |
| `HUMAN_INPUT_REQUIRED` | Human | Agent needs human input to continue |
| `HUMAN_RESPONSE` | Human | Resumed with human input |

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

    # Typed EventData or legacy dict payload
    data: EventData | dict[str, Any]

    # Execution context
    run_context: RunContext  # run_id, messages, human_response, etc.
```

**Convenience properties** provide typed access with dict fallback:

```python
event.output           # COMPLETED: final output
event.error            # FAILED: error message
event.token            # STREAMING: raw token string
event.thinking_content # THINKING: reasoning content
event.partial_output   # PARTIAL: partial model instance
event.tool_name        # TOOL_CALLING/TOOL_RESULT: tool name
event.tool_input       # TOOL_CALLING: tool parameters
event.tool_output      # TOOL_RESULT: tool return value
event.human_prompt     # HUMAN_INPUT_REQUIRED: question string
event.message_history  # run_context.messages for resumption
```

### RunContext

Execution context carried by every event (`akd/_base/tool_calling.py`):

```python
class RunContext(BaseModel):
    model_config = ConfigDict(extra="allow")  # Accepts arbitrary extra keys

    human_response: HumanResponse | None = None  # For resumption after human input
    messages: list[dict[str, Any]] | None = None  # Conversation history
    run_id: str | None = None                     # Unique execution run ID
```

`RunContext` replaces the previous `context: dict[str, Any]` parameter. It provides type safety for known fields while allowing arbitrary extra keys via `extra="allow"`.

### StreamingMixin

Base mixin that adds `astream()` to any agent (`akd/_base/streaming.py`):

```python
class StreamingMixin:
    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Public streaming API with input/output validation."""
        params = self._validate_input(params)

        async for event in self._astream(params, run_context, **kwargs):
            if event.event_type == StreamEventType.COMPLETED:
                output = self._validate_output(event.output)
                yield CompletedEvent(
                    source=event.source,
                    message=event.message,
                    data=CompletedEventData(output=output),
                    run_context=event.run_context,
                )
            else:
                yield event

    async def _astream(self, params, run_context, **kwargs):
        """Override this for custom streaming logic."""
        # Default: STARTING -> RUNNING -> _arun() -> COMPLETED/FAILED
        ...
```

Key design:
- `astream()` handles validation, `_astream()` contains streaming logic
- Input validation happens before any events are yielded
- Output validation happens on COMPLETED event before yielding
- Default `_astream()` wraps `_arun()` for backward compatibility
- `run_context` auto-generates a `run_id` if not provided

---

## Typed Event Data Models

Each event type has a corresponding typed data model (`akd/_base/streaming.py`). These provide type safety and IDE autocompletion.

### Data Models

| Event Type | Data Model | Fields |
|------------|-----------|--------|
| `STARTING` | `StartingEventData[T]` | `params: T \| None` |
| `RUNNING` | `RunningEventData` | `extra="allow"` (flexible) |
| `COMPLETED` | `CompletedEventData[T]` | `output: T` |
| `FAILED` | `FailedEventData` | `error: str`, `error_type: str` |
| `STREAMING` | `StreamingEventData` | `token: str` |
| `THINKING` | `ThinkingEventData` | `thinking_content: str \| None`, `reflection_prompt: str \| None`, `streaming: bool` |
| `PARTIAL` | `PartialEventData[T]` | `partial_output: T` |
| `TOOL_CALLING` | `ToolCallingEventData` | `tool_call: ToolCall` |
| `TOOL_RESULT` | `ToolResultEventData` | `result: ToolResult` |
| `HUMAN_INPUT_REQUIRED` | `HumanInputRequiredEventData` | `human_input: Any`, `tool_call_id: str`, `tool_name: str` |
| `HUMAN_RESPONSE` | `HumanResponseEventData` | `tool_call_id: str`, `response: Any` |

The `EventData` union type represents all data models:

```python
EventData = (
    StartingEventData | RunningEventData | CompletedEventData
    | FailedEventData | StreamingEventData | ThinkingEventData
    | PartialEventData | ToolCallingEventData | ToolResultEventData
    | HumanInputRequiredEventData | HumanResponseEventData
)
```

### Generic Data Models

`StartingEventData[T]`, `CompletedEventData[T]`, and `PartialEventData[T]` are generic. The type parameter matches the agent's input/output schema:

```python
# STARTING carries input params
StartingEventData[MyInputSchema](params=input_data)

# COMPLETED carries validated output
CompletedEventData[MyOutputSchema](output=result)

# PARTIAL carries partial output
PartialEventData[PartialModel[MyOutputSchema]](partial_output=partial)
```

---

## Discriminated Event Subclasses

Each event type has a typed subclass with a `Literal` discriminator. These enable `isinstance` checks and provide fully typed `data` access.

| Subclass | Event Type | Data Type |
|----------|-----------|-----------|
| `StartingEvent` | `STARTING` | `StartingEventData` |
| `RunningEvent` | `RUNNING` | `RunningEventData` |
| `CompletedEvent` | `COMPLETED` | `CompletedEventData` |
| `FailedEvent` | `FAILED` | `FailedEventData` |
| `StreamingTokenEvent` | `STREAMING` | `StreamingEventData` |
| `ThinkingEvent` | `THINKING` | `ThinkingEventData` |
| `PartialOutputEvent` | `PARTIAL` | `PartialEventData` |
| `ToolCallingEvent` | `TOOL_CALLING` | `ToolCallingEventData` |
| `ToolResultEvent` | `TOOL_RESULT` | `ToolResultEventData` |
| `HumanInputRequiredEvent` | `HUMAN_INPUT_REQUIRED` | `HumanInputRequiredEventData` |
| `HumanResponseEvent` | `HUMAN_RESPONSE` | `HumanResponseEventData` |

### isinstance Pattern (Recommended)

```python
from akd._base.streaming import (
    CompletedEvent, FailedEvent, StreamingTokenEvent,
    ThinkingEvent, PartialOutputEvent, ToolCallingEvent,
    ToolResultEvent, HumanInputRequiredEvent, HumanResponseEvent,
)

async for event in agent.astream(input_data):
    if isinstance(event, StreamingTokenEvent):
        print(event.data.token, end="")        # Typed: str

    elif isinstance(event, ThinkingEvent):
        print(event.data.thinking_content)      # Typed: str | None

    elif isinstance(event, PartialOutputEvent):
        print(event.data.partial_output)        # Typed: T

    elif isinstance(event, ToolCallingEvent):
        tc = event.data.tool_call               # Typed: ToolCall
        print(f"Calling {tc.tool_name}({tc.arguments})")

    elif isinstance(event, ToolResultEvent):
        r = event.data.result                   # Typed: ToolResult
        print(f"{r.tool_name} → {r.content}")

    elif isinstance(event, HumanInputRequiredEvent):
        print(event.data.human_input.question)  # Typed access
        saved = event.run_context.messages      # For resumption

    elif isinstance(event, HumanResponseEvent):
        print(f"Human said: {event.data.response}")

    elif isinstance(event, CompletedEvent):
        print(event.data.output)                # Typed: T

    elif isinstance(event, FailedEvent):
        print(event.data.error)                 # Typed: str
```

### event_type Pattern (Alternative)

```python
async for event in agent.astream(input_data):
    match event.event_type:
        case StreamEventType.STREAMING:
            print(event.token, end="")          # Convenience property
        case StreamEventType.THINKING:
            print(event.thinking_content)
        case StreamEventType.COMPLETED:
            result = event.output
        case StreamEventType.FAILED:
            print(event.error)
```

---

## Event Types in Detail

### STARTING

Emitted once at execution start. Carries input params.

```python
StartingEvent(
    source="DeepLitSearchAgent",
    message="Starting deep literature search",
    data=StartingEventData(params=input_data),
    run_context=run_context,
)
```

### RUNNING

Emitted for pipeline progress. `RunningEventData` uses `extra="allow"` for flexible data.

```python
RunningEvent(
    source="DeepLitSearchAgent",
    message="Starting research iteration 2/5",
    data=RunningEventData(
        step="research.iteration",
        substep="iteration_2",
        step_index=4,
        total_steps=5,
    ),
    run_context=run_context,
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

Raw LLM tokens as they arrive (batched for efficiency, default `token_batch_size=10`):

```python
StreamingTokenEvent(
    source="LiteLLMInstructorBaseAgent",
    data=StreamingEventData(token="The research shows"),
    run_context=run_context,
)
```

### THINKING

Reasoning tokens from Claude extended thinking, o1, or reflection prompts:

```python
ThinkingEvent(
    source="LiteLLMInstructorBaseAgent",
    message="Reasoning...",
    data=ThinkingEventData(
        thinking_content="Let me analyze the query structure...",
        streaming=True,  # True during streaming, False for reflection
    ),
    run_context=run_context,
)
```

Also emitted when `reflection_prompt` is injected after tool results:

```python
ThinkingEvent(
    message="Reflecting on results...",
    data=ThinkingEventData(reflection_prompt="Reflect on the results..."),
    run_context=run_context,
)
```

### PARTIAL

Validated partial output as JSON builds:

```python
PartialOutputEvent(
    source="DeepLitSearchAgent",
    message="Research results available",
    data=PartialEventData(
        partial_output=PartialModel[LitSearchAgentOutputSchema](
            results=[...],
            extra={"key_findings": [...]},
        )
    ),
    run_context=run_context,
)
```

The partial model has all fields Optional via `PartialModel[T]`.

### TOOL_CALLING

Emitted for each tool call before execution:

```python
ToolCallingEvent(
    source="LiteLLMInstructorBaseAgent",
    message="Calling search_papers",
    data=ToolCallingEventData(
        tool_call=ToolCall(
            tool_call_id="call_abc123",
            tool_name="search_papers",
            arguments={"query": "quantum computing"},
        )
    ),
    run_context=run_context,
)
```

### TOOL_RESULT

Emitted after tool execution completes:

```python
ToolResultEvent(
    source="LiteLLMInstructorBaseAgent",
    message="Result from search_papers",
    data=ToolResultEventData(
        result=ToolResult(
            tool_call_id="call_abc123",
            tool_name="search_papers",
            content={"results": [...]},
        )
    ),
    run_context=run_context,
)
```

### HUMAN_INPUT_REQUIRED

Emitted when the agent calls `ask_human` and pauses for input:

```python
HumanInputRequiredEvent(
    source="LiteLLMInstructorBaseAgent",
    message="Human input required: What is your name?",
    data=HumanInputRequiredEventData(
        human_input=HumanToolInput(question="What is your name?"),
        tool_call_id="call_human_456",
        tool_name="ask_human",
    ),
    run_context=run_context,  # .messages contains conversation history
)
```

The stream ends gracefully after this event. See [HUMAN_INTERRUPT.md](./HUMAN_INTERRUPT.md) for the full pause/resume lifecycle.

### HUMAN_RESPONSE

Emitted when resuming with a human response:

```python
HumanResponseEvent(
    source="LiteLLMInstructorBaseAgent",
    message="Resumed with human input",
    data=HumanResponseEventData(
        tool_call_id="call_human_456",
        response={"response": "Alice"},
    ),
    run_context=run_context,
)
```

### COMPLETED

Successful completion with validated output:

```python
CompletedEvent(
    source="DeepLitSearchAgent",
    message="Completed DeepLitSearchAgent",
    data=CompletedEventData(output=LitSearchAgentOutputSchema(...)),
    run_context=run_context,
)
```

### FAILED

Error with details. The original exception is re-raised after this event.

```python
FailedEvent(
    source="DeepLitSearchAgent",
    message="Failed: Connection timeout",
    data=FailedEventData(error="Connection timeout", error_type="TimeoutError"),
    run_context=run_context,
)
```

---

## Adding Streaming to New Agents

### Step 1: Override `_astream()`

```python
from akd._base.streaming import (
    StreamEvent, StreamEventType, StartingEvent, StartingEventData,
    RunningEvent, RunningEventData, CompletedEvent, CompletedEventData,
    FailedEvent, FailedEventData,
)
from akd._base.tool_calling import RunContext

class MyAgent(LiteLLMInstructorBaseAgent):

    async def _astream(
        self,
        params: MyInputSchema,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution with progress events."""
        class_name = self.__class__.__name__

        # Setup context with auto-generated run_id
        run_context = (run_context or RunContext()).model_copy()
        run_context.run_id = run_context.run_id or uuid.uuid4().hex[:8]

        # STARTING event
        yield StartingEvent(
            source=class_name,
            message=f"Starting {class_name}",
            data=StartingEventData(params=params),
            run_context=run_context,
        )

        try:
            # Your pipeline steps here...
            yield RunningEvent(
                source=class_name,
                message="Processing step 1",
                data=RunningEventData(step="step1"),
                run_context=run_context,
            )

            result = await self._do_work(params)

            # COMPLETED event
            yield CompletedEvent(
                source=class_name,
                message=f"Completed {class_name}",
                data=CompletedEventData(output=result),
                run_context=run_context,
            )

        except Exception as e:
            # FAILED event
            yield FailedEvent(
                source=class_name,
                message=f"Failed: {e!s}",
                data=FailedEventData(error=str(e), error_type=type(e).__name__),
                run_context=run_context,
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
        if isinstance(event, CompletedEvent):
            output = event.data.output

    if output is None:
        raise RuntimeError("No output received from _astream()")
    return output
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
yield PartialOutputEvent(
    source=class_name,
    message="Search results available",
    data=PartialEventData(
        partial_output=PartialModel[LitSearchAgentOutputSchema](
            results=all_results,
            extra={"iterations_performed": iteration},
        ),
    ),
    run_context=run_context,
)
```

### LLM Streaming with PartialModel

`LiteLLMInstructorBaseAgent._stream_llm_response()` validates partial JSON as it streams:

```python
PartialResponseModel = PartialModel[response_model]
accumulated = ""

async for chunk in response:
    accumulated += delta.content

    parsed = self._try_parse_json(accumulated)
    if parsed and parsed != last_partial_dict:
        last_partial_dict = parsed
        try:
            partial = PartialResponseModel.model_validate(parsed)
            yield PartialOutputEvent(
                source=class_name,
                message="Partial...",
                data=PartialEventData(partial_output=partial),
                run_context=run_context,
            )
        except Exception:
            pass  # Skip invalid partials
```

---

## Usage Examples

### Full Event Handling with Typed Subclasses

```python
from akd._base.streaming import *
from akd._base.tool_calling import HumanResponse, RunContext

agent = MyAgent(config=config)
input_data = MyInputSchema(query="climate change research")

async for event in agent.astream(input_data):
    if isinstance(event, StartingEvent):
        print(f"Started: {event.message}")

    elif isinstance(event, RunningEvent):
        print(f"Progress: {event.message}")

    elif isinstance(event, StreamingTokenEvent):
        print(event.data.token, end="", flush=True)

    elif isinstance(event, ThinkingEvent):
        if event.data.thinking_content:
            print(f"\n[Thinking] {event.data.thinking_content}")

    elif isinstance(event, PartialOutputEvent):
        partial = event.data.partial_output
        if hasattr(partial, 'results') and partial.results:
            print(f"\n  Results so far: {len(partial.results)}")

    elif isinstance(event, ToolCallingEvent):
        tc = event.data.tool_call
        print(f"\nCalling tool: {tc.tool_name}")

    elif isinstance(event, ToolResultEvent):
        r = event.data.result
        status = "error" if r.error else "ok"
        print(f"  {r.tool_name}: {status}")

    elif isinstance(event, HumanInputRequiredEvent):
        question = event.data.human_input.question
        answer = input(f"\n{question}: ")
        # Save for resumption
        resume_ctx = RunContext(
            messages=event.run_context.messages,
            human_response=HumanResponse(
                tool_call_id=event.data.tool_call_id,
                content={"response": answer},
            ),
        )
        break

    elif isinstance(event, CompletedEvent):
        result = event.data.output
        print(f"\nComplete: {result}")

    elif isinstance(event, FailedEvent):
        print(f"\nError: {event.data.error}")
```

### DeepLitSearchAgent - Multi-Step Pipeline

```python
from akd.agents.search.deep_search import DeepLitSearchAgent, LitSearchAgentInputSchema

agent = DeepLitSearchAgent()
input_data = LitSearchAgentInputSchema(query="climate change mitigation strategies")

async for event in agent.astream(input_data):
    if isinstance(event, RunningEvent):
        print(f"  {event.message}")
    elif isinstance(event, PartialOutputEvent):
        partial = event.data.partial_output
        if partial.results:
            print(f"  Results so far: {len(partial.results)}")
    elif isinstance(event, CompletedEvent):
        result = event.data.output
        print(f"Complete: {len(result.results)} results")
```

### Base Agent LLM Streaming

```python
async for event in agent.astream(input_data, token_batch_size=20):
    if isinstance(event, StreamingTokenEvent):
        print(event.data.token, end="", flush=True)
    elif isinstance(event, ThinkingEvent):
        print(f"\n[Thinking] {event.data.thinking_content}")
    elif isinstance(event, PartialOutputEvent):
        partial = event.data.partial_output
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
    agent = MyAgent(config=config)
    events = []

    async for event in agent.astream(input_data):
        events.append(event)

    # Verify event sequence
    assert isinstance(events[0], StartingEvent)
    assert isinstance(events[-1], (CompletedEvent, FailedEvent))

    # Check for expected steps
    running_events = [e for e in events if isinstance(e, RunningEvent)]
    assert len(running_events) > 0
```

### Event Type Assertions

```python
def assert_streaming_contract(events: list[StreamEvent]):
    """Verify streaming contract is followed."""
    # Must start with STARTING
    assert isinstance(events[0], StartingEvent)

    # Must end with COMPLETED or FAILED
    assert isinstance(events[-1], (CompletedEvent, FailedEvent))

    # COMPLETED must have output
    if isinstance(events[-1], CompletedEvent):
        assert events[-1].data.output is not None

    # FAILED must have error
    if isinstance(events[-1], FailedEvent):
        assert events[-1].data.error is not None
```

### Human Interrupt Testing

```python
@pytest.mark.asyncio
async def test_human_interrupt_flow():
    agent = MyAgent(config=MyAgentConfig(tools=[HumanTool()]))

    # First call - pause for human input
    human_event = None
    async for event in agent.astream(input_data):
        if isinstance(event, HumanInputRequiredEvent):
            human_event = event
            break

    assert human_event is not None
    assert human_event.run_context.messages is not None

    # Resume with human response
    resume_ctx = RunContext(
        run_id=human_event.run_context.run_id,
        messages=human_event.run_context.messages,
        human_response=HumanResponse(
            tool_call_id=human_event.data.tool_call_id,
            content={"response": "Test answer"},
        ),
    )

    got_response = False
    got_completed = False
    async for event in agent.astream(input_data, run_context=resume_ctx):
        if isinstance(event, HumanResponseEvent):
            got_response = True
        if isinstance(event, CompletedEvent):
            got_completed = True

    assert got_response
    assert got_completed
```

### Mock Event Consumer

```python
class StreamEventCollector:
    """Collect and categorize stream events for testing."""

    def __init__(self):
        self.events: list[StreamEvent] = []
        self.by_type: dict[str, list[StreamEvent]] = {}

    async def collect(self, stream: AsyncIterator[StreamEvent]):
        async for event in stream:
            self.events.append(event)
            self.by_type.setdefault(event.event_type, []).append(event)
        return self

    @property
    def output(self):
        completed = [e for e in self.events if isinstance(e, CompletedEvent)]
        return completed[0].data.output if completed else None

    @property
    def partials(self):
        return [e.data.partial_output for e in self.events if isinstance(e, PartialOutputEvent)]

# Usage
collector = StreamEventCollector()
await collector.collect(agent.astream(input_data))
assert collector.output is not None
```

---

## Exports

All streaming types are exported from `akd._base`:

```python
from akd._base import (
    # Core
    StreamEvent, StreamEventType, StreamingMixin,
    # Event data models
    StartingEventData, RunningEventData, CompletedEventData,
    FailedEventData, StreamingEventData, ThinkingEventData,
    PartialEventData, ToolCallingEventData, ToolResultEventData,
    HumanInputRequiredEventData, HumanResponseEventData, EventData,
    # Discriminated event subclasses
    StartingEvent, RunningEvent, CompletedEvent, FailedEvent,
    StreamingTokenEvent, ThinkingEvent, PartialOutputEvent,
    ToolCallingEvent, ToolResultEvent,
    HumanInputRequiredEvent, HumanResponseEvent,
    # Tool calling support
    ToolCall, ToolResult, RunContext, ToolCallingMixin, HumanResponse,
)
```

---

## Best Practices

1. **Always yield STARTING first** - Consumers expect it for initialization
2. **Always yield COMPLETED or FAILED last** - Never leave streams hanging (exception: `HUMAN_INPUT_REQUIRED` also ends the stream)
3. **Re-raise after FAILED** - Allow error propagation after event emission
4. **Use typed event subclasses** - Prefer `isinstance` checks with typed subclasses for type safety
5. **Use consistent step naming** - `parent.child` pattern (e.g., `research.search`)
6. **Include run_context in all events** - Enables filtering, correlation, and resumption
7. **Batch tokens for efficiency** - Default `token_batch_size=10` balances latency/overhead
8. **Emit PARTIAL progressively** - Let consumers see results as they become available
9. **Delegate `_arun()` to `_astream()`** - Single source of truth for execution logic
10. **Use RunContext for resumption** - Carry messages and human_response for pause/resume workflows
