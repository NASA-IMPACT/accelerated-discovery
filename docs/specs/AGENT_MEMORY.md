# Agent Memory

AKD agents use a session-based memory system for managing conversation history. Memory handles message storage, lifecycle management (stateless/stateful), automatic trimming, and integration with `RunContext` for pause/resume workflows.

## Memory Class

`Memory[T]` (`akd/_base/memory.py`) is a generic Pydantic model storing a list of messages.

```python
class Memory[T](BaseModel):
    messages: list[T] = Field(default_factory=list)
```

### Methods

| Method | Description |
|--------|-------------|
| `append(msg)` | Add a single message |
| `extend(items)` | Add multiple messages |
| `clear()` | Remove all messages |
| `all()` | Get a copy of all messages |
| `sync(items)` | Replace all messages with new list |
| `last()` | Get the last message (or `None`) |

Supports `len()`, `bool()`, iteration, and indexing.

### Initialization

Every `BaseAgent` initializes memory in `__init__()`:

```python
def __init__(self, config=None, memory=None, **kwargs):
    self.memory = memory or Memory()
```

Call `agent.reset_memory()` to clear all stored messages.

---

## Session Lifecycle

Memory operations happen inside session context managers. Both sync (`session()`) and async (`asession()`) variants exist with identical behavior.

```python
async with self.memory.asession(
    stateless=True,
    run_context=run_context,
    enable_trimming=True,
    model_name="gpt-4o",
    max_tokens=128000,
    trim_ratio=0.75,
) as messages:
    # messages is a reference to self.memory.messages
    # Mutations here directly modify memory
    messages.append({"role": "user", "content": "Hello"})
```

### Three Phases

```
SETUP (before yield)
├─ 1. If stateless=True: clear memory
├─ 2. If run_context.messages exists: sync into memory (replaces all messages)
└─ 3. If trimming enabled: trim messages to max_tokens * trim_ratio

EXECUTION (yield)
└─ Return reference to self.messages
   (caller mutates messages in place)

CLEANUP (after yield)
└─ If stateless=True: clear memory
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stateless` | `False` | Clear memory at start and end of session |
| `run_context` | `None` | `RunContext` with messages to sync |
| `enable_trimming` | `False` | Enable automatic message trimming |
| `model_name` | `None` | Model name for token counting during trimming |
| `max_tokens` | `None` | Maximum token budget for messages |
| `trim_ratio` | `0.75` | Target ratio after trimming (0.75 = use 75% of max) |

---

## Stateless vs Stateful

### Stateless (`stateless=True`, default)

Memory is cleared before **and** after each session. Each `astream()` / `_arun()` call starts fresh.

```python
agent = MyAgent(config=MyAgentConfig(stateless=True))

await agent.arun(input1)  # messages: [system, user, assistant] → cleared
await agent.arun(input2)  # messages: [system, user, assistant] → cleared
# agent.memory.messages == []
```

Use for: isolated single-turn requests, stateless API endpoints, independent tool executions.

### Stateful (`stateless=False`)

Memory persists across calls. Messages accumulate until explicitly cleared.

```python
agent = MyAgent(config=MyAgentConfig(stateless=False))

await agent.achat("My name is Alice")
# memory: [system, user("My name is Alice"), assistant("Nice to meet you")]

response = await agent.achat("What's my name?")
# memory includes prior messages → response.content mentions "Alice"

agent.reset_memory()  # Explicit clear
```

Use for: multi-turn conversations, chat interfaces, context-dependent follow-ups.

---

## Message Trimming

When `enable_trimming=True` (default in `BaseAgentConfig`), memory automatically trims messages during session setup using `litellm.utils.trim_messages`.

Trimming:
- Counts tokens using the model's tokenizer
- Preserves the system message and most recent messages
- Removes older messages to fit within `max_tokens * trim_ratio`
- Only triggers if `model_name` and `max_tokens` are provided

Configuration:

```python
config = BaseAgentConfig(
    max_tokens=128000,     # Model's input token limit
    trim_ratio=0.75,       # Keep 75% of max → 96,000 tokens
    enable_trimming=True,  # Default
)
```

---

## RunContext and Message Resumption

`RunContext` (`akd/_base/tool_calling.py`) carries conversation state for resumption:

```python
class RunContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    human_response: HumanResponse | None = None
    messages: list[dict[str, Any]] | None = None
    run_id: str | None = None
```

When `run_context.messages` is provided, they are synced into memory at session start (replacing current messages). This enables:

1. **Human interrupt resumption**: After `HUMAN_INPUT_REQUIRED`, the caller passes back the conversation history
2. **External state injection**: Pre-populate memory with messages from a database or prior session
3. **Cross-agent handoff**: Pass conversation context between agents

---

## Message Lifecycle in `_astream()`

The complete message flow within `LiteLLMInstructorBaseAgent._astream()` (`akd/agents/_base.py:1131`):

```
_astream(params, run_context)
│
├─ 1. Copy run_context (or create empty one)
├─ 2. Auto-generate run_id if not set
│
├─ 3. Yield STARTING event
│
├─ 4. Open memory session:
│     ├─ If stateless: clear memory
│     ├─ If run_context.messages: sync into memory
│     └─ If trimming: trim to budget
│
├─ 5. Add system message if messages list is empty
│     └─ {"role": "system", "content": system_prompt}
│
├─ 6. Add user message (SKIP if resuming with human_response)
│     └─ {"role": "user", "content": params.model_dump_json()}
│
├─ 7. Yield RUNNING event
│
├─ 8. Route based on tools:
│     ├─ HAS tools → _run_tool_loop(messages, run_context)
│     └─ NO tools  → _stream_llm_response(messages, run_context)
│
├─ 9. Yield events from inner streamer
│     ├─ If COMPLETED: capture output
│     └─ If HUMAN_INPUT_REQUIRED: return immediately
│
├─ 10. Append final assistant message
│      └─ {"role": "assistant", "content": output.model_dump_json()}
│
└─ 11. Session cleanup (stateless clear)
```

### In the Tool Loop (`_run_tool_loop`)

Messages are mutated in place during each iteration:

```
[Iteration Start]
│
├─ If human_response: inject tool result message
│   └─ {"role": "tool", "tool_call_id": "...", "content": "..."}
│
├─ LLM call with current messages + tool definitions
│
├─ After tool calls parsed:
│   ├─ Append assistant message:
│   │   {"role": "assistant", "content": "...", "tool_calls": [...]}
│   │
│   ├─ Execute tools → append results:
│   │   {"role": "tool", "tool_call_id": "...", "content": "..."}
│   │
│   └─ If reflection_prompt:
│       {"role": "user", "content": reflection_prompt}
│
[Loop to next iteration]
```

### In Direct Streaming (`_stream_llm_response`)

```
├─ If human_response: inject as user message
│   └─ {"role": "user", "content": "human's response"}
│
├─ LLM call with response_format (structured output, no tools)
│
└─ Parse and validate response
```

---

## Message Shapes

### Standard Messages

```python
# System message
{"role": "system", "content": "You are a research assistant..."}

# User message (input parameters serialized as JSON)
{"role": "user", "content": '{"query": "climate change research"}'}

# Assistant message (final output serialized as JSON)
{"role": "assistant", "content": '{"answer": "...", "results": [...]}'}
```

### Tool Calling Messages

```python
# Assistant message with tool calls
{
    "role": "assistant",
    "content": None,  # or reasoning text
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "search_papers",
                "arguments": '{"query": "quantum computing"}'
            }
        }
    ]
}

# Tool result message
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": '{"results": [...]}'
}
```

### Human Interaction Messages

```python
# ask_human tool call (in assistant message)
{
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call_human_456",
        "type": "function",
        "function": {
            "name": "ask_human",
            "arguments": '{"question": "What is your name?"}'
        }
    }]
}

# Human response (tool calling mode - injected as tool result)
{
    "role": "tool",
    "tool_call_id": "call_human_456",
    "content": '"Alice"'  # json.dumps("Alice")
}

# Human response (non-tool mode - injected as user message)
{
    "role": "user",
    "content": "Alice"
}

# Reflection prompt (if configured)
{"role": "user", "content": "Reflect on the results..."}
```

---

## RunContext Flow Through Events

In `_run_tool_loop()`, the run_context holds a **live reference** to messages:

```python
run_context.messages = messages  # Live reference, not a copy
```

Every `StreamEvent` emitted carries `run_context`. This means:
- Callers can access `event.run_context.messages` at any point during streaming
- The messages list reflects the current conversation state at that event
- On `HUMAN_INPUT_REQUIRED`, `run_context.messages` is explicitly set to a copy of messages for safe resumption

### Event → RunContext Access

```python
async for event in agent.astream(input_data):
    # All events carry run_context
    run_id = event.run_context.run_id

    if isinstance(event, HumanInputRequiredEvent):
        # Messages are a snapshot for resumption
        saved_messages = event.run_context.messages
        tool_call_id = event.data.tool_call_id

    if isinstance(event, ToolResultEvent):
        # Messages reflect state after tool execution
        current_messages = event.run_context.messages
```

---

## Complete Example: Stateful Multi-Turn with Human Interrupt

```python
from akd._base.structures import HumanResponse, RunContext
from akd._base.streaming import HumanInputRequiredEvent, HumanResponseEvent, CompletedEvent
from akd.tools.human import HumanTool

agent = MyAgent(config=MyAgentConfig(
    stateless=False,  # Persist memory across calls
    tools=[HumanTool()],
))

# Turn 1: Agent asks for human input
human_event = None
async for event in agent.astream(MyInput(query="Help me with research")):
    if isinstance(event, HumanInputRequiredEvent):
        human_event = event
        break

# Turn 2: Resume with human response
resume_ctx = RunContext(
    messages=human_event.run_context.messages,
    human_response=HumanResponse(
        tool_call_id=human_event.data.tool_call_id,
        content="Focus on quantum computing",
    ),
)

async for event in agent.astream(MyInput(query="Help me with research"), run_context=resume_ctx):
    if isinstance(event, CompletedEvent):
        result = event.output

# Turn 3: Follow-up (memory persists since stateless=False)
async for event in agent.astream(MyInput(query="Can you elaborate on the third point?")):
    if isinstance(event, CompletedEvent):
        followup = event.output
        # Agent has context from turns 1 and 2
```
