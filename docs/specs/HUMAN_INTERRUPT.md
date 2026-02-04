# Human Interrupt

AKD agents support pausing execution to request human input and resuming with the response. This enables human-in-the-loop workflows where agents can ask for clarification, approval, or additional information during execution.

## Interaction Modes

Human interaction works through two primary interfaces:

1. **Streaming (`astream()`)** - Real-time event-based interaction with two sub-modes:
   - **Tool calling mode**: Agent calls `ask_human` tool, pauses, resumes with tool result
   - **Non-tool mode**: Human response injected as user message into direct LLM streaming
2. **Freeform chat (`achat()`)** - Simple text-based conversation that routes through `_arun()`

---

## Mode 1: Streaming with Tool Calling (ask_human)

When an agent has tools configured (including `HumanTool`), the LLM can call `ask_human` to request human input. The tool loop intercepts this call **before execution** and pauses the stream.

### Flow

```
astream(input_data)
    │
    ▼
LLM generates tool call: ask_human(question="What is your name?")
    │
    ▼
Tool loop intercepts (does NOT execute HumanTool._arun)
    ├─ Adds assistant message with tool_calls to messages
    ├─ Stores messages in run_context for resumption
    ├─ Yields HUMAN_INPUT_REQUIRED event
    └─ Returns (stream ends gracefully)

    ─── Caller collects human input ───

astream(input_data, run_context=resume_context)
    │
    ▼
Tool loop checks run_context.human_response
    ├─ Injects response as tool result: {"role": "tool", "tool_call_id": ..., "content": ...}
    ├─ Yields HUMAN_RESPONSE event
    └─ Continues ReAct loop → eventually COMPLETED
```

### Pause: HUMAN_INPUT_REQUIRED Event

When the LLM calls `ask_human`, the tool loop:

1. Parses `HumanToolInput` from tool call arguments
2. Appends the assistant message (with `tool_calls`) to the conversation
3. Stores the full message list in `run_context.messages`
4. Yields `HumanInputRequiredEvent` with:
   - `data.human_input`: The `HumanToolInput` object (question, context, options)
   - `data.tool_call_id`: ID to link the response back
   - `data.tool_name`: `"ask_human"`
   - `run_context.messages`: Full conversation history for resumption
5. Returns from the generator (stream ends)

### Resume: Constructing RunContext

To resume, construct a `RunContext` with:

```python
from akd._base.tool_calling import HumanResponse, RunContext

resume_context = RunContext(
    run_id=event.run_context.run_id,
    messages=event.run_context.messages,
    human_response=HumanResponse(
        tool_call_id=event.data.tool_call_id,
        content={"response": "The human's answer"},  # Dict matching HumanToolOutput
    ),
)
```

The `content` field must be a dict matching `HumanToolOutput` schema: `{"response": "..."}`. This is because the response is injected as a tool result message and the LLM expects it in the schema format.

### Resume: Processing

When `astream()` is called with a `RunContext` containing `human_response`:

1. Memory session syncs `run_context.messages` into memory
2. User message is **skipped** (not appended, since we're resuming)
3. Tool loop detects `human_response` at the top:
   - Appends tool result: `{"role": "tool", "tool_call_id": ..., "content": json.dumps(content)}`
   - Yields `HumanResponseEvent`
4. Tool loop continues from where it left off

### Complete Example

```python
from akd._base.streaming import CompletedEvent, HumanInputRequiredEvent, HumanResponseEvent
from akd._base.tool_calling import HumanResponse, RunContext
from akd.tools.human import HumanTool

# Agent with HumanTool
agent = MyAgent(config=MyAgentConfig(
    tools=[HumanTool()],
    system_prompt="Ask the human for their name before responding.",
))

input_data = MyInputSchema(query="Hello, who am I?")

# First call - pauses for human input
human_event = None
async for event in agent.astream(input_data):
    if isinstance(event, HumanInputRequiredEvent):
        human_event = event
        break  # Stream ends after this event

assert human_event is not None
print(f"Agent asks: {human_event.data.human_input.question}")

# Collect human input (e.g., from UI, CLI, API)
user_answer = input("Your answer: ")

# Resume with human response
resume_context = RunContext(
    run_id=human_event.run_context.run_id,
    messages=human_event.run_context.messages,
    human_response=HumanResponse(
        tool_call_id=human_event.data.tool_call_id,
        content={"response": user_answer},
    ),
)

async for event in agent.astream(input_data, run_context=resume_context):
    if isinstance(event, HumanResponseEvent):
        print("Agent received human input")
    if isinstance(event, CompletedEvent):
        print(f"Result: {event.output}")
```

---

## Mode 2: Streaming Without Tool Calling

When an agent has **no tools** configured, it uses `_stream_llm_response()` for direct LLM streaming. Human responses can still be provided via `RunContext.human_response`, but they are injected differently.

### Flow

```
astream(input_data, run_context=RunContext(human_response=...))
    │
    ▼
_stream_llm_response()
    ├─ Checks run_context.human_response
    ├─ Injects response as USER message: {"role": "user", "content": "..."}
    ├─ Yields HUMAN_RESPONSE event
    └─ Continues LLM streaming → COMPLETED
```

### Key Difference: User Message vs Tool Result

| Aspect | Tool Calling Mode | Non-Tool Mode |
|--------|------------------|---------------|
| Message role | `"tool"` | `"user"` |
| Content format | `json.dumps({"response": "..."})` | Plain string or `json.dumps(content)` |
| Trigger | LLM calls `ask_human` | Caller provides `human_response` directly |
| Interception | Tool loop intercepts before execution | Checked at start of `_stream_llm_response()` |

### Non-Tool Example

```python
# Agent without tools - resuming with prior conversation
run_context = RunContext(
    run_id="test",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "What do you need?"},
    ],
    human_response=HumanResponse(
        tool_call_id="msg_123",
        content="Just confirm you received this",  # Plain string for non-tool
    ),
)

async for event in agent.astream(input_data, run_context=run_context):
    if isinstance(event, HumanResponseEvent):
        print("Human response injected as user message")
    if isinstance(event, CompletedEvent):
        print(f"Result: {event.output}")
```

---

## Freeform Chat (achat)

`achat()` provides a simple text-based interface on any agent. It wraps input in `TextInput`, calls `_arun()`, and returns `TextOutput`.

```python
response = await agent.achat("What is quantum computing?")
print(response.content)
```

`achat()` accepts an optional `run_context` for resumption after human input:

```python
response = await agent.achat("Continue", run_context=resume_context)
```

This routes through the standard `_arun()` pipeline, which uses the same memory session and tool loop/streaming path as `astream()`.

---

## Data Structures

### HumanTool (`akd/tools/human.py`)

```python
class HumanToolInput(InputSchema):
    question: str                     # Question to present to the human
    context: str | None = None        # Additional context
    options: list[str] | None = None  # Optional choices

class HumanToolOutput(OutputSchema):
    response: str                     # The human's response

class HumanTool(BaseTool[HumanToolInput, HumanToolOutput]):
    # _arun() raises HumanInputRequired (intercepted by tool loop)
```

### HumanResponse (`akd/_base/tool_calling.py`)

```python
class HumanResponse(ToolResult):
    tool_name: str = "ask_human"
    # Inherited: tool_call_id, content, timestamp, error
```

### HumanInputRequired (`akd/_base/errors.py`)

```python
class HumanInputRequired(ToolError):
    prompt: str
    tool_call_id: str | None
    context: dict | None
    options: list[str] | None
```

### RunContext (`akd/_base/tool_calling.py`)

```python
class RunContext(BaseModel):
    model_config = ConfigDict(extra="allow")  # Accepts arbitrary extra keys

    human_response: HumanResponse | None = None  # For resumption
    messages: list[dict[str, Any]] | None = None  # Conversation history
    run_id: str | None = None                     # Execution run ID
```

### Streaming Events (`akd/_base/streaming.py`)

```python
class HumanInputRequiredEventData(BaseModel):
    human_input: Any       # HumanToolInput object
    tool_call_id: str      # For linking response
    tool_name: str         # "ask_human"

class HumanResponseEventData(BaseModel):
    tool_call_id: str      # Links to original request
    response: Any          # The human's response content

# Discriminated event subclasses
class HumanInputRequiredEvent(StreamEvent):
    event_type: Literal[StreamEventType.HUMAN_INPUT_REQUIRED]
    data: HumanInputRequiredEventData

class HumanResponseEvent(StreamEvent):
    event_type: Literal[StreamEventType.HUMAN_RESPONSE]
    data: HumanResponseEventData
```

---

## User Message Skip on Resume

When resuming with `human_response`, the agent skips adding a new user message to avoid duplicating input. In `_astream()`:

```python
# Add user message (skip if resuming with human response)
if params and not (run_context and run_context.human_response):
    messages.append({"role": "user", "content": params.model_dump_json(...)})
```

This is because the conversation history from `run_context.messages` already contains the original user message.

---

## Message Format Summary

### Tool Calling Mode

After `ask_human` interception, the messages look like:

```
[system] System prompt
[user]   Original query
[assistant] { tool_calls: [{ name: "ask_human", arguments: { question: "..." } }] }
  ── pause ──
[tool]   { "response": "human's answer" }  ← injected on resume
[assistant] Final response using human's input
```

### Non-Tool Mode

With `human_response` in non-tool streaming:

```
[system] System prompt
[user]   Original query
[assistant] Previous response
  ── pause ──
[user]   "human's answer"  ← injected as user message on resume
[assistant] New response
```

---

## Convenience Properties

`StreamEvent` provides shortcut access for human interaction events:

```python
event.human_prompt    # HUMAN_INPUT_REQUIRED: question string
event.message_history # run_context.messages for resumption
```
