# Tool Calling

Tool calling in AKD enables agents to invoke external functions during execution. The system implements a ReAct-style loop where the LLM generates tool calls, tools execute in parallel, and results feed back into the conversation until the agent produces a final answer.

## Core Data Structures

### ToolCall

Normalized tool call request, aligned with Pydantic AI's `ToolCallPart`. Defined in `akd/_base/tool_calling.py`.

```python
class ToolCall(BaseModel):
    tool_call_id: str   # Unique identifier for this call
    tool_name: str      # Name of tool to call
    arguments: dict[str, Any]  # Input arguments as dict
```

### ToolResult

Tool execution result, aligned with Pydantic AI's `ToolReturnPart`.

```python
class ToolResult(BaseModel):
    tool_call_id: str        # Links to original ToolCall
    tool_name: str           # Name of tool that was called
    content: Any             # Return value from tool (JSON-serializable)
    timestamp: datetime      # UTC timestamp
    error: str | None = None # Error message if failed
```

### HumanResponse

Extends `ToolResult`. Represents a human's response to the `ask_human` tool.

```python
class HumanResponse(ToolResult):
    tool_name: str = "ask_human"  # Defaults to ask_human
```

---

## BaseTool

All tools extend `BaseTool[InSchema, OutSchema]` (`akd/tools/_base.py`). Each tool has:

- **`input_schema`**: Pydantic model inheriting `InputSchema` - defines accepted parameters
- **`output_schema`**: Pydantic model inheriting `OutputSchema` - defines return shape
- **`config_schema`**: `BaseToolConfig` subclass with `name`, `description`, `debug`
- **`_arun(params)`**: Abstract async method implementing the tool's logic

### Tool Definition Format

`as_tool_definition()` converts a tool to OpenAI-compatible function calling format:

```python
{
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search for academic papers...",
        "parameters": {
            # JSON Schema from input_schema.model_json_schema()
            "type": "object",
            "properties": { ... },
            "required": [ ... ]
        }
    }
}
```

### Creating a Tool

```python
from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig

class MyToolInput(InputSchema):
    """Input for my tool."""
    query: str

class MyToolOutput(OutputSchema):
    """Output from my tool."""
    result: str

class MyTool(BaseTool[MyToolInput, MyToolOutput]):
    input_schema = MyToolInput
    output_schema = MyToolOutput

    async def _arun(self, params: MyToolInput, **kwargs) -> MyToolOutput:
        return MyToolOutput(result=f"Processed: {params.query}")
```

---

## ToolCallingMixin

`ToolCallingMixin` (`akd/_base/tool_calling.py`) provides reusable tool execution helpers to any class with a `self.tools` list.

### `_find_tool(name, tools=None)`

Finds a tool by name. Checks both `tool.name` and the class name.

### `_execute_tool(tool_call, tools=None)`

Executes a single tool call:
1. Finds tool by name via `_find_tool()`
2. Validates input: `tool.input_schema(**tool_call.arguments)`
3. Calls `tool.arun(input_obj)`
4. Returns `ToolResult` with `content=result.model_dump(mode="json")` or `error=str(e)`

### `_execute_tools_parallel(tool_calls, tools=None)`

Executes multiple tool calls concurrently using `asyncio.gather()`. When an LLM returns multiple tool calls in a single response, they are independent by definition and can run in parallel.

---

## Special Tools

### OutputTool (`final_answer`)

Implements the Pydantic AI pattern for structured final answers (`akd/tools/output.py`).

- Dynamically created with the agent's `output_schema` as both input and output schema
- Named `"final_answer"` - signals completion when called
- Flattens JSON Schema (removes `$defs`) for better LLM compatibility
- When called, returns validated input as the final output

The agent creates an `OutputTool` at the start of each tool loop iteration and adds it alongside regular tools. When the LLM calls `final_answer`, the tool loop knows the agent is done.

### HumanTool (`ask_human`)

Requests human input during agent execution (`akd/tools/human.py`).

```python
class HumanToolInput(InputSchema):
    question: str                    # Question to present to human
    context: str | None = None       # Additional context
    options: list[str] | None = None # Optional choices

class HumanToolOutput(OutputSchema):
    response: str                    # The human's response
```

The `_arun()` method raises `HumanInputRequired` exception. This exception is never actually caught by the tool executor - the tool loop intercepts `ask_human` calls **before** execution and emits a `HUMAN_INPUT_REQUIRED` streaming event instead.

See [HUMAN_INTERRUPT.md](./HUMAN_INTERRUPT.md) for the full human interaction lifecycle.

---

## The Tool Loop (ReAct Pattern)

`_run_tool_loop()` in `akd/agents/_base.py:790` implements the agent's ReAct loop. It runs for up to `max_tool_iterations` turns.

### Per-Iteration Flow

```
1. Call LLM with tools (streaming)
   ├─ Accumulate content tokens → STREAMING events (batched)
   ├─ Accumulate reasoning tokens → THINKING events
   ├─ Accumulate tool calls across chunks (streamed arguments)
   └─ Buffer partial output validation (PARTIAL events)

2. Post-streaming: Check for tool calls
   ├─ NO tool calls + content:
   │   ├─ Try JSON parse → if valid output schema, treat as COMPLETED (fallback)
   │   └─ If invalid: append as assistant reasoning, nudge to use tools, retry
   ├─ NO tool calls + no content:
   │   └─ Raise UnexpectedModelBehavior
   └─ HAS tool calls:
       └─ Continue to step 3

3. Convert accumulated chunks → ToolCall objects
   └─ Emit TOOL_CALLING event per tool call

4. Check for special tools
   ├─ ask_human: Intercept BEFORE execution
   │   ├─ Add assistant message with tool_calls to messages
   │   ├─ Store messages in run_context for resumption
   │   ├─ Yield HUMAN_INPUT_REQUIRED event
   │   └─ Return (end generator gracefully)
   └─ Continue to step 5

5. Check max_tool_calls limit
   └─ Raise MaxToolCallsExceeded if exceeded

6. Execute all tools in parallel
   └─ _execute_tools_parallel(tool_calls, all_tool_instances)

7. Check for final_answer tool result
   ├─ If found: Yield COMPLETED event → return
   └─ If not: Continue to step 8

8. Emit TOOL_RESULT events for regular tools

9. Update message history
   ├─ Append assistant message (content + tool_calls)
   ├─ Append tool result messages
   └─ Inject reflection_prompt if configured

10. Loop back to step 1
```

### Resumption After Human Input

If the previous call ended with `HUMAN_INPUT_REQUIRED`, the caller resumes with a `RunContext` containing `human_response`. At the top of the tool loop:

1. Check `run_context.human_response`
2. Inject response as tool result message: `{"role": "tool", "tool_call_id": ..., "content": json.dumps(content)}`
3. Yield `HUMAN_RESPONSE` event
4. Continue the loop

### JSON Fallback Pattern

When the LLM responds with text instead of calling a tool, the loop attempts to parse it as valid JSON matching the output schema. If successful, it's treated as the final answer. If not, the text is appended as an assistant message and a validation nudge is injected:

```
"Validation feedback: Please call one of the provided tool functions instead."
```

This follows the Pydantic AI pattern for handling models that give direct structured output without tool calls.

---

## Configuration

Tool calling behavior is controlled by `BaseAgentConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `tools` | `[]` | List of `BaseTool` instances available to the agent |
| `max_tool_iterations` | `25` | Maximum ReAct loop turns (1-50) |
| `max_tool_calls` | `None` | Maximum total tool calls across the run (None = unlimited) |
| `reflection_prompt` | `None` | If set, injected after tool results to force reasoning before next iteration |

### Reflection Prompt

When `reflection_prompt` is set, the prompt is injected as a user message after every tool result, before the next LLM call. This forces the model to reason about the results before deciding its next action.

After tool results are appended to messages, the tool loop appends:

```python
{"role": "user", "content": reflection_prompt}
```

and emits a `ThinkingEvent` with `data.reflection_prompt` set. The LLM then sees the reflection prompt and must reason before calling the next tool or producing a final answer.

```python
agent = SearchAgent(BaseAgentConfig(
    tools=[SerperSearchTool(), HumanTool()],
    reflection_prompt="""Evaluate the results so far:
1. Did you get enough information to answer the query?
2. Are there gaps that need another search?
3. If a human responded, was their answer clear enough to proceed?

Only call final_answer when you have sufficient evidence.""",
))
```

---

## Streaming Events

Tool calling emits these event types during execution:

| Event Type | Typed Class | When | Data |
|------------|-------------|------|------|
| `TOOL_CALLING` | `ToolCallingEvent` | Before tool execution | `ToolCallingEventData(tool_call=ToolCall)` |
| `TOOL_RESULT` | `ToolResultEvent` | After tool execution | `ToolResultEventData(result=ToolResult)` |
| `HUMAN_INPUT_REQUIRED` | `HumanInputRequiredEvent` | `ask_human` intercepted | `HumanInputRequiredEventData(human_input, tool_call_id, tool_name)` |
| `HUMAN_RESPONSE` | `HumanResponseEvent` | Resumed with human input | `HumanResponseEventData(tool_call_id, response)` |

Additionally, throughout the tool loop: `STREAMING` (raw tokens), `THINKING` (reasoning tokens), `PARTIAL` (validated partial output), `COMPLETED` (final answer), `FAILED` (error).

---

## Error Handling

| Exception | When Raised |
|-----------|-------------|
| `MaxToolIterationsExceeded` | Agent exceeds `max_tool_iterations` without producing final answer |
| `MaxToolCallsExceeded` | Total tool calls exceed `max_tool_calls` limit |
| `UnexpectedModelBehavior` | LLM returns empty response without tool calls |
| `HumanInputRequired` | `HumanTool._arun()` called (intercepted before this in practice) |

All exceptions inherit from `AKDError` via `AgentError` or `ToolError` (`akd/_base/errors.py`).

---

## Usage Examples

### Defining the Agent

```python
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.tools.search.serper import SerperSearchTool
from akd.tools import HumanTool


class SearchAgentInputSchema(InputSchema):
    """Input schema for SearchAgent."""
    query: str


class SearchAgentOutputSchema(OutputSchema):
    """Output schema for SearchAgent."""
    response: str = Field(..., description="The final answer in markdown format with references")


class SearchAgent(LiteLLMInstructorBaseAgent):
    input_schema = SearchAgentInputSchema
    output_schema = SearchAgentOutputSchema
```

### Usage 1: Agentic Tool Calling (No Human Tool)

Agent autonomously searches and produces a final answer via the `final_answer` tool.

```python
from akd._base.streaming import CompletedEvent, ToolCallingEvent, ToolResultEvent

agent = SearchAgent(BaseAgentConfig(
    tools=[SerperSearchTool()],
    model_name="gpt-4o-mini",
    max_tool_iterations=5,
))

async for event in agent.astream(SearchAgentInputSchema(query="quantum computing breakthroughs 2024")):
    if isinstance(event, ToolCallingEvent):
        print(f"Calling: {event.data.tool_call.tool_name}")
    elif isinstance(event, ToolResultEvent):
        print(f"Got result from: {event.data.result.tool_name}")
    elif isinstance(event, CompletedEvent):
        print(event.data.output.response)
```

### Usage 2: Agentic Tool Calling with Human Interrupt

Agent has both `SerperSearchTool` and `HumanTool`. When it encounters ambiguity, it calls `ask_human` to clarify before searching.

```python
from akd._base.streaming import CompletedEvent, HumanInputRequiredEvent, HumanResponseEvent
from akd._base.tool_calling import HumanResponse, RunContext

agent = SearchAgent(BaseAgentConfig(
    tools=[SerperSearchTool(), HumanTool()],
    system_prompt="If the query is ambiguous, ask the human for clarification before searching.",
    model_name="gpt-4o-mini",
    max_tool_iterations=5,
))

input_data = SearchAgentInputSchema(query="latest transformer research")

# First call - agent may ask for clarification
human_event = None
async for event in agent.astream(input_data):
    if isinstance(event, HumanInputRequiredEvent):
        human_event = event
        break
    if isinstance(event, CompletedEvent):
        print(event.data.output.response)

# If agent asked for human input, resume with response
if human_event:
    print(f"Agent asks: {human_event.data.human_input.question}")

    resume_context = RunContext(
        messages=human_event.run_context.messages,
        human_response=HumanResponse(
            tool_call_id=human_event.data.tool_call_id,
            content="I mean ML transformers like BERT and GPT",
        ),
    )

    async for event in agent.astream(input_data, run_context=resume_context):
        if isinstance(event, HumanResponseEvent):
            print("Agent received clarification")
        if isinstance(event, CompletedEvent):
            print(event.data.output.response)
```

---

## Complete Lifecycle Diagram

```
User Input
    │
    ▼
_astream()
    │
    ├─ Memory session (clear/sync/trim)
    ├─ Add system message (if empty)
    ├─ Add user message (skip if resuming with human_response)
    │
    ├─ Has tools? ─── NO ──→ _stream_llm_response() (no tool calling)
    │       │
    │      YES
    │       │
    │       ▼
    │   _run_tool_loop()
    │       │
    │       ├─ [If human_response] Inject as tool result → HUMAN_RESPONSE event
    │       │
    │       ▼
    │   ┌─────────────────────────────────┐
    │   │  ReAct Loop (max_tool_iterations)│
    │   │                                  │
    │   │  LLM call (streaming)            │
    │   │    → STREAMING, THINKING events  │
    │   │                                  │
    │   │  Tool calls?                     │
    │   │    ├─ ask_human → HUMAN_INPUT_REQUIRED → return
    │   │    ├─ final_answer → COMPLETED → return
    │   │    ├─ regular → execute parallel  │
    │   │    │   → TOOL_CALLING events      │
    │   │    │   → TOOL_RESULT events       │
    │   │    │   → append to messages       │
    │   │    │   → [reflection_prompt]      │
    │   │    │   → loop                     │
    │   │    └─ none → JSON fallback or retry
    │   └─────────────────────────────────┘
    │
    ├─ Append final output to messages
    └─ Session cleanup
```
