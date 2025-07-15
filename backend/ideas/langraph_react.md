# LangGraph <> React Communication Guide

This guide explains how to connect any LangGraph-based multi-agent framework (like AKD) to a stateful React frontend where each agent has UI elements representing their execution state and progress.

## 1. Frontend: React useStream Hook

The `useStream()` hook from `@langchain/langgraph-sdk/react` serves as the central communication layer:

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";

const thread = useStream<{
  messages: Message[];
  initial_search_query_count: number;
  max_research_loops: number;
  reasoning_model: string;
}>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
  onUpdateEvent: (event) => { /* Process real-time updates */ },
  onError: (error) => { /* Handle errors */ },
});
```

## 2. Backend: LangGraph State Management

The backend uses TypedDict classes to define structured state:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
import operator

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_status: Annotated[dict, operator.add]  # Track each agent's status
    task_results: Annotated[list, operator.add]
    agent_outputs: Annotated[dict, operator.add]  # Outputs from each agent
    workflow_config: dict  # Configuration parameters
    current_step: str  # Current workflow step
    agent_progress: dict  # Progress tracking per agent
```

## 3. Frontend → Backend Communication

### 3.1 Message Submission

When users submit queries, the frontend sends structured data to the backend:

```typescript
const handleSubmit = (userInput: string, config: WorkflowConfig) => {
  const newMessages: Message[] = [
    ...(thread.messages || []),
    {
      type: "human",
      content: userInput,
      id: Date.now().toString(),
    },
  ];
  
  thread.submit({
    messages: newMessages,
    workflow_config: config,
    agent_status: {},  // Initialize agent status tracking
    current_step: "start",
    agent_progress: {},
  });
};
```

### 3.2 State Configuration

The frontend can dynamically configure backend behavior by passing workflow configuration, agent parameters, and UI state preferences to the `thread.submit()` method:

```typescript
interface WorkflowConfig {
  agents: string[];  // Which agents to activate
  maxIterations: number;
  parallelExecution: boolean;
  uiUpdateFrequency: number;
  debugMode: boolean;
}
```

### 3.3 Thread Management

- **Thread Lifecycle**: `thread.submit()`, `thread.stop()`, `thread.messages`, `thread.isLoading`
- **Thread State**: `thread.state` - access to current workflow state
- **Thread Events**: `thread.onUpdateEvent`, `thread.onCustomEvent`, `thread.onError`

## 4. Backend → Frontend Communication

### 4.1 Event Streaming Protocol

The backend streams structured events for different agents and workflow phases:

```typescript
onUpdateEvent: (event: any) => {
  let processedEvent: AgentEvent | null = null;
  
  // Handle different agent events
  if (event.agent_started) {
    processedEvent = {
      agentId: event.agent_started.agent_id,
      title: `${event.agent_started.agent_name} Started`,
      status: "running",
      data: event.agent_started.initial_state,
      timestamp: Date.now(),
    };
  } else if (event.agent_progress) {
    processedEvent = {
      agentId: event.agent_progress.agent_id,
      title: `${event.agent_progress.agent_name} Progress`,
      status: "running",
      data: event.agent_progress.progress_data,
      timestamp: Date.now(),
    };
  } else if (event.agent_completed) {
    processedEvent = {
      agentId: event.agent_completed.agent_id,
      title: `${event.agent_completed.agent_name} Completed`,
      status: "completed",
      data: event.agent_completed.results,
      timestamp: Date.now(),
    };
  }
  
  // Update agent-specific UI elements
  updateAgentUI(processedEvent);
}
```

### 4.2 LangGraph Node Execution

Each agent node emits events when executed:

```python
def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    agent_id = config.get("agent_id", "unknown")
    
    # Emit agent started event
    emit_event("agent_started", {
        "agent_id": agent_id,
        "agent_name": config.get("agent_name"),
        "initial_state": state.get("current_step")
    })
    
    # Perform agent work
    result = agent_logic(state, config)
    
    # Emit progress updates
    emit_event("agent_progress", {
        "agent_id": agent_id,
        "progress_data": result.progress
    })
    
    # Emit completion event
    emit_event("agent_completed", {
        "agent_id": agent_id,
        "results": result.output
    })
    
    return {
        "agent_outputs": {agent_id: result.output},
        "agent_status": {agent_id: "completed"},
        "task_results": [result.output]
    }
```

How this event is handled in the frontend:

```typescript
interface AgentEvent {
  agentId: string;
  title: string;
  status: 'running' | 'completed' | 'error';
  data: any;
  timestamp: number;
}

function updateAgentUI(event: AgentEvent): void {
  // Update agent-specific UI component
  setAgentStates(prev => ({
    ...prev,
    [event.agentId]: {
      ...prev[event.agentId],
      status: event.status,
      lastUpdate: event.timestamp,
      currentData: event.data,
      title: event.title
    }
  }));
  
  // Update global timeline
  setGlobalTimeline(prev => [...prev, event]);
}
```

## 5. Backend State Management

- **State Graph**: LangGraph manages workflow state transitions between agents.
- **Message Handling**: `add_messages` annotation aggregates messages from all agents. This is used to track the conversation history for example.
- **Agent State Tracking**: Each agent's state is maintained separately within the overall workflow state.

## 6. Thread Management

```typescript
// Thread automatically created on first submission
const thread = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  // Thread ID auto-generated or can be specified
});

// Thread operations
thread.submit(newState);    // Send new state
thread.stop();              // Cancel execution
thread.messages;            // Access message history
thread.isLoading;           // Check execution status
```

- **Database Storage**: Thread state persisted with agent-specific data
- **Recovery**: Automatic thread resumption after disconnection
- **Caching**: Local caching for immediate display of agent states
- **Agent State Recovery**: Individual agent states are restored on reconnection

## 7. Error Handling and Recovery

```typescript
const thread = useStream({
  // ... other config
  onError: (error) => {
    console.error("Stream error:", error);
    // Automatic reconnection handled by SDK
  },
});
```

- **Stream Errors**: Backend errors automatically streamed to frontend with agent context
- **Graceful Degradation**: Partial results displayed during errors, other agents continue
- **Retry Mechanisms**: Built-in connection retry logic
- **Agent-Specific Errors**: Errors are isolated to specific agents without affecting the entire workflow