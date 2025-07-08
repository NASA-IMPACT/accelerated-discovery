### Problem Statement

Currently, our agentic literature search agent relies on LangGraph for node structuring. However, we aim to allow users to integrate their own tools and agents, implemented in any framework, into a global tool pool. To ensure flexibility, we need a standardized node template that is independent of LangGraph but can still be wrapped within a LangGraph node.

Each node should adhere to a structured design that includes:
- A well-defined state (pydantic types / base model with accessible keys)
  - Node State
  - Global State
- Input/Ooutput guardrails for validation
  - Input Guardrails will validate the input to the node
  - Output guardrails wil lvalidate the output from the node supervisor
- A node supervisor (which runs the core logic, and could be custom logic or LLM-based agent)
  - A supervisor that only has visibility over the node’s tool subset
- A subset of tools accessible within the node
- Global tool calls
- IO Checks and Validation
  - Example: Relevancy Checker, Reflection Agennt Tool
  - Execute consistently across nodes

This structure effectively creates a small internal graph within each node and should be uniformly applied across all nodes.

### Proposed Solution

Introduce a standardized, LangGraph-agnostic node template class `NodeTemplate` that can be wrapped into LangGraph nodes but remains independent. This template should encapsulate all necessary components and enforce a consistent execution pipeline across nodes mentioned in the problem statement.

Current Implementation is at `akd/nodes`
- `akd/nodes/states.py` : Has state related data structure
- `akd/nodes/supervisor.py` : Contains implementation for supervisor
  - `akd.nodes.supervisor.BaseSupervisor` defines the base abstract class
  - `akd.nodes.supervisor.LLMSupervisor` defines the type for LLM-based supervision
    - contains tool binding for the model that can automatically call tools
    - `akd.nodes.supervisor.ReActLLMSupervisor` implements ReAct-based agentic flow
- `akd/nodes/templates.py`: Contains implementation for node template
  - `akd.nodes.templates.AbstractNodeTemplate` provides the abstraction for node template
  - `akd.nodes.templates.DefaultNodeTemplate` is the default implementation that:
    - Runs Input/Output guardrails
    - Runs supervisor
  - The node templates have `to_langgraph_node(key=...)` method that automatically converts the template object to langgraph-compatible node.

Example:
```python
from akd import NodeTemplate

node_t = NodeTemplate(
    supervisor=ReActLLMSupervisor(...)  # LLM-based or custom logic
    input_guardrails=[],  # Input validation rules
    output_guardrails=[],  # Output validation rules
)

node_lg = node.as_langgraph_node(...)
```

This approach ensures that node logic remains decoupled from LangGraph, enabling portability across different orchestration frameworks.


### Benefits

- **Framework Agnosticism:** Users can bring their own tool implementations without being restricted to LangGraph.
- **Reusability:** A standardized approach enables reuse across different workflows and projects.
- **Modularity:** Nodes remain modular, making it easier to manage dependencies and updates.
- **Consistency:** Ensures that all nodes follow a uniform structure, reducing maintenance overhead.



This aims to create a flexible and maintainable architecture for our agentic literature search system while fostering interoperability across different tool frameworks. This design allows easy extension for future enhancements.
