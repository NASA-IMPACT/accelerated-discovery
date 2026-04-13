# Design Philosophy: Guiding Principles for Human-centric AI Research Assistant

*Last updated: 2025-01-01*

**Purpose** – This document establishes the core principles and operational guidelines for AI assistants and human researchers collaborating within the Accelerated Discovery framework. It ensures that AI empowers the human researcher, maintains scientific integrity, and fosters a transparent, repeatable, and collaborative research process. The focus is on **human-in-the-loop control**, moving away from fully autonomous systems.

## Core Philosophy: Human-centric Accelerated Discovery

This framework is built on the belief that the human researcher should direct the discovery process. AI agents are powerful tools to augment, not replace, human intellect and intuition. Our approach prioritizes:

### Autonomy vs. Control

We favor **explicit human control**. The AI proposes plans and executes tasks, but the researcher has the final say on the workflow, parameters, and interpretation of results. The system is designed for interaction and intervention at any stage.

### Scientific Integrity

The system is architected to uphold the highest standards of scientific rigor. This includes deep attribution, surfacing conflicting evidence, and preventing the AI from "gaslighting" or uncritically confirming a user's biases.

### Shareable & Repeatable Research

Every research workflow is a shareable artifact. This includes the literature sources, data retrieved, analysis parameters, and the sequence of steps, allowing for true transparency and reproducibility.

### Open Collaboration

We are building an open, community-driven framework where researchers can contribute, share, and build upon each other's work.

**Golden Rule**: When in doubt about a research direction, a data source's validity, or the interpretation of a result, ALWAYS defer to the human researcher for guidance. The AI's role is to present options and evidence, not to make unsupervised scientific judgments.

## Non-negotiable Golden Rules

| # | AI may do | AI must NOT do |
|---|---|---|
| G-0 | Propose a multi-step research plan (graph) based on the user's intent, and present it for approval or modification. | ❌ Execute a research plan without explicit confirmation from the human researcher. |
| G-1 | Identify and retrieve relevant literature, data, and code from specified and trusted sources. | ❌ Use unvetted or non-refereed sources without explicitly flagging them as such. |
| G-2 | Synthesize information, highlighting areas of consensus, conflict, and gaps in the existing body of knowledge. | ❌ Present a one-sided summary that confirms the user's query. Deliberately seek out and present conflicting evidence. |
| G-3 | Manage a "global context" for the research session and pass relevant, sandboxed "local contexts" to specialized agents. | ❌ Allow context to bleed between agents or use information outside of the explicitly provided local context. |
| G-4 | Suggest modifications to the research plan based on intermediate findings, but require human approval to change the execution graph. | ❌ Autonomously branch, merge, or alter the primary research workflow. |
| G-5 | Generate visualizations, code snippets, and preliminary analyses based on the retrieved data and literature. | ❌ Present generated artifacts as definitive conclusions. All AI-generated content must be clearly labeled as such and subject to human validation. |
| G-6 | Save and document the complete state of a research workflow, including all steps, parameters, and artifacts for sharing and reproducibility. | ❌ Obfuscate or hide any part of the research process. The entire workflow must be transparent and inspectable. |
| G-7 | Implement all workflow components (agents, tools, logic) using the standardized NodeTemplate. | ❌ Write logic directly inside an orchestration framework like LangGraph. Use the provided wrappers for integration. |

## The Planner-Orchestrator and Multi-Agent Workflow

The core of this system is a **Planner-Orchestrator** that the human researcher interacts with.

### Planner-Researcher Interaction

The primary loop is turn-based. The researcher provides an intent, the Planner proposes a workflow (a graph of NodeTemplate instances), and the researcher approves or modifies it.

### Standardized Agent Nodes

The framework uses a repository of specialized, pre-defined agents (e.g., LiteratureSearchAgent, DataSearchAgent). Crucially, every agent and logical step in a workflow **must** be an implementation of the framework-agnostic NodeTemplate. This ensures modularity and reusability.

### Context Management

1. A **Global Context** stores the overall state of the research project (e.g., hypothesis, key literature, datasets).
2. Each agent operates on a **Local Context**, which is a strict, need-to-know subset of the Global Context, managed within its NodeTemplate. This ensures portability and reduces context-related errors.

### Human-in-the-Loop (HITL) Control Points

- **Plan Approval**: The initial plan and any significant modifications must be approved.
- **Parameter Tuning**: The researcher can inspect and change the parameters of any NodeTemplate in the workflow.
- **Branching/Merging Decisions**: The AI can suggest branching the workflow, but the researcher must approve this. The merging of branches is also a human-directed process.

## Node Architecture: The Standardized NodeTemplate

To ensure framework-agnosticism, reusability, and consistency, all functional units within a research graph **MUST** be implemented using the NodeTemplate class. This creates a standard for building and integrating tools, whether they are custom-built or from external libraries. The core logic remains decoupled from any specific orchestration engine (e.g., LangGraph).

A NodeTemplate encapsulates a small, self-contained graph within the larger research workflow.

### Anatomy of a NodeTemplate

| Component | Description |
|---|---|
| **State** | A well-defined, typed dictionary with accessible keys representing the node's internal memory. |
| **Input Guardrails** | A set of validation functions that MUST be executed on the input state. The node will halt if validation fails, preventing propagation of bad data. |
| **Node Supervisor** | An LLM agent or custom logic responsible for the node's internal orchestration. It has a sandboxed view, with access ONLY to the node's local state and its assigned tool subset. |
| **Tool Subset** | An explicit list of tools the supervisor is permitted to use from the global tool pool. This enforces the principle of least privilege. |
| **Global Tools** | A set of mandatory tools (e.g., a relevancy checker, reflection agent) that MUST be executed consistently within every node to ensure global standards are met. |
| **Output Guardrails** | A set of validation functions that MUST be executed on the node's output before it can be passed to the next node in the graph. |

### Example Implementation

The core logic is defined independently of any orchestration framework:

```python
# Define the node using the standardized template
from akd import NodeTemplate

node_logic = NodeTemplate(
    state={...},              # Define the expected state structure
    supervisor=...,           # LLM-based or custom logic supervisor
    tools=[...],              # Subset of tools available within this node
    input_guardrails=[...],   # Input validation rules
    output_guardrails=[...],  # Output validation rules
    global_tools=[...],       # Mandatory global checks
)
```

To use it in a specific framework like LangGraph, a wrapper is applied. The core logic inside `node_logic` remains unchanged:

```python
langgraph_node = node_logic.as_langgraph_node(...)
```

## Deep Guardrails for Scientific Trust

Trust is paramount. Our guardrails go beyond simple safety and focus on scientific validity. The NodeTemplate architecture is the primary mechanism for implementing these guardrails at a granular level.

### Deep Attribution

Every claim or piece of synthesized text must be traceable to the specific sentence or data point in the source material.

### Conflicting Literature Identification

The LiteratureReviewAgent is explicitly designed to identify and surface papers that present conflicting or contradictory findings.

### Agentic RAG for Scientific Coverage

Instead of a simple RAG, we employ a multi-agent approach to information retrieval:

- A **Gap Agent** actively looks for what's *missing* from the retrieved literature.
- A **Conflict Agent** specifically searches for contradictory evidence.

### Relevance and Quality

The system prioritizes refereed, high-quality journals and data repositories. The relevance of all fetched artifacts is continuously checked against the Global Context, often via a mandatory global_tool in each node.

## Shareable, Repeatable, and Reusable Research Artifacts

The output is not just a final report or a chat history. It is a complete, stateful research object.

### Stateful Execution Flow

The entire research process is captured as an execution graph composed of NodeTemplate instances.

### Reproducibility

Another researcher can take this graph, inspect every node, see the exact literature and data used, and either re-run it to validate the findings or branch from it to build their own work.

### Components of a Shared Artifact

- The final execution graph
- A manifest of all literature (with DOIs), data (with identifiers), and code used
- The Global Context at each major step
- Human annotations and decisions made during the workflow

## Design Considerations for Control and Trust

### Consistency in Planning

The Planner uses a few-shot RAG approach, learning from previously successful and human-validated research workflows to propose more effective plans. This promotes consistency and leverages community-validated best practices.

### Extensible Tools

Agents are equipped with a toolset. These tools are designed to be self-contained and decoupled, allowing for easy extension and integration of new capabilities developed by the community. Any new tool can be added to the global pool and then assigned to the tool subset of relevant NodeTemplates.

### Clarity over "Magic"

The system avoids being a "black box". The standardized structure of the NodeTemplate makes each step's logic, inputs, and outputs inspectable.

## Implementation Guidelines

When developing within this framework:

1. **Always use NodeTemplate** for any functional component
2. **Require human approval** for any workflow modifications
3. **Implement comprehensive guardrails** at input and output levels
4. **Maintain clear attribution** for all sources and claims
5. **Design for transparency** - every step should be inspectable
6. **Prioritize scientific rigor** over convenience or speed
7. **Enable reproducibility** through complete state capture

This philosophy ensures that the Accelerated Discovery framework remains true to its core mission: empowering human researchers with AI tools while maintaining the highest standards of scientific integrity and transparency.
