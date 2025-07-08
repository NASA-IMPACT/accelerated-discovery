"""
Workflow templates for common LangGraph patterns using Node Templates.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from .states import GlobalState
from .supervisor import BaseSupervisor
from .templates import AbstractNodeTemplate, NodeTemplateConfig


@dataclass
class WorkflowTemplate:
    """Base class for workflow templates."""

    name: str
    description: str
    nodes: List[AbstractNodeTemplate]
    edges: List[tuple]  # (from_node, to_node) or (from_node, to_node, condition)

    def create_langgraph_workflow(
        self,
        initial_state: Optional[GlobalState] = None,
        enable_interrupts: bool = True,
        enable_checkpointing: bool = True,
    ) -> StateGraph:
        """Create a LangGraph workflow from the template."""
        graph = StateGraph(GlobalState)

        # Add all nodes to the graph
        for node in self.nodes:
            graph.add_node(
                node.node_id,
                node.to_langgraph_node(
                    enable_interrupts=enable_interrupts,
                    enable_checkpointing=enable_checkpointing,
                ),
            )

        # Add edges
        for edge in self.edges:
            if len(edge) == 2:
                # Simple edge
                from_node, to_node = edge
                if from_node == "START":
                    graph.add_edge(START, to_node)
                elif to_node == "END":
                    graph.add_edge(from_node, END)
                else:
                    graph.add_edge(from_node, to_node)
            elif len(edge) == 3:
                # Conditional edge
                from_node, to_node, condition = edge
                if callable(condition):
                    graph.add_conditional_edges(from_node, condition)
                else:
                    # Simple mapping
                    graph.add_conditional_edges(
                        from_node,
                        lambda x: condition,
                        {condition: to_node},
                    )

        return graph

    def get_execution_plan(self) -> List[str]:
        """Get the planned execution order of nodes."""
        # Simple topological sort based on edges
        plan = []
        visited = set()

        def visit(node_id: str):
            if node_id in visited or node_id in ["START", "END"]:
                return
            visited.add(node_id)

            # Find dependencies (incoming edges)
            for edge in self.edges:
                if len(edge) >= 2 and edge[1] == node_id and edge[0] not in ["START"]:
                    visit(edge[0])

            plan.append(node_id)

        # Start from nodes that have START as input
        for edge in self.edges:
            if len(edge) >= 2 and edge[0] == "START":
                visit(edge[1])

        return plan


class SequentialWorkflowTemplate(WorkflowTemplate):
    """Template for sequential execution of nodes."""

    def __init__(
        self,
        nodes: List[AbstractNodeTemplate],
        name: str = "Sequential Workflow",
    ):
        # Create sequential edges
        edges = [("START", nodes[0].node_id)]
        for i in range(len(nodes) - 1):
            edges.append((nodes[i].node_id, nodes[i + 1].node_id))
        edges.append((nodes[-1].node_id, "END"))

        super().__init__(
            name=name,
            description=f"Sequential execution of {len(nodes)} nodes",
            nodes=nodes,
            edges=edges,
        )


class ParallelWorkflowTemplate(WorkflowTemplate):
    """Template for parallel execution of nodes with optional merge."""

    def __init__(
        self,
        nodes: List[AbstractNodeTemplate],
        merge_node: Optional[AbstractNodeTemplate] = None,
        name: str = "Parallel Workflow",
    ):
        # Create parallel edges
        edges = []

        # Connect START to all nodes
        for node in nodes:
            edges.append(("START", node.node_id))

        if merge_node:
            # Connect all nodes to merge node
            for node in nodes:
                edges.append((node.node_id, merge_node.node_id))
            edges.append((merge_node.node_id, "END"))
            nodes = nodes + [merge_node]
        else:
            # Connect all nodes to END
            for node in nodes:
                edges.append((node.node_id, "END"))

        super().__init__(
            name=name,
            description=f"Parallel execution of {len(nodes)} nodes"
            + (" with merge" if merge_node else ""),
            nodes=nodes,
            edges=edges,
        )


class ConditionalWorkflowTemplate(WorkflowTemplate):
    """Template for conditional execution based on node outputs."""

    def __init__(
        self,
        decision_node: AbstractNodeTemplate,
        condition_map: Dict[str, AbstractNodeTemplate],
        name: str = "Conditional Workflow",
    ):
        def routing_function(state: GlobalState) -> str:
            """Route based on decision node output."""
            decision_output = state.get_node_state(decision_node.node_id)
            if decision_output and decision_output.output:
                decision_key = decision_output.output.get("decision", "default")
                return decision_key if decision_key in condition_map else "default"
            return "default"

        nodes = [decision_node] + list(condition_map.values())
        edges = [
            ("START", decision_node.node_id),
            (decision_node.node_id, routing_function),
        ]

        # Add edges from condition nodes to END
        for node in condition_map.values():
            edges.append((node.node_id, "END"))

        super().__init__(
            name=name,
            description=f"Conditional workflow with {len(condition_map)} branches",
            nodes=nodes,
            edges=edges,
        )


class LoopWorkflowTemplate(WorkflowTemplate):
    """Template for loop-based execution."""

    def __init__(
        self,
        loop_body: List[AbstractNodeTemplate],
        condition_node: AbstractNodeTemplate,
        max_iterations: int = 10,
        name: str = "Loop Workflow",
    ):
        def continue_loop(state: GlobalState) -> str:
            """Check if loop should continue."""
            condition_output = state.get_node_state(condition_node.node_id)
            if condition_output and condition_output.output:
                # Check iteration count
                iteration_count = state.get_from_shared_context("iteration_count", 0)
                if iteration_count >= max_iterations:
                    return "END"

                # Check condition
                should_continue = condition_output.output.get("continue", False)
                if should_continue:
                    # Increment iteration count
                    state.add_to_shared_context("iteration_count", iteration_count + 1)
                    return loop_body[0].node_id
                else:
                    return "END"
            return "END"

        nodes = loop_body + [condition_node]
        edges = [("START", loop_body[0].node_id)]

        # Connect loop body sequentially
        for i in range(len(loop_body) - 1):
            edges.append((loop_body[i].node_id, loop_body[i + 1].node_id))

        # Connect last loop body node to condition
        edges.append((loop_body[-1].node_id, condition_node.node_id))

        # Conditional edge from condition node
        edges.append((condition_node.node_id, continue_loop))

        super().__init__(
            name=name,
            description=f"Loop workflow with {len(loop_body)} body nodes, max {max_iterations} iterations",
            nodes=nodes,
            edges=edges,
        )


class ErrorHandlingWorkflowTemplate(WorkflowTemplate):
    """Template for workflows with error handling and retry logic."""

    def __init__(
        self,
        main_nodes: List[AbstractNodeTemplate],
        error_handler: AbstractNodeTemplate,
        retry_limit: int = 3,
        name: str = "Error Handling Workflow",
    ):
        def error_check(state: GlobalState) -> str:
            """Check for errors and determine next step."""
            # Check if any main node failed
            failed_nodes = state.get_failed_nodes()
            if failed_nodes:
                retry_count = state.get_from_shared_context("retry_count", 0)
                if retry_count < retry_limit:
                    state.add_to_shared_context("retry_count", retry_count + 1)
                    # Reset failed nodes and retry
                    for node_id in failed_nodes:
                        if node_id in state.node_states:
                            state.node_states[node_id].supervisor_state.set_status(
                                "pending",
                            )
                    return main_nodes[0].node_id
                else:
                    return error_handler.node_id
            return "END"

        nodes = main_nodes + [error_handler]
        edges = [("START", main_nodes[0].node_id)]

        # Connect main nodes sequentially
        for i in range(len(main_nodes) - 1):
            edges.append((main_nodes[i].node_id, main_nodes[i + 1].node_id))

        # Connect last main node to error check
        edges.append((main_nodes[-1].node_id, error_check))

        # Connect error handler to END
        edges.append((error_handler.node_id, "END"))

        super().__init__(
            name=name,
            description=f"Error handling workflow with {len(main_nodes)} main nodes and {retry_limit} retry limit",
            nodes=nodes,
            edges=edges,
        )


class WorkflowBuilder:
    """Builder for creating complex workflows."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.nodes: List[AbstractNodeTemplate] = []
        self.edges: List[tuple] = []
        self.entry_points: List[str] = []
        self.exit_points: List[str] = []

    def add_node(self, node: AbstractNodeTemplate) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        self.nodes.append(node)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "WorkflowBuilder":
        """Add a simple edge."""
        self.edges.append((from_node, to_node))
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[GlobalState], str],
    ) -> "WorkflowBuilder":
        """Add a conditional edge."""
        self.edges.append((from_node, condition))
        return self

    def set_entry_point(self, node_id: str) -> "WorkflowBuilder":
        """Set the entry point for the workflow."""
        self.entry_points.append(node_id)
        return self

    def set_exit_point(self, node_id: str) -> "WorkflowBuilder":
        """Set an exit point for the workflow."""
        self.exit_points.append(node_id)
        return self

    def build(self) -> WorkflowTemplate:
        """Build the workflow template."""
        # Add START and END connections
        edges = []

        # Connect START to entry points
        for entry in self.entry_points:
            edges.append(("START", entry))

        # Add all defined edges
        edges.extend(self.edges)

        # Connect exit points to END
        for exit_point in self.exit_points:
            edges.append((exit_point, "END"))

        return WorkflowTemplate(
            name=self.name,
            description=self.description,
            nodes=self.nodes,
            edges=edges,
        )


# Example usage and convenience functions
def create_simple_sequential_workflow(
    supervisors: List[BaseSupervisor],
    node_configs: Optional[List[NodeTemplateConfig]] = None,
) -> SequentialWorkflowTemplate:
    """Create a simple sequential workflow from supervisors."""
    from .templates import DefaultNodeTemplate

    nodes = []
    for i, supervisor in enumerate(supervisors):
        config = (
            node_configs[i]
            if node_configs and i < len(node_configs)
            else NodeTemplateConfig()
        )
        node = DefaultNodeTemplate(
            supervisor=supervisor,
            input_guardrails=[],
            output_guardrails=[],
            config=config,
        )
        nodes.append(node)

    return SequentialWorkflowTemplate(nodes)


def create_research_workflow(
    query_supervisor: BaseSupervisor,
    search_supervisor: BaseSupervisor,
    analysis_supervisor: BaseSupervisor,
) -> WorkflowTemplate:
    """Create a typical research workflow: Query -> Search -> Analysis."""
    from .templates import DefaultNodeTemplate

    query_node = DefaultNodeTemplate(
        node_id="query_node",
        supervisor=query_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Query Processing",
            description="Process and expand the research query",
            tags=["query", "preprocessing"],
        ),
    )

    search_node = DefaultNodeTemplate(
        node_id="search_node",
        supervisor=search_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Literature Search",
            description="Search for relevant literature",
            tags=["search", "literature"],
            dependencies=[query_node.node_id],
        ),
    )

    analysis_node = DefaultNodeTemplate(
        node_id="analysis_node",
        supervisor=analysis_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Analysis",
            description="Analyze search results",
            tags=["analysis", "postprocessing"],
            dependencies=[search_node.node_id],
        ),
    )

    return SequentialWorkflowTemplate(
        nodes=[query_node, search_node, analysis_node],
        name="Research Workflow",
    )
