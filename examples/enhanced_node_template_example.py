"""Working 3-Node Workflow Example

Simple workflow with:
- Node 1: Query generation
- Node 2: Search results
- Node 3: Follow-up analysis
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from akd.agents.query import FollowUpQueryAgent, QueryAgent
from akd.nodes.states import GlobalState
from akd.nodes.supervisor import ReActLLMSupervisor
from akd.nodes.templates import DefaultNodeTemplate, NodeTemplateConfig
from akd.nodes.workflow_templates import SequentialWorkflowTemplate
from akd.tools.search import SearxNGSearchTool


async def main():
    """Working 3-node workflow example."""

    # Check if we have OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== Working 3-Node Workflow Example ===")

    # Create tools and agents
    query_agent = QueryAgent()
    search_tool = SearxNGSearchTool()
    follow_up_agent = FollowUpQueryAgent()

    # Create LLM for supervisors
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Node 1: Query Generation
    query_supervisor = ReActLLMSupervisor(
        llm_client=llm,
        tools=[query_agent.to_langchain_structured_tool()],
        system_message="You are a query generation expert. Generate 3 diverse search queries for the given topic.",
    )

    query_node = DefaultNodeTemplate(
        node_id="query_generator",
        supervisor=query_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Query Generator",
            description="Generate search queries from user input",
            required_inputs={"query"},
            expected_outputs={"queries"},
        ),
    )

    # Node 2: Search Results
    search_supervisor = ReActLLMSupervisor(
        llm_client=llm,
        tools=[search_tool.to_langchain_structured_tool()],
        system_message="You are a search expert. Use the search tool to find relevant results for the given queries.",
    )

    search_node = DefaultNodeTemplate(
        node_id="search_results",
        supervisor=search_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Search Results",
            description="Search for results using generated queries",
            required_inputs={"queries"},
            expected_outputs={"search_results"},
        ),
    )

    # Node 3: Follow-up Analysis
    analysis_supervisor = ReActLLMSupervisor(
        llm_client=llm,
        tools=[follow_up_agent.to_langchain_structured_tool()],
        system_message="You are an analysis expert. Analyze search results and generate follow-up queries for deeper insights.",
    )

    analysis_node = DefaultNodeTemplate(
        node_id="followup_analysis",
        supervisor=analysis_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        config=NodeTemplateConfig(
            name="Follow-up Analysis",
            description="Analyze results and generate follow-up queries",
            required_inputs={"search_results", "original_queries"},
            expected_outputs={"analysis", "followup_queries"},
        ),
    )

    # Create sequential workflow
    workflow = SequentialWorkflowTemplate(
        nodes=[query_node, search_node, analysis_node],
        name="Research Workflow",
    )

    print(f"Created workflow: {workflow.name}")
    print(f"Execution plan: {workflow.get_execution_plan()}")

    # Create and compile LangGraph
    graph = workflow.create_langgraph_workflow(
        enable_interrupts=False,
        enable_checkpointing=True,
    )

    # Add memory for checkpointing
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    # Create initial state
    initial_state = GlobalState(
        inputs={"query": "machine learning applications in climate science"},
        shared_context={"max_results": 5},
    )

    # Initialize node states
    for node in workflow.nodes:
        initial_state.create_node_state(
            node.node_id,
            initial_inputs=initial_state.inputs,
        )

    print(f"\nInitialized {len(initial_state.node_states)} node states")
    print(f"Input query: {initial_state.inputs['query']}")

    # Execute workflow
    try:
        print("\n=== Starting Workflow Execution ===")

        config = {"configurable": {"thread_id": "test-workflow"}}

        # Stream the execution to see progress
        final_result = None
        async for event in compiled.astream(initial_state, config=config):
            print(f"\nEvent received: {event}")
            node_name = list(event.keys())[0] if event else "unknown"
            print(f"\nCompleted node: {node_name}")

            if node_name in event:
                node_state = event[node_name]
                final_result = node_state
                if hasattr(node_state, "node_states"):
                    for node_id, state in node_state.node_states.items():
                        if state.output:
                            print(
                                f"  {node_id} output keys: {list(state.output.keys())}",
                            )

                            # Show some sample output
                            if "queries" in state.output:
                                print(
                                    f"    Generated queries: {state.output['queries'][:2]}...",
                                )
                            elif "search_results" in state.output:
                                results = state.output["search_results"]
                                if isinstance(results, dict) and "results" in results:
                                    print(
                                        f"    Found {len(results['results'])} search results",
                                    )
                                elif isinstance(results, list):
                                    print(f"    Found {len(results)} search results")
                            elif "followup_queries" in state.output:
                                print(
                                    f"    Follow-up queries: {state.output['followup_queries'][:2]}...",
                                )

        print("\n=== Workflow Completed Successfully ===")

        # Show final summary
        print(final_result)
        if final_result and hasattr(final_result, "node_states"):
            print("\n=== Final Results Summary ===")
            for node_id, state in final_result.node_states.items():
                if state.output:
                    print(f"{node_id}: {list(state.output.keys())}")
                    if state.supervisor_state.tool_calls:
                        print(
                            f"  Tool calls made: {len(state.supervisor_state.tool_calls)}",
                        )
                    if state.supervisor_state.steps:
                        print(f"  Steps completed: {len(state.supervisor_state.steps)}")

        return compiled, final_result

    except Exception as e:
        print(f"\nWorkflow execution failed: {e}")
        print("This might be due to missing API keys or network issues.")
        return compiled, None


async def demonstrate_workflow_patterns(compiled):
    """Demonstrate different ways to use the compiled workflow."""

    print("\n\n=== Workflow Usage Patterns ===")

    # Pattern 1: Simple invoke
    print("\n1. Simple Invoke Pattern:")
    simple_state = GlobalState(
        inputs={"query": "neural networks for weather prediction"},
        shared_context={"max_results": 3},
    )

    try:
        result = await compiled.ainvoke(
            simple_state,
            config={"configurable": {"thread_id": "demo-1"}},
        )
        print(f"   Result type: {type(result)}")
        print(f"   Has node states: {hasattr(result, 'node_states')}")
        if hasattr(result, "node_states"):
            print(f"   Node count: {len(result.node_states)}")
    except Exception as e:
        print(f"   Error: {e}")

    # Pattern 2: Stream pattern
    print("\n2. Stream Pattern:")
    stream_state = GlobalState(
        inputs={"query": "deep learning climate modeling"},
        shared_context={"max_results": 2},
    )

    try:
        events = []
        async for event in compiled.astream(
            stream_state,
            config={"configurable": {"thread_id": "demo-2"}},
        ):
            events.append(list(event.keys())[0] if event else "unknown")
        print(f"   Events processed: {events}")
    except Exception as e:
        print(f"   Error: {e}")

    # Pattern 3: Batch pattern
    print("\n3. Batch Pattern:")
    batch_states = [
        GlobalState(
            inputs={"query": "AI climate change"},
            shared_context={"max_results": 1},
        ),
        GlobalState(
            inputs={"query": "ML environmental science"},
            shared_context={"max_results": 1},
        ),
    ]

    try:
        batch_configs = [
            {"configurable": {"thread_id": f"batch-{i}"}}
            for i in range(len(batch_states))
        ]
        results = await compiled.abatch(batch_states, config=batch_configs)
        print(f"   Batch results count: {len(results)}")
        for i, result in enumerate(results):
            print(
                f"   Result {i}: {type(result)} with {len(getattr(result, 'node_states', {})) if hasattr(result, 'node_states') else 0} nodes",
            )
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    compiled, result = asyncio.run(main())

    if compiled and result:
        # Demonstrate different workflow usage patterns
        asyncio.run(demonstrate_workflow_patterns(compiled))
