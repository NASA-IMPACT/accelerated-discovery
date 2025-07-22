#!/usr/bin/env python3
"""
End-to-end agents with node template implementation.
Converted from marimo notebook to regular Python script.
"""

import asyncio
from typing import Any, Dict, List
from loguru import logger
from pydantic import Field, BaseModel

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# AKD imports
from akd.tools.search import (
    SearxNGSearchToolInputSchema,
    SearxNGSearchToolOutputSchema,
    SearxNGSearchTool,
)
from akd.agents.query import (
    QueryAgent,
    QueryAgentInputSchema,
    QueryAgentOutputSchema,
)
from akd.tools.utils import tool_wrapper
from akd.tools import BaseTool, BaseToolConfig
from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.nodes.states import (
    NodeState,
    GlobalState,
)
from akd.nodes.states_v2 import (
    NodeTemplateState,
    SupervisorState,
)
from akd.nodes.templates import AbstractNodeTemplate, SupervisedNodeTemplate
from akd.nodes.supervisor import (
    ManualSupervisor,
    ReActLLMSupervisor,
)
from akd.serializers import AKDSerializer

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver


# Initialize tools and agents
SEARCH_TOOL = SearxNGSearchTool()
QUERY_AGENT = QueryAgent()


# Dummy Code Search Tool Implementation
class DummyCodeSearchInputSchema(InputSchema):
    """
    Dummy input schema for code search
    """
    queries: list[str] = Field(
        ..., description="Input queries for code search"
    )


class DummyCodeSearchOutputSchema(OutputSchema):
    """
    Dummy output schema for code search
    """
    results: list[SearchResultItem] = Field(
        ..., description="List of search result items"
    )


class DummyCodeSearchToolConfig(BaseToolConfig):
    max_results: int = Field(
        default=10, description="Maximum number of search result items"
    )


class DummyCodeSearchTool(BaseTool):
    input_schema = DummyCodeSearchInputSchema
    output_schema = DummyCodeSearchOutputSchema
    config_schema = DummyCodeSearchToolConfig

    async def _arun(
        self, params: DummyCodeSearchInputSchema
    ) -> DummyCodeSearchOutputSchema:
        res = []
        for query in params.queries:
            res.append(
                SearchResultItem(
                    url="https://github.com/NASA-IMPACT/accelerated-discovery",
                    content=" A multiagent framework to augment research workflow ",
                    title="NASA-IMPACT/accelerated-discovery",
                    query=query,
                )
            )
        return DummyCodeSearchOutputSchema(results=res)


# Node Template Implementations
class LiteratureSearchNode(AbstractNodeTemplate):
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        query = node_state.inputs.get("query")
        node_state.outputs["search"] = await SEARCH_TOOL.arun(
            SEARCH_TOOL.input_schema(queries=[query])
        )
        return node_state


class CodeSearchNode(AbstractNodeTemplate):
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        search_tool = DummyCodeSearchTool()
        query = global_state.node_states.get("lit_search").inputs.get("query")
        node_state.outputs["search"] = await search_tool.arun(
            search_tool.input_schema(queries=[query])
        )
        return node_state


# Supervisor Implementation
class CodeSearchSupervisor(ManualSupervisor):
    async def _arun(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        search_tool = DummyCodeSearchTool()
        query = global_state.node_states.get("lit_search").inputs.get("query")
        node_state.outputs["search"] = await search_tool.arun(
            search_tool.input_schema(queries=[query])
        )
        return node_state


# Guardrail functions
@tool_wrapper
async def always_bad(inp: str) -> str:
    """Always returns a bad result for testing."""
    raise ValueError("This guardrail always fails")


@tool_wrapper
async def always_good(inp: str) -> str:
    """Always returns a good result for testing."""
    return "All good!"


@tool_wrapper
async def search_result_min_length(results: list, min_length: int = 10) -> bool:
    """Check if search results meet minimum length requirement."""
    return len(results) >= min_length


# Test functions
async def test_query_agent():
    """Test the query agent functionality."""
    tool = QUERY_AGENT.to_langchain_structured_tool()
    result = await tool.ainvoke(input=dict(query="flood landslide nepal", num_queries=5))
    print("Query Agent Result:", result)
    return result


async def test_search_tool():
    """Test the search tool functionality."""
    tool = SEARCH_TOOL.to_langchain_structured_tool()
    result = await tool.ainvoke(input=dict(queries=["flood landslide nepal"]))
    print("Search Tool Result:", result)
    return result


async def test_dummy_code_search():
    """Test the dummy code search tool."""
    tool = DummyCodeSearchTool()
    result = await tool.arun(DummyCodeSearchInputSchema(queries=["dummy input"]))
    print("Dummy Code Search Result:", result.model_dump())
    return result


async def test_literature_search_node():
    """Test the literature search node."""
    search_node = LiteratureSearchNode(
        node_id="lit_search",
        input_guardrails=[always_bad],
        output_guardrails=[
            (always_good, {"inp": "results"}),
            search_result_min_length,
        ],
        debug=True,
        mutation=True,
    )

    global_state = GlobalState(
        node_states={
            "lit_search": NodeState(
                inputs={
                    "query": "landslide casualties in Nepal last year",
                }
            ),
        }
    )

    result = await search_node.arun(global_state)
    print("Literature Search Node Result:", result.model_dump())
    return result


async def test_code_search_node():
    """Test the code search node."""
    code_search_node = CodeSearchNode(
        node_id="code_search",
        debug=True,
        mutation=True,
    )

    global_state = GlobalState(
        node_states={
            "lit_search": NodeState(
                inputs={
                    "query": "landslide casualties in Nepal last year",
                }
            ),
            "code_search": NodeState(
                inputs={
                    "query": "landslide casualties in Nepal last year",
                }
            ),
        }
    )

    result = await code_search_node.arun(global_state)
    print("Code Search Node Result:", result.model_dump())
    return result


async def create_and_run_workflow():
    """Create and run the complete workflow with langgraph."""
    # Initialize memory and config
    memory = InMemorySaver(serde=AKDSerializer())
    config = {"configurable": {"thread_id": "123"}}
    
    # Create nodes
    search_node = LiteratureSearchNode(
        node_id="lit_search",
        debug=True,
        mutation=True,
    )
    
    code_search_node = CodeSearchNode(
        node_id="code_search",
        debug=True,
        mutation=True,
    )
    
    # Build graph
    graph = StateGraph(GlobalState)
    
    # Add nodes
    graph.add_node("lit_search", search_node.to_langgraph_node(key="lit_search"))
    graph.add_node("code_search", code_search_node.to_langgraph_node(key="code_search"))
    
    # Set up flow
    graph.set_entry_point("lit_search")
    graph.add_edge("lit_search", "code_search")
    graph.add_edge("code_search", END)
    
    # Initial state
    global_state = GlobalState(
        node_states={
            "lit_search": NodeState(
                inputs={
                    "query": "landslide casualties in Nepal last year",
                }
            ),
            "code_search": NodeState(),
        }
    )
    
    # Compile workflow
    workflow = graph.compile(checkpointer=memory)
    
    # Run workflow
    result_graph = await workflow.ainvoke(global_state, config=config)
    
    print("=" * 80)
    print("Workflow Result:")
    print(result_graph)
    print("=" * 80)
    
    # Stream results
    print("\nStreaming workflow steps:")
    print("-" * 50)
    async for chunk in workflow.astream(global_state, config=config):
        print(f"Step: {chunk}")
        print("-" * 50)
    
    return result_graph, global_state


async def main():
    """Main function to run all tests."""
    print("Running E2E Agents Node Template Tests\n")
    
    # Run individual tests
    print("1. Testing Query Agent...")
    try:
        await test_query_agent()
    except Exception as e:
        logger.error(f"Query Agent test failed: {e}")
    
    print("\n2. Testing Search Tool...")
    try:
        await test_search_tool()
    except Exception as e:
        logger.error(f"Search Tool test failed: {e}")
    
    print("\n3. Testing Dummy Code Search...")
    try:
        await test_dummy_code_search()
    except Exception as e:
        logger.error(f"Dummy Code Search test failed: {e}")
    
    print("\n4. Testing Literature Search Node...")
    try:
        await test_literature_search_node()
    except Exception as e:
        logger.error(f"Literature Search Node test failed: {e}")
    
    print("\n5. Testing Code Search Node...")
    try:
        await test_code_search_node()
    except Exception as e:
        logger.error(f"Code Search Node test failed: {e}")
    
    print("\n6. Running Complete Workflow...")
    try:
        await create_and_run_workflow()
    except Exception as e:
        logger.error(f"Workflow test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())