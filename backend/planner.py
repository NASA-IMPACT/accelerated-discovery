"""
Simple planner module that creates a StateGraph with NodeTemplate nodes.
Based on e2e_agents_node_template.py example.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, BaseModel

# Add parent directory to path to import AKD modules
sys.path.append(str(Path(__file__).parent.parent))

from akd.nodes.states import GlobalState, NodeState, NodeTemplateState
from akd.nodes.templates import AbstractNodeTemplate
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig
from akd.tools import BaseTool, BaseToolConfig
from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.serializers import AKDSerializer

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class LiteratureSearchNode(AbstractNodeTemplate):
    """
    Literature search node that uses SearxNG for web searching.
    """
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        # Get query from inputs
        query = node_state.inputs.get("query", "")
        
        if not query:
            node_state.outputs["error"] = "No query provided"
            return node_state
        
        # Create and use the search tool
        searxng_config = SearxNGSearchToolConfig(
            base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
            max_results=int(os.getenv("SEARXNG_MAX_RESULTS", 10)),
            debug=False
        )
        search_tool = SearxNGSearchTool(config=searxng_config)
        
        try:
            # Run the search
            result = await search_tool.arun(
                search_tool.input_schema(queries=[query])
            )
            node_state.outputs["search_results"] = result.model_dump()
            node_state.outputs["query"] = query
            node_state.outputs["status"] = "success"
        except Exception as e:
            # If SearxNG is not available, provide mock results
            node_state.outputs["search_results"] = {
                "results": [
                    {
                        "title": f"Mock result 1 for: {query}",
                        "url": "https://example.com/1",
                        "content": f"This is a mock search result for query: {query}",
                        "score": 0.95
                    },
                    {
                        "title": f"Mock result 2 for: {query}",
                        "url": "https://example.com/2",
                        "content": f"Another mock result for: {query}",
                        "score": 0.89
                    }
                ],
                "query": query,
                "total_results": 2
            }
            node_state.outputs["query"] = query
            node_state.outputs["status"] = "mock_results"
            node_state.outputs["error"] = str(e)
        
        return node_state


# Dummy Code Search Tool for demonstration
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
        return DummyCodeSearchOutputSchema(results=res[: self.max_results])


class CodeSearchNode(AbstractNodeTemplate):
    """
    Code search node that searches for relevant code.
    """
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        # Get query from literature search node's inputs
        lit_search_state = global_state.node_states.get("lit_search")
        if lit_search_state:
            query = lit_search_state.inputs.get("query", "")
        else:
            query = node_state.inputs.get("query", "")
        
        if not query:
            node_state.outputs["error"] = "No query available"
            return node_state
        
        # Use dummy code search tool
        search_tool = DummyCodeSearchTool()
        result = await search_tool.arun(
            search_tool.input_schema(queries=[query])
        )
        
        node_state.outputs["code_results"] = result.model_dump()
        node_state.outputs["query"] = query
        node_state.outputs["status"] = "success"
        
        return node_state


class ReportGenerationNode(AbstractNodeTemplate):
    """
    Report generation node that combines results from previous nodes.
    """
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        # Collect results from previous nodes
        lit_results = []
        code_results = []
        
        # Get literature search results
        lit_search_state = global_state.node_states.get("lit_search")
        if lit_search_state and "search_results" in lit_search_state.outputs:
            lit_results = lit_search_state.outputs["search_results"].get("results", [])
        
        # Get code search results
        code_search_state = global_state.node_states.get("code_search")
        if code_search_state and "code_results" in code_search_state.outputs:
            code_results = code_search_state.outputs["code_results"].get("results", [])
        
        # Generate simple report
        report = {
            "summary": f"Found {len(lit_results)} literature results and {len(code_results)} code results",
            "literature_findings": [
                {"title": r.get("title", ""), "content": r.get("content", "")} 
                for r in lit_results[:3]  # Top 3 results
            ],
            "code_findings": [
                {"title": r.get("title", ""), "url": r.get("url", "")} 
                for r in code_results
            ],
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        node_state.outputs["report"] = report
        node_state.outputs["status"] = "completed"
        
        return node_state


def create_simple_plan() -> StateGraph:
    """
    Create a simple StateGraph with just a literature search node.
    
    Returns:
        StateGraph with one node: lit_search -> END
    """
    # Create graph with GlobalState
    graph = StateGraph(GlobalState)
    
    # Create the search node
    search_node = LiteratureSearchNode(
        node_id="lit_search",
        debug=True,
        mutation=True,
    )
    
    # Add node to graph
    graph.add_node("lit_search", search_node.to_langgraph_node(key="lit_search"))
    
    # Set entry point and edge
    graph.set_entry_point("lit_search")
    graph.add_edge("lit_search", END)
    
    return graph


def create_full_plan() -> StateGraph:
    """
    Create a complete workflow with multiple nodes:
    lit_search -> code_search -> report_generation -> END
    
    Returns:
        StateGraph with complete workflow
    """
    # Create graph with GlobalState
    graph = StateGraph(GlobalState)
    
    # Create nodes
    lit_search_node = LiteratureSearchNode(
        node_id="lit_search",
        debug=True,
        mutation=True,
    )
    
    code_search_node = CodeSearchNode(
        node_id="code_search",
        debug=True,
        mutation=True,
    )
    
    report_node = ReportGenerationNode(
        node_id="report_generation",
        debug=True,
        mutation=True,
    )
    
    # Add nodes to graph
    graph.add_node("lit_search", lit_search_node.to_langgraph_node(key="lit_search"))
    graph.add_node("code_search", code_search_node.to_langgraph_node(key="code_search"))
    graph.add_node("report_generation", report_node.to_langgraph_node(key="report_generation"))
    
    # Set up flow
    graph.set_entry_point("lit_search")
    graph.add_edge("lit_search", "code_search")
    graph.add_edge("code_search", "report_generation")
    graph.add_edge("report_generation", END)
    
    return graph

