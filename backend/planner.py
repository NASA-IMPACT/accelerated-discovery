"""
Simple planner module that creates a StateGraph with NodeTemplate nodes.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Add parent directory to path to import AKD modules
sys.path.append(str(Path(__file__).parent.parent))

from akd.nodes.states import GlobalState, NodeState
from akd.nodes.supervisor import BaseSupervisor
from akd.nodes.templates import SupervisedNodeTemplate
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class SearchNodeSupervisor(BaseSupervisor):
    """
    A custom supervisor that calls the search tool.
    """
    
    async def _arun(
        self,
        state: NodeState,
        **kwargs,
    ) -> NodeState:
        # Get query from inputs
        query = state.inputs.get("query", "")
        
        if not query:
            raise ValueError("State must provide an 'inputs' field with a 'query'.")
        
        # Get the search tool
        if self.tools:
            search_tool = self.tools[0]
            
            # Create input for the tool
            tool_input = search_tool.input_schema(
                queries=[query],
                category="science",
                max_results=10
            )
            
            # Call the tool
            result = await search_tool.arun(tool_input)
            
            # Store results in outputs
            state.outputs["search_results"] = result.model_dump()
            state.outputs["query"] = query
        else:
            state.outputs["error"] = "No search tool available"
        
        return state


def create_planner_graph() -> StateGraph:
    """
    Create a simple StateGraph with a search node using SearxNG.
    
    Returns:
        StateGraph with one node: search_node -> END
    """
    # Create graph with GlobalState
    graph = StateGraph(GlobalState)
    
    # Create the search node using the helper function
    search_node = create_search_node()
    
    # Add node to graph (convert to langgraph node)
    graph.add_node("search_node", search_node.to_langgraph_node("search_node"))
    
    # Set entry point and edge
    graph.set_entry_point("search_node")
    graph.add_edge("search_node", END)
    
    return graph


def create_search_node() -> SupervisedNodeTemplate:
    """
    Create a single node that uses SearxNGSearchTool for web searching.
    
    Returns:
        SupervisedNodeTemplate configured with SearxNG search capabilities
    """
    # Create the search tool with configuration
    searxng_config = SearxNGSearchToolConfig(
        base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
        max_results=int(os.getenv("SEARXNG_MAX_RESULTS", 10)),
        debug=False
    )
    searxng_tool = SearxNGSearchTool(config=searxng_config)
    
    # Create supervisor with the search tool
    search_supervisor = SearchNodeSupervisor(
        tools=[searxng_tool],
        debug=False
    )
    
    # Create the node using SupervisedNodeTemplate
    search_node = SupervisedNodeTemplate(
        supervisor=search_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        node_id="search_node",
        debug=False,
        mutation=True  # Allow state mutation
    )
    
    return search_node