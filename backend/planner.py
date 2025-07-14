"""
Simple planner module that creates a StateGraph with NodeTemplate nodes.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Add parent directory to path to import AKD modules
sys.path.append(str(Path(__file__).parent.parent))

from akd.nodes.states import GlobalState
from akd.nodes.states import GlobalState as AKDGlobalState
from akd.nodes.supervisor import DummyLLMSupervisor, SupervisorState
from akd.nodes.templates import DefaultNodeTemplate
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig, SearxNGSearchToolInputSchema


# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class SearchNodeSupervisor(DummyLLMSupervisor):
    """
    A custom supervisor that calls the search tool.
    """
    
    async def arun(
        self,
        state: SupervisorState,
        global_state: Optional[AKDGlobalState] = None,
        **kwargs,
    ) -> SupervisorState:
        # Get query with fallback options like the older code
        query = (
            state.inputs.get("query", "")
            or state.inputs.get("question", "")
            or state.inputs.get("input", "")
        )
        
        if not query:
            raise ValueError(
                "State must provide an 'inputs' field with a 'query' or 'question'.",
            )
        
        execution_state = state.model_copy()
        
        try:
            # Check if we have tools
            if self.tools and len(self.tools) > 0:
                search_tool = self.tools[0]
                

                
                # Use the supervisor's arun_tool method which handles converted tools properly
                tool_params = {
                    "queries": [query],
                    "category": "science",
                    "max_results": 10
                }
                
                # Add debug print to see what's happening
                print(f"DEBUG: About to call search tool with query: {query}")
                print(f"DEBUG: Tool type: {type(search_tool)}")
                
                result = await self.arun_tool(search_tool, tool_params)
                
                print(f"DEBUG: Result received, type: {type(result)}")
                
                execution_state.output = {
                    "query": query,
                    "results": result.results if hasattr(result, 'results') else []
                }
            else:
                execution_state.output = {
                    "query": query,
                    "status": "error",
                    "error": "No search tool available"
                }
                
        except Exception as e:
            print(f"DEBUG: Exception in arun: {type(e).__name__}: {str(e)}")
            execution_state.output = {
                "query": query,
                "status": "error",
                "error": str(e)
            }
        
        execution_state.steps = {}
        self.update_state(execution_state)
        return execution_state


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


def create_search_node() -> DefaultNodeTemplate:
    """
    Create a single node that uses SearxNGSearchTool for web searching.
    
    Returns:
        DefaultNodeTemplate configured with SearxNG search capabilities
    """
    # Create the search tool with configuration
    # It will use environment variables by default:
    # SEARXNG_BASE_URL (default: http://localhost:8080)
    # SEARXNG_MAX_RESULTS (default: 10)
    # SEARXNG_ENGINES (default: google,arxiv,google_scholar)
    searxng_config = SearxNGSearchToolConfig(
        base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
        max_results=int(os.getenv("SEARXNG_MAX_RESULTS", 10)),
        debug=False
    )
    searxng_tool = SearxNGSearchTool(config=searxng_config)
    
    # Create LLM client
    # Get API key from environment (now loaded from .env file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please add it to the .env file in the project root directory: "
            f"{env_path}"
        )
    
    llm_client = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.0
    )
    
    # Use SearchNodeSupervisor that properly handles search tool results
    search_supervisor = SearchNodeSupervisor(
        llm_client=llm_client,
        tools=[searxng_tool],
        debug=False
    )
    
    # Create the node using DefaultNodeTemplate (concrete implementation)
    search_node = DefaultNodeTemplate(
        supervisor=search_supervisor,
        input_guardrails=[],
        output_guardrails=[],
        node_id="search_node",
        debug=False
    )
    
    return search_node