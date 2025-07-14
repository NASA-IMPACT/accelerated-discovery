"""
Simple planner module that creates a StateGraph with NodeTemplate nodes.
"""

import sys
from pathlib import Path

# Add parent directory to path to import AKD modules
sys.path.append(str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from akd.nodes.states import GlobalState
from akd.nodes.templates import DefaultNodeTemplate
from akd.nodes.supervisor import DummyLLMSupervisor, SupervisorState
from akd.nodes.states import GlobalState as AKDGlobalState
from typing import Optional
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class SearchNodeSupervisor(DummyLLMSupervisor):
    """
    A custom supervisor specifically for the search node that properly handles search results.
    """
    
    async def arun(
        self,
        state: SupervisorState,
        global_state: Optional[AKDGlobalState] = None,
        **kwargs,
    ) -> SupervisorState:
        # Get the query from inputs
        query = (
            state.inputs.get("query", "")
            or state.inputs.get("question", "")
            or state.inputs.get("input", "")
        )
        
        if not query:
            raise ValueError(
                "State must provide an 'inputs' field with a 'query' or 'question'.",
            )
        
        # For now, let's create a simpler flow that directly calls the search tool
        execution_state = state.model_copy()
        
        try:
            # Get the search tool
            if self.tools and len(self.tools) > 0:
                search_tool = self.tools[0]
                
                # Call the search tool directly, similar to test_search_tool.py
                from akd.tools.search import SearxNGSearchToolInputSchema
                
                # Create proper input schema
                tool_input = SearxNGSearchToolInputSchema(
                    queries=[query],
                    category="science",
                    max_results=10
                )
                
                # Call the tool directly
                result = await search_tool.arun(tool_input)
                
                # Process the search results
                # The result is a SearxNGSearchToolOutputSchema object
                if result and hasattr(result, 'results'):
                    results_list = result.results
                    
                    if results_list:
                        # Extract relevant information from search results
                        search_results = []
                        for item in results_list[:5]:  # Limit to top 5 results
                            # The items are SearchResultItem objects
                            search_results.append({
                                "title": getattr(item, 'title', 'No title'),
                                "url": getattr(item, 'url', ''),
                                "content": (getattr(item, 'content', '')[:200] + "...") if getattr(item, 'content', None) else ''
                            })
                        
                        execution_state.output = {
                            "query": query,
                            "results": search_results,
                            "status": "success",
                            "total_results": len(results_list)
                        }
                    else:
                        execution_state.output = {
                            "query": query,
                            "results": [],
                            "status": "no_results",
                            "message": "No search results found"
                        }
                else:
                    execution_state.output = {
                        "query": query,
                        "results": [],
                        "status": "no_results",
                        "message": "No search results found"
                    }
            else:
                execution_state.output = {
                    "query": query,
                    "status": "error",
                    "error": "No search tool available"
                }
                
        except Exception as e:
            # Handle any errors gracefully
            execution_state.output = {
                "query": query,
                "status": "error",
                "error": str(e)
            }
        
        # Clear steps to avoid any update issues
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