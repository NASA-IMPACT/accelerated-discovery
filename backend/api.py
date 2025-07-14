"""
Simple API for executing planner graph.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

from akd.nodes.states import GlobalState
from planner import create_planner_graph

# Create FastAPI app
app = FastAPI(title="AKD Simple Backend")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory saver for checkpointing
memory = MemorySaver()

# Store compiled graphs by workflow_id
compiled_graphs = {}


class ChatRequest(BaseModel):
    message: str


class GraphStructure(BaseModel):
    nodes: list[str]
    edges: list[tuple[str, str]]


@app.post("/api/plan")
async def create_plan(request: ChatRequest):
    """
    Accept chat message and return graph structure.
    Store compiled graph for later execution.
    """
    # Create the planner graph
    graph = create_planner_graph()
    
    # Get graph structure
    nodes = list(graph.nodes.keys())
    edges = list(graph.edges)
    
    # Generate workflow ID for this plan
    workflow_id = str(uuid4())
    
    # Compile and store the graph
    compiled_graph = graph.compile(checkpointer=memory)
    compiled_graphs[workflow_id] = compiled_graph
    
    return {
        "workflow_id": workflow_id,
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "message": f"Graph created for query: {request.message}"
    }


class ExecuteRequest(BaseModel):
    state: Dict[str, Any]
    workflow_id: str


@app.post("/api/execute")
async def execute_graph(request: ExecuteRequest):
    """
    Execute a previously created graph with given state.
    
    Args:
        request: Contains state dict and workflow_id
    """
    # Check if graph exists
    if request.workflow_id not in compiled_graphs:
        raise HTTPException(status_code=404, detail=f"Graph not found for workflow_id: {request.workflow_id}")
    
    # Get the compiled graph
    compiled_graph = compiled_graphs[request.workflow_id]
    
    # Create GlobalState from dict
    workflow_state = GlobalState(**request.state)
    
    # Create config with thread_id for checkpointing
    config = {"configurable": {"thread_id": request.workflow_id}}
    
    try:
        # Execute graph
        result = await compiled_graph.ainvoke(workflow_state, config)
        
        # Convert result to dict - it should be a GlobalState (Pydantic model)
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()  # Older Pydantic versions
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"data": str(result)}
            
        return {
            "status": "success",
            "result": result_dict,
            "workflow_id": request.workflow_id
        }
    except ValueError as e:
        # Log the full error for debugging
        import traceback
        error_details = {
            "error_type": "ValueError",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "workflow_state": workflow_state.model_dump() if hasattr(workflow_state, 'model_dump') else str(workflow_state)
        }
        print(f"ValueError Details: {error_details}")
        raise HTTPException(status_code=500, detail=f"ValueError: {str(e)}")
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"Error Details: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)