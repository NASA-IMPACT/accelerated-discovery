"""
Simple API for executing planner graph.
"""

import sys
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

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
    """Execute a previously created graph with given state."""
    if request.workflow_id not in compiled_graphs:
        raise HTTPException(status_code=404, detail=f"Graph not found for workflow_id: {request.workflow_id}")
    
    try:
        compiled_graph = compiled_graphs[request.workflow_id]
        workflow_state = GlobalState(**request.state)
        config = {"configurable": {"thread_id": request.workflow_id}}
        
        result = await compiled_graph.ainvoke(workflow_state, config)
        
        # Handle different result formats
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"data": str(result)}
        
        return {
            "status": "success",
            "result": result_dict,
            "workflow_id": request.workflow_id
        }
    except Exception as e:
        import traceback
        print(f"ERROR in execute_graph: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)