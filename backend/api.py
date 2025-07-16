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

from akd.nodes.states import GlobalState, NodeTemplateState
from akd.serializers import AKDSerializer
from planner import create_planner_graph, create_simple_search_graph, create_full_workflow_graph

# Create FastAPI app
app = FastAPI(title="AKD Simple Backend")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory saver for checkpointing with AKD serializer
memory = MemorySaver(serde=AKDSerializer())

# Store compiled graphs by workflow_id
compiled_graphs = {}


class ChatRequest(BaseModel):
    message: str
    workflow_type: str = "simple"  # simple, full


class GraphStructure(BaseModel):
    nodes: list[str]
    edges: list[tuple[str, str]]


@app.post("/api/plan")
async def create_plan(request: ChatRequest):
    """
    Accept chat message and return graph structure.
    Store compiled graph for later execution.
    """
    # Create the appropriate graph based on workflow type
    if request.workflow_type == "full":
        graph = create_full_workflow_graph()
    else:
        graph = create_simple_search_graph()
    
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


class WorkflowRequest(BaseModel):
    query: str
    workflow_type: str = "simple"  # simple, full


class ResumeRequest(BaseModel):
    state: Dict[str, Any]
    workflow_id: str
    start_node: str  # Node to start execution from


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


@app.post("/api/resume")
async def resume_graph(request: ResumeRequest):
    """
    Resume execution of a workflow from a specific node.
    This allows skipping already executed nodes and starting from a particular point.
    """
    if request.workflow_id not in compiled_graphs:
        raise HTTPException(status_code=404, detail=f"Graph not found for workflow_id: {request.workflow_id}")
    
    try:
        compiled_graph = compiled_graphs[request.workflow_id]
        workflow_state = GlobalState(**request.state)
        
        # Configure to resume from specific node
        config = {
            "configurable": {
                "thread_id": request.workflow_id,
                "checkpoint_ns": request.start_node  # This tells LangGraph where to resume from
            }
        }
        
        # Get the graph and update it to start from the specified node
        # We need to get the underlying graph to modify entry point
        from planner import create_full_workflow_graph, create_simple_search_graph
        
        # Determine which graph type based on nodes in state
        nodes_in_state = list(workflow_state.node_states.keys())
        if "report_generation" in nodes_in_state:
            graph = create_full_workflow_graph()
        else:
            graph = create_simple_search_graph()
        
        # Override the entry point to start from specified node
        graph.set_entry_point(request.start_node)
        
        # Recompile with the same memory/checkpointer
        resumed_graph = graph.compile(checkpointer=memory)
        
        # Execute from the resume point
        result = await resumed_graph.ainvoke(workflow_state, config)
        
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
            "workflow_id": request.workflow_id,
            "resumed_from": request.start_node
        }
    except Exception as e:
        import traceback
        print(f"ERROR in resume_graph: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows")
async def list_workflows():
    """List all stored workflows."""
    return {
        "workflows": list(compiled_graphs.keys()),
        "count": len(compiled_graphs)
    }


@app.get("/api/workflow/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status and structure of a specific workflow."""
    if workflow_id not in compiled_graphs:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    
    compiled_graph = compiled_graphs[workflow_id]
    
    # Get the graph structure
    # Note: compiled graphs don't expose nodes/edges directly, so we return basic info
    return {
        "workflow_id": workflow_id,
        "exists": True,
        "checkpointer_type": type(compiled_graph.checkpointer).__name__
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)