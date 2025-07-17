"""
Simplified LangServe API for AKD Backend.
This version mimics the FastAPI approach with proper workflow management.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from akd.nodes.states import GlobalState, NodeTemplateState
from akd.serializers import AKDSerializer
from planner import create_simple_plan, create_full_plan

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Create FastAPI app
app = FastAPI(
    title="AKD LangServe Backend",
    description="LangServe-based API for AKD multi-agent workflows"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory saver for checkpointing with AKD serializer
memory = MemorySaver(serde=AKDSerializer())

# Store compiled graphs by workflow_id (mimicking FastAPI approach)
compiled_graphs = {}


# Request models (matching FastAPI approach)
class WorkflowPlanRequest(BaseModel):
    message: str
    workflow_type: str = "simple"  # simple, full

class WorkflowExecuteRequest(BaseModel):
    workflow_id: str
    query: str  # Direct query instead of full state for LangServe simplicity

# Transform functions to prepare state
def prepare_simple_state(input_dict: Dict[str, Any]) -> GlobalState:
    """Prepare state for simple workflow."""
    query = input_dict.get("query", "")
    state_dict = {
        "inputs": {},
        "outputs": {},
        "messages": [],
        "node_states": {
            "lit_search": {
                "inputs": {"query": query},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {},
                "steps": {}
            }
        }
    }
    return GlobalState(**state_dict)


def prepare_full_state(input_dict: Dict[str, Any]) -> GlobalState:
    """Prepare state for full workflow."""
    query = input_dict.get("query", "")
    state_dict = {
        "inputs": {},
        "outputs": {},
        "messages": [],
        "node_states": {
            "lit_search": {
                "inputs": {"query": query},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {},
                "steps": {}
            },
            "code_search": {
                "inputs": {},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {},
                "steps": {}
            },
            "report_generation": {
                "inputs": {},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {},
                "steps": {}
            }
        }
    }
    return GlobalState(**state_dict)


# Workflow planning function (creates and stores compiled graphs)
async def plan_workflow(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create a workflow plan and store compiled graph."""
    request = WorkflowPlanRequest(**input_dict)
    
    # Create the appropriate graph based on workflow type
    if request.workflow_type == "full":
        graph = create_full_plan()
    else:
        graph = create_simple_plan()
    
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
        "message": f"Graph created for query: {request.message}",
        "workflow_type": request.workflow_type
    }

# Workflow execution function
async def execute_workflow(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a previously created workflow."""
    request = WorkflowExecuteRequest(**input_dict)
    
    if request.workflow_id not in compiled_graphs:
        return {
            "error": f"Workflow not found: {request.workflow_id}",
            "status": "error"
        }
    
    # Prepare state based on workflow type
    compiled_graph = compiled_graphs[request.workflow_id]
    
    # Determine workflow type by checking nodes
    if "report_generation" in str(compiled_graph):
        state = prepare_full_state({"query": request.query})
    else:
        state = prepare_simple_state({"query": request.query})
    
    # Execute with proper config
    config = {"configurable": {"thread_id": request.workflow_id}}
    
    try:
        result = await compiled_graph.ainvoke(state, config)
        
        # Convert to JSON-serializable format
        import json
        from pydantic import HttpUrl
        
        def custom_serializer(obj):
            if isinstance(obj, HttpUrl):
                return str(obj)
            elif hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        # Serialize and deserialize to ensure everything is JSON-compatible
        result_json = json.dumps(result, default=custom_serializer)
        result_dict = json.loads(result_json)
        
        return {
            "status": "success",
            "result": result_dict,
            "workflow_id": request.workflow_id
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "workflow_id": request.workflow_id
        }

# Create runnable chains for LangServe
plan_chain = RunnableLambda(plan_workflow).with_types(
    input_type=Dict[str, str],
    output_type=Dict[str, Any]
)

execute_chain = RunnableLambda(execute_workflow).with_types(
    input_type=Dict[str, str],
    output_type=Dict[str, Any]
)

# Add routes using LangServe
add_routes(
    app,
    plan_chain,
    path="/workflow/plan",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default",
)

add_routes(
    app,
    execute_chain,
    path="/workflow/execute",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default",
)

# Add a root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AKD LangServe Backend (FastAPI-style)",
        "endpoints": {
            "workflow_plan": {
                "invoke": "POST /workflow/plan/invoke",
                "stream": "POST /workflow/plan/stream", 
                "playground": "GET /workflow/plan/playground",
                "description": "Create a workflow plan with nodes and edges"
            },
            "workflow_execute": {
                "invoke": "POST /workflow/execute/invoke",
                "stream": "POST /workflow/execute/stream",
                "playground": "GET /workflow/execute/playground",
                "description": "Execute a previously created workflow"
            }
        },
        "workflow": {
            "step1": "Call /workflow/plan/invoke with {message: 'your query', workflow_type: 'simple' or 'full'}",
            "step2": "Get workflow_id from response",
            "step3": "Call /workflow/execute/invoke with {workflow_id: 'id', query: 'your query'}"
        },
        "playground_instructions": "Visit /workflow/plan/playground or /workflow/execute/playground for interactive UI"
    }


# Additional endpoints to match FastAPI implementation
@app.get("/api/workflow/list")
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
    
    return {
        "workflow_id": workflow_id,
        "exists": True,
        "checkpointer_type": type(compiled_graphs[workflow_id].checkpointer).__name__
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "akd-langserve"}


if __name__ == "__main__":
    import uvicorn
    print("Starting AKD LangServe Backend (FastAPI-style)...")
    print("Workflow planning playground: http://localhost:8001/workflow/plan/playground")
    print("Workflow execution playground: http://localhost:8001/workflow/execute/playground")
    print("\nWorkflow steps:")
    print("1. Plan a workflow at /workflow/plan/playground")
    print("2. Execute the workflow at /workflow/execute/playground using the workflow_id")
    uvicorn.run(app, host="0.0.0.0", port=8001)