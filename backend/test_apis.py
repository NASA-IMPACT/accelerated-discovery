"""
Test script for the updated backend APIs with resume functionality.
"""

import asyncio
import json
import httpx


async def test_simple_workflow(client: httpx.AsyncClient, base_url: str):
    """Test simple workflow (just literature search)."""
    print("=== Testing Simple Workflow ===")
    
    # 1. Create plan
    print("\n1. Creating simple plan...")
    response = await client.post(
        f"{base_url}/api/workflow/plan",
        json={"message": "What is the impact of climate change on coral reefs?", "workflow_type": "simple"}
    )
    print(f"Status: {response.status_code}")
    plan_data = response.json()
    workflow_id = plan_data["workflow_id"]
    print(f"Workflow ID: {workflow_id}")
    print(f"Graph nodes: {plan_data['graph']['nodes']}")
    print(f"Graph edges: {plan_data['graph']['edges']}")
    
    # 2. Execute workflow
    print("\n2. Executing workflow...")
    initial_state = {
        "inputs": {},
        "outputs": {},
        "messages": [],
        "node_states": {
            "lit_search": {
                "inputs": {"query": "What is the impact of climate change on coral reefs?"},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {},
                "steps": {}
            }
        }
    }
    
    response = await client.post(
        f"{base_url}/api/workflow/execute",
        json={"state": initial_state, "workflow_id": workflow_id}
    )
    print(f"Execute Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        # Print search results
        lit_search_outputs = result.get("result", {}).get("node_states", {}).get("lit_search", {}).get("outputs", {})
        print(f"Query: {lit_search_outputs.get('query')}")
        print(f"Status: {lit_search_outputs.get('status')}")
        if "search_results" in lit_search_outputs:
            results = lit_search_outputs["search_results"].get("results", [])
            print(f"Results: {len(results)} items found")
            for i, r in enumerate(results[:2]):  # Show first 2 results
                print(f"  {i+1}. {r.get('title', 'No title')}")
    else:
        print(f"Error: {response.text}")
    
    return workflow_id


async def test_full_workflow(client: httpx.AsyncClient, base_url: str):
    """Test full workflow (lit search -> code search -> report)."""
    print("\n\n=== Testing Full Workflow ===")
    
    # 1. Create plan
    print("\n1. Creating full workflow plan...")
    response = await client.post(
        f"{base_url}/api/workflow/plan",
        json={"message": "Research on machine learning for climate modeling", "workflow_type": "full"}
    )
    print(f"Status: {response.status_code}")
    plan_data = response.json()
    workflow_id = plan_data["workflow_id"]
    print(f"Workflow ID: {workflow_id}")
    print(f"Graph nodes: {plan_data['graph']['nodes']}")
    print(f"Graph edges: {plan_data['graph']['edges']}")
    
    # 2. Execute workflow
    print("\n2. Executing full workflow...")
    initial_state = {
        "inputs": {},
        "outputs": {},
        "messages": [],
        "node_states": {
            "lit_search": {
                "inputs": {"query": "machine learning for climate modeling"},
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
    
    response = await client.post(
        f"{base_url}/api/workflow/execute",
        json={"state": initial_state, "workflow_id": workflow_id}
    )
    print(f"Execute Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        
        # Check report generation output
        report_outputs = result.get("result", {}).get("node_states", {}).get("report_generation", {}).get("outputs", {})
        if "report" in report_outputs:
            report = report_outputs["report"]
            print(f"\nGenerated Report Summary: {report.get('summary')}")
            print(f"Literature findings: {len(report.get('literature_findings', []))} items")
            print(f"Code findings: {len(report.get('code_findings', []))} items")
    else:
        print(f"Error: {response.text}")
    
    return workflow_id, result if response.status_code == 200 else None


async def test_resume_workflow(client: httpx.AsyncClient, base_url: str, workflow_id: str, previous_state: dict):
    """Test resuming a workflow from a specific node."""
    print("\n\n=== Testing Resume Functionality ===")
    
    # Resume from code_search node (skip lit_search)
    print("\n1. Resuming workflow from 'code_search' node...")
    
    # Modify the state to simulate that lit_search is already done
    resume_state = previous_state["result"]
    
    response = await client.post(
        f"{base_url}/api/workflow/resume",
        json={
            "state": resume_state,
            "workflow_id": workflow_id,
            "start_node": "code_search"
        }
    )
    
    print(f"Resume Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully resumed from: {result.get('resumed_from')}")
        
        # Check if code search was executed again
        code_outputs = result.get("result", {}).get("node_states", {}).get("code_search", {}).get("outputs", {})
        print(f"Code search status: {code_outputs.get('status')}")
    else:
        print(f"Error: {response.text}")


async def test_list_workflows(client: httpx.AsyncClient, base_url: str):
    """Test listing all workflows."""
    print("\n\n=== Testing List Workflows ===")
    
    response = await client.get(f"{base_url}/api/workflow/list")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total workflows: {data['count']}")
        print(f"Workflow IDs: {data['workflows'][:5]}...")  # Show first 5


async def test_get_workflow_status(client: httpx.AsyncClient, base_url: str, workflow_id: str):
    """Test getting specific workflow status."""
    print("\n\n=== Testing Get Workflow Status ===")
    
    response = await client.get(f"{base_url}/api/workflow/{workflow_id}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Workflow exists: {data['exists']}")
        print(f"Checkpointer type: {data['checkpointer_type']}")


async def main():
    """Run all API tests."""
    base_url = "http://localhost:8000"
    
    print("=== Testing Updated AKD Backend APIs ===\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test simple workflow
        simple_workflow_id = await test_simple_workflow(client, base_url)
        
        # Test full workflow
        full_workflow_id, full_workflow_state = await test_full_workflow(client, base_url)
        
        # Test resume functionality (only if full workflow succeeded)
        if full_workflow_state:
            await test_resume_workflow(client, base_url, full_workflow_id, full_workflow_state)
        
        # Test list workflows
        await test_list_workflows(client, base_url)
        
        # Test get workflow status
        await test_get_workflow_status(client, base_url, simple_workflow_id)


if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    asyncio.run(main())
    
    print("\n\nAll tests completed!")