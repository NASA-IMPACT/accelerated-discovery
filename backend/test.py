"""
Test script for the backend APIs.
"""

import asyncio
import json
from uuid import uuid4

import httpx


async def test_plan_api(client: httpx.AsyncClient, base_url: str, query: str) -> str:
    """Test the plan creation API."""
    print("Testing POST /api/plan")
    response = await client.post(
        f"{base_url}/api/plan",
        json={"message": query}
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}\n")
    return result.get("workflow_id")


async def test_execute_api(client: httpx.AsyncClient, base_url: str, workflow_id: str, query: str):
    """Test the execute API."""
    print("Testing POST /api/execute")
    
    initial_state = {
        "inputs": {},
        "outputs": {},
        "messages": [],
        "node_states": {
            "search_node": {
                "inputs": {"query": query},
                "outputs": {},
                "messages": [],
                "input_guardrails": {},
                "output_guardrails": {}
            }
        }
    }
    
    response = await client.post(
        f"{base_url}/api/execute",
        json={"state": initial_state, "workflow_id": workflow_id}
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    
    search_node_output = result.get("result", {}).get("node_states", {}).get("search_node", {}).get("outputs", {})
    print(f"Search results: {search_node_output}")


async def test_invalid_workflow(client: httpx.AsyncClient, base_url: str):
    """Test execute API with invalid workflow ID."""
    print("\nTesting POST /api/execute with non-existent workflow")
    response = await client.post(
        f"{base_url}/api/execute",
        json={
            "state": {"workflow_id": str(uuid4()), "workflow_config": {}},
            "workflow_id": str(uuid4())
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


async def main():
    """Run all API tests."""
    base_url = "http://localhost:8000"
    query = "What is the impact of climate change on coral reefs?"
    
    print("=== Testing AKD Backend APIs ===\n")
    
    async with httpx.AsyncClient(timeout=3000.0) as client:
        # Test plan creation
        workflow_id = await test_plan_api(client, base_url, query)
        
        # Test execution
        await test_execute_api(client, base_url, workflow_id, query)
        
        # Test invalid workflow
        await test_invalid_workflow(client, base_url)


if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    asyncio.run(main())
    
    print("\n\nTests completed!")