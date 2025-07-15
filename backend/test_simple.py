"""
Simple test for the API.
"""

import asyncio
import httpx
import json

async def test():
    base_url = "http://localhost:8000"
    query = "What is the impact of climate change on coral reefs?"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create plan
        print("Creating plan...")
        response = await client.post(
            f"{base_url}/api/plan",
            json={"message": query}
        )
        print(f"Plan Status: {response.status_code}")
        plan_data = response.json()
        workflow_id = plan_data["workflow_id"]
        print(f"Workflow ID: {workflow_id}")
        
        # Execute
        print("\nExecuting...")
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
                    "output_guardrails": {},
                    "steps": {}
                }
            }
        }
        
        response = await client.post(
            f"{base_url}/api/execute",
            json={"state": initial_state, "workflow_id": workflow_id}
        )
        print(f"Execute Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            # Print search results
            outputs = result.get("result", {}).get("node_states", {}).get("search_node", {}).get("outputs", {})
            print(f"Query: {outputs.get('query')}")
            print(f"Results: {len(outputs.get('search_results', {}).get('results', []))} items found")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test())