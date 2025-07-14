"""
Test script for the backend APIs.
"""

import asyncio
import httpx
import json
from uuid import uuid4


async def test_apis():
    """Test the backend APIs."""
    base_url = "http://localhost:8000"
    
    # Increase timeout to 30 seconds for all requests
    async with httpx.AsyncClient(timeout=3000.0) as client:
        print("=== Testing AKD Backend APIs ===\n")
        
        # 1. Test plan creation
        print("1. Testing POST /api/plan")
        plan_request = {
            "message": "What is the impact of climate change on coral reefs?"
        }
        try:
            response = await client.post(
                f"{base_url}/api/plan",
                json=plan_request
            )
            print(f"Status: {response.status_code}")
            plan_response = response.json()
            print(f"Response: {json.dumps(plan_response, indent=2)}\n")
            
            # Extract workflow_id for next test
            workflow_id = plan_response.get("workflow_id")
            
            # 2. Test graph execution
            if workflow_id:
                print("2. Testing POST /api/execute")
                
                # Create initial state for the search_node
                initial_state = {
                    "inputs": {},  # Global inputs
                    "output": {},  # Global output
                    "messages": [],  # Global messages
                    "node_states": {
                        "search_node": {
                            "inputs": {
                                "query": plan_request["message"]  # The query for the search
                            },
                            "output": {},
                            "messages": [],
                            "supervisor_state": {
                                "inputs": {
                                    "query": plan_request["message"]
                                },
                                "output": {},
                                "messages": [],
                                "tool_calls": [],
                                "steps": {}
                            },
                            "input_guardrails": {},
                            "output_guardrails": {}
                        }
                    }
                }
                
                execute_request = {
                    "state": initial_state,
                    "workflow_id": workflow_id
                }
                
                try:
                    response = await client.post(
                        f"{base_url}/api/execute",
                        json=execute_request
                    )
                    print(f"Status: {response.status_code}")
                    execute_response = response.json()
                    
                    # Pretty print the response but limit output
                    if execute_response.get("status") == "success":
                        print("Response: Success!")
                        print(f"Workflow ID: {execute_response.get('workflow_id')}")
                        
                        # Print a summary of the result
                        result = execute_response.get("result", {})
                        print(f"Workflow Status: {result.get('workflow_status')}")
                        
                        # Check if search_node completed
                        search_node_state = result.get("node_states", {}).get("search_node", {})
                        if search_node_state.get("output"):
                            print("Search completed successfully!")
                            output = search_node_state["output"]
                            print(f"Search results: {output}")
                        else:
                            print("No search results yet.")
                    else:
                        print(f"Response: {json.dumps(execute_response, indent=2)}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                
            else:
                print("No workflow_id received from plan creation")
                
        except Exception as e:
            print(f"Error: {e}\n")
        
        # 3. Test execution with non-existent workflow
        print("\n3. Testing POST /api/execute with non-existent workflow")
        fake_request = {
            "state": {
                "workflow_id": str(uuid4()),
                "workflow_config": {}
            },
            "workflow_id": str(uuid4())
        }
        try:
            response = await client.post(
                f"{base_url}/api/execute",
                json=fake_request
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error: {e}")


async def test_multiple_workflows():
    """Test creating and executing multiple workflows."""
    base_url = "http://localhost:8000"
    
    print("\n\n=== Testing Multiple Workflows ===\n")
    
    queries = [
        "What are the latest advances in quantum computing?",
        "How does machine learning impact healthcare?",
        "What is the role of AI in climate science?"
    ]
    
    # Increase timeout to 30 seconds for all requests
    async with httpx.AsyncClient(timeout=30.0) as client:
        workflow_ids = []
        
        # Create multiple plans
        print("Creating multiple plans...")
        for query in queries:
            response = await client.post(
                f"{base_url}/api/plan",
                json={"message": query}
            )
            if response.status_code == 200:
                workflow_id = response.json()["workflow_id"]
                workflow_ids.append((workflow_id, query))
                print(f"Created workflow {workflow_id} for: {query}")
        
        print(f"\nCreated {len(workflow_ids)} workflows")
        
        # Execute each workflow
        print("\nExecuting workflows...")
        for workflow_id, query in workflow_ids:
            initial_state = {
                "inputs": {},
                "output": {},
                "messages": [],
                "node_states": {
                    "search_node": {
                        "inputs": {"query": query},
                        "output": {},
                        "messages": [],
                        "supervisor_state": {
                            "inputs": {"query": query},
                            "output": {},
                            "messages": [],
                            "tool_calls": [],
                            "steps": {}
                        },
                        "input_guardrails": {},
                        "output_guardrails": {}
                    }
                }
            }
            
            response = await client.post(
                f"{base_url}/api/execute",
                json={
                    "state": initial_state,
                    "workflow_id": workflow_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status")
                print(f"Workflow {workflow_id}: {status}")
            else:
                print(f"Workflow {workflow_id}: Failed with status {response.status_code}")


if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    # Run tests
    asyncio.run(test_apis())
    asyncio.run(test_multiple_workflows())
    
    print("\n\nTests completed!")