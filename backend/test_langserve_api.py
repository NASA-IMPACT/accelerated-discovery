"""
Test script for the updated LangServe API implementation.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_langserve_workflow():
    """Test the LangServe workflow mimicking FastAPI approach."""
    
    print("Testing AKD LangServe Backend (FastAPI-style)...")
    
    # Step 1: Plan a workflow
    print("\n1. Planning a workflow...")
    plan_payload = {
        "message": "What are the latest advances in quantum computing?",
        "workflow_type": "simple"
    }
    
    plan_response = requests.post(
        f"{BASE_URL}/workflow/plan/invoke",
        json={"input": plan_payload}
    )
    
    if plan_response.status_code != 200:
        print(f"Error planning workflow: {plan_response.text}")
        return
    
    plan_result = plan_response.json()
    print(f"Plan response: {json.dumps(plan_result, indent=2)}")
    
    # Extract workflow_id from the output
    workflow_id = plan_result.get("output", {}).get("workflow_id")
    if not workflow_id:
        print("Error: No workflow_id in response")
        return
    
    print(f"\nWorkflow ID: {workflow_id}")
    
    # Step 2: Execute the workflow
    print("\n2. Executing the workflow...")
    execute_payload = {
        "workflow_id": workflow_id,
        "query": "What are the latest advances in quantum computing?"
    }
    
    execute_response = requests.post(
        f"{BASE_URL}/workflow/execute/invoke",
        json={"input": execute_payload}
    )
    
    if execute_response.status_code != 200:
        print(f"Error executing workflow: {execute_response.text}")
        return
    
    execute_result = execute_response.json()
    print(f"Execute response: {json.dumps(execute_result, indent=2)}")
    
    # Step 3: List workflows
    print("\n3. Listing all workflows...")
    list_response = requests.get(f"{BASE_URL}/api/workflow/list")
    
    if list_response.status_code == 200:
        list_result = list_response.json()
        print(f"Workflows: {json.dumps(list_result, indent=2)}")
    
    # Step 4: Get workflow status
    print(f"\n4. Getting status for workflow {workflow_id}...")
    status_response = requests.get(f"{BASE_URL}/api/workflow/{workflow_id}")
    
    if status_response.status_code == 200:
        status_result = status_response.json()
        print(f"Workflow status: {json.dumps(status_result, indent=2)}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    # Give warning about starting the server
    print("⚠️  Make sure the LangServe API is running on port 8001")
    print("   Run: python api-langserve.py")
    print("   Waiting 3 seconds before starting tests...\n")
    time.sleep(3)
    
    test_langserve_workflow()