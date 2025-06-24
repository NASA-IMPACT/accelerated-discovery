import json
import subprocess
import time

import pytest
import websockets


@pytest.fixture(scope="session", autouse=True)
def uvicorn_server():
    # Start the server
    process = subprocess.Popen(
        ["uvicorn", "akd.web.web:app", "--host", "localhost", "--port", "8999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(1)

    yield

    process.terminate()
    process.wait()


@pytest.mark.asyncio
async def test_websocket():
    """
        Test the websocket connection for the graph. 
        This is a simple test to ensure that the websocket is working.
    """
    async with websockets.connect("ws://localhost:8999/ws?thread_id=34") as websocket:
        message = "What is the weather in London?"
        jsonData = {
            "thread_id": 34,
            "message": message
        }
        await websocket.send(json.dumps(jsonData))
        #response = await websocket.recv()
        received_messages = []
        async for msg in websocket:
            if msg == "##END##":  # Stop when END is received
                break
            received_messages.append(msg)

        assert "sunny" in received_messages[-1].lower()
        

@pytest.mark.asyncio
async def test_memory():
    """
        Test that the agent/graph can restore/remember details from the last conversation.
    """
    async with websockets.connect("ws://localhost:8999/ws?thread_id=34") as websocket:
        message = "In which country this place is?"
        jsonData = {
            "thread_id": 34,
            "message": message
        }
        await websocket.send(json.dumps(jsonData))
        #response = await websocket.recv()
        received_messages = []
        async for msg in websocket:
            if msg == "##END##":  # Stop when END is received
                break
            received_messages.append(msg)
        
        assert "england" in received_messages[-1].lower() or "united kingdom" in received_messages[-1].lower()