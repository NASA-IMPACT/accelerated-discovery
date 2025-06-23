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
async def test_websocket_echo():
    async with websockets.connect("ws://localhost:8999/ws") as websocket:
        message = "What is the weather in London?"
        await websocket.send(message)
        #response = await websocket.recv()
        received_messages = []
        async for msg in websocket:
            if msg == "##END##":  # Stop when END is received
                break
            received_messages.append(msg)

        #print(received_messages)
        assert "sunny" in received_messages[-1].lower()