#uvicorn akd.web.web:app --reload
import uuid

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from starlette.websockets import WebSocketState

from akd.agents.dummy_agent import get_graph_builder

#app
app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    memory = InMemorySaver()
    config = {"thread_id": uuid.uuid4()}
    graph = get_graph_builder().compile(checkpointer=memory)
    while True:
        user_input = await websocket.receive_text()
        await websocket.send_text(f"UserMessage: {user_input}")
        await websocket.send_text("Agent thinking...")
        await stream_graph_updates(user_input,graph,websocket,config)
        await websocket.send_text("##END##") #denoting end of conversation from the agent.



async def stream_graph_updates(user_input: str, graph:CompiledStateGraph, websocket:WebSocket, config=None):
    event: tuple
    async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},stream_mode=["updates", "custom"], config=config, checkpoint_during=True ): # type: ignore #,stream_mode="updates" 
        if event[0]== "updates":
            for value in event[1].values():
                message = value["messages"][-1]
                message.pretty_print()
                
                if websocket.client_state == WebSocketState.CONNECTED and websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_text(f"{type(message).__name__}: {message.content}")
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            await websocket.send_text(f"Tool Call ==> name : {tool_call['name']}, args: {tool_call['args']}")



##NOTE sample page to test websocket. need to be removed
##TODO to be removed
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off" value="What is the weather in London?" style="width:20em;"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)
