import json

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from akd.agents.dummy_agent import get_graph_builder

# app
app = FastAPI()

memory: InMemorySaver = InMemorySaver()


class Message(BaseModel):
    """
    A message that is sent from UI to the agent or from the agent to the UI through webscoket.

    Attributes:
        thread_id: (int) The id of the message thread.
        message: (str) The message content.
    """

    thread_id: int
    message_type: str = "UserMessage"
    message: str


# uvicorn akd.web.web:app --reload
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    The websocket endpoint that handles the communication between the browser and the agent.
    It loads the graph from the checkpoint(if available) and resumes the conversation with the user.

    To run `uvicorn akd.web.web:app --reload`
    """
    await websocket.accept()
    thread_id = websocket.query_params.get("thread_id")

    graph = get_graph_builder().compile(checkpointer=memory)
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if state and state.values:  # restore the state if the thread is already present.
        await graph.aupdate_state(config, state)
        messages = await get_past_messages(config, graph)
        [
            await websocket.send_text(f"{message.message_type}:{message.message}")
            for message in messages
        ]
    while True:
        msg_str = await websocket.receive_text()
        user_message = Message.model_validate(json.loads(msg_str))
        await websocket.send_text(f"UserMessage: {user_message.message}")
        await websocket.send_text("Agent thinking...")
        await stream_graph_updates(
            user_message.message,
            graph,
            websocket,
            config["configurable"],
        )
        await websocket.send_text(
            "##END##",
        )  # denoting end of conversation from the agent.


async def get_past_messages(config: dict, graph: CompiledStateGraph) -> list[Message]:
    """
    Get the past messages for a given thread id.
    Returns empty array if its a new thread.

    Args:
        config: Thread identifier config for the chat
        graph: The current compiled graph
    """
    return_messages = []
    state = graph.get_state(config)
    if state.values and len(state.values["messages"]) > 0:
        for message in state.values["messages"]:
            message_obj = Message(
                thread_id=config["configurable"]["thread_id"],
                message_type=type(message).__name__,
                message=message.content,
            )
            return_messages.append(message_obj)

    return return_messages


async def stream_graph_updates(
    user_input: str,
    graph: CompiledStateGraph,
    websocket: WebSocket,
    config=None,
):
    """
    Run a graph with the user input and stream the graph updates over websocket.

    Args:
        user_input: The user input from the UI.
        graph: The current compiled graph
        websocket: The websocket object to send the updates to the browser.
        config: The config for the graph run. (Thread id in config)
    """
    event: tuple
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=["updates", "custom"],
        config=config,
        checkpoint_during=True,
    ):  # type: ignore #,stream_mode="updates"
        if event[0] == "updates":
            for value in event[1].values():
                message = value["messages"][-1]
                message.pretty_print()

                if (
                    websocket.client_state == WebSocketState.CONNECTED
                    and websocket.application_state == WebSocketState.CONNECTED
                ):
                    await websocket.send_text(
                        f"{type(message).__name__}: {message.content}",
                    )
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            await websocket.send_text(
                                f"Tool Call ==> name : {tool_call['name']}, args: {tool_call['args']}",
                            )


##NOTE sample page to test websocket. need to be removed
##TODO to be removed
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>🤖 Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            Thread ID: <input type="text" id="thread_id" autocomplete="off" value="1" readonly/><br>
            <input type="text" id="messageText" autocomplete="off" value="What is the weather in London?" style="width:20em;"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            function getUrlParameter(param) {
                const urlParams = new URLSearchParams(window.location.search);
                return urlParams.get(param);
            }

            var thread_id = document.getElementById("thread_id")
            if (getUrlParameter('thread_id')) {
                thread_id.value = getUrlParameter('thread_id');
            } else {
                // If the parameter is not found, you can set a random value as a fallback
                thread_id.value = Math.floor(Math.random() * 100) + 1;
            }

            var ws = new WebSocket("ws://localhost:8000/ws?thread_id=" + thread_id.value);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                var thread_id = document.getElementById("thread_id")
                var jsonData = {
                "thread_id": thread_id.value,
                "message": input.value
                };
                ws.send(JSON.stringify(jsonData))
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
