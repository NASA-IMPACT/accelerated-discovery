###Dummy agent used to test websocket connections

import asyncio
import json
import os
from typing import Annotated, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The add_messages function in the annotation defines how this state key should be updated.
    # In this case it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


llm = init_chat_model(os.environ["FAST_LLM"])


@tool
async def get_weather(city: str) -> str:
    """Get weather for location."""

    for i in range(5):
        logger.info(f"Progress: step {i}")
        await asyncio.sleep(1)

    return f"It's always sunny in {city}!"


tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)


def get_graph_builder():
    """
    Dummy agent graph builder used to test websocket connections
    """
    graph_builder = StateGraph(State)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""

        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        async def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                tool_result = await tool.ainvoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    ),
                )
            return {"messages": outputs}

    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    def route_tools(
        state: State,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            state_list: list = cast(list, state)
            msg: BaseMessage = state_list[-1]
        elif messages := state.get("messages", []):
            msg: BaseMessage = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if isinstance(msg, AIMessage):
            msg = cast(AIMessage, msg)
            if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                return "tools"
        return END

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    # memory = InMemorySaver()
    # graph = graph_builder.compile() #checkpointer=memory
    return graph_builder
