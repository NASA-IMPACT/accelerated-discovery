import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from loguru import logger
    return


@app.cell
def _():
    from pydantic import Field, BaseModel
    return BaseModel, Field


@app.cell
def _():
    from typing import Any
    from pprint import pprint
    return (Any,)


@app.cell
def _():
    import random
    return


@app.cell
def _():
    from akd.tools.search import (
        SearxNGSearchToolInputSchema,
        SearxNGSearchToolOutputSchema,
        SearxNGSearchTool,
    )
    return


@app.cell
def _():
    from akd.agents.query import (
        QueryAgent,
        QueryAgentInputSchema,
        QueryAgentOutputSchema,
    )
    return


@app.cell
def _():
    from akd.tools.utils import tool_wrapper
    return


@app.cell
def _():
    from akd.tools import BaseTool, BaseToolConfig
    from akd._base import InputSchema, OutputSchema
    from akd.structures import SearchResultItem
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langgraph.prebuilt import create_react_agent
    return


@app.cell
def _():
    from langgraph.graph import StateGraph, MessagesState, START, END
    return END, StateGraph


@app.cell
def _(mo):
    mo.md(r"""# What is state in lnaggraph""")
    return


@app.cell
def _():
    import time
    return


@app.cell
def _():
    # from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    return (InMemorySaver,)


@app.cell
def _(Any, BaseModel, Field):
    class DummyGlobalState(BaseModel):
        ajinkya: str | None = None
        slesa: float | None = None
        paridhi: float | None = None
        kumar: int | None = None
        nish: Any | None = Field(default=None)
    return (DummyGlobalState,)


@app.cell
def _(DummyGlobalState):
    def random_node(state: DummyGlobalState):
        print("Executing node :: ajinkya random node")
        return dict(ajinkya="not Wow")
    return (random_node,)


@app.cell
def _(DummyGlobalState):
    def slesa_node(state: DummyGlobalState):
        print("Executing node:: slesa node")
        ajinkya_value = state.ajinkya

        # return dict(slesa=ajinkya_value.lower() == "wow")
        return dict(slesa=state.slesa + 1)
    return (slesa_node,)


@app.cell
def _(DummyGlobalState):
    def paridhi_node(state: DummyGlobalState):
        print("Executing node:: paridhi node")
        return dict(slesa=state.slesa + 1)
    return (paridhi_node,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# state manipualation time travel""")
    return


@app.cell
def _(InMemorySaver):
    memory = InMemorySaver()
    config = {"configurable": {"thread_id": "123"}}
    return config, memory


@app.cell
def _(config):
    config
    return


@app.cell
def _(DummyGlobalState):
    initial_state = DummyGlobalState(
        slesa=1,
    )
    initial_state
    return (initial_state,)


@app.cell
async def _(
    DummyGlobalState,
    END,
    StateGraph,
    config,
    initial_state,
    memory,
    paridhi_node,
    random_node,
    slesa_node,
):
    graph = StateGraph(DummyGlobalState)
    graph.add_node("ajinkya_node", random_node)
    graph.add_node("slesa_node", slesa_node)
    graph.add_node("paridhi_node", paridhi_node)
    graph.add_edge("ajinkya_node", "slesa_node")
    graph.add_edge("slesa_node", "paridhi_node")
    graph.set_entry_point("ajinkya_node")
    graph.add_edge("paridhi_node", END)

    workflow = graph.compile(checkpointer=memory)

    _result = await workflow.ainvoke(initial_state, config=config)
    _result
    return (workflow,)


@app.cell
def _(config, workflow):
    # The states are returned in reverse chronological order.
    for state in workflow.get_state_history(config):
        print(state)
        print(state.config["configurable"]["checkpoint_id"])
        print("-"*100)
    return


@app.function
def print_workflow(w, cfg):
    # The states are returned in reverse chronological order.
    for state in w.get_state_history(cfg):
        print(state)
        print(state.config["configurable"]["checkpoint_id"])
        print("-"*42)


@app.cell
def _(config, workflow):
    print_workflow(workflow, config)
    return


@app.cell
def _(config, workflow):
    states = list(workflow.get_state_history(config))
    _config = states[2].config
    print(_config)

    new_state = workflow.update_state(_config, dict(slesa=99))

    print(new_state)
    return (new_state,)


@app.cell
def _(new_state):
    new_state
    return


@app.cell
async def _(new_state, workflow):
    result = await workflow.ainvoke(None, config=new_state)
    return (result,)


@app.cell
def _(result):
    result
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
