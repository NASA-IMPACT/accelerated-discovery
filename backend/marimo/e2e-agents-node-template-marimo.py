import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


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
    return (Field,)


@app.cell
def _():
    from typing import Any
    return


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
    return (SearxNGSearchTool,)


@app.cell
def _():
    from akd.agents.query import (
        QueryAgent,
        QueryAgentInputSchema,
        QueryAgentOutputSchema,
    )
    return (QueryAgent,)


@app.cell
def _():
    from akd.tools.utils import tool_wrapper
    return


@app.cell
def _():
    from akd.tools import BaseTool, BaseToolConfig
    from akd._base import InputSchema, OutputSchema
    from akd.structures import SearchResultItem
    return (
        BaseTool,
        BaseToolConfig,
        InputSchema,
        OutputSchema,
        SearchResultItem,
    )


@app.cell
def _(QueryAgent, SearxNGSearchTool):
    SEARCH_TOOL = SearxNGSearchTool()

    QUERY_AGENT = QueryAgent()
    return QUERY_AGENT, SEARCH_TOOL


@app.cell
async def _(QUERY_AGENT):
    _tool = QUERY_AGENT.to_langchain_structured_tool()
    await _tool.ainvoke(input=dict(query="flood landslide nepal", num_queries=5))
    return


@app.cell
async def _(SEARCH_TOOL):
    _tool = SEARCH_TOOL.to_langchain_structured_tool()
    await _tool.ainvoke(input=dict(queries=["flood landslide nepal"]))
    return


@app.cell
def _(mo):
    mo.md(r"""# Node Template imports""")
    return


@app.cell
def _():
    from akd.nodes.states import (
        NodeState,
        NodeTemplateState,
        GlobalState,
        SupervisorState,
    )
    from akd.nodes.templates import AbstractNodeTemplate, SupervisedNodeTemplate
    from akd.nodes.supervisor import (
        ManualSupervisor,
        ReActLLMSupervisor,
    )
    return AbstractNodeTemplate, GlobalState, ManualSupervisor, NodeState


@app.cell
def _(mo):
    mo.md(
        r"""
    # Possible Node Hand-off scenarios

    0. Planner | Human -> START
    1. START -> Lit Search
    2. Lit Search -> Lit Search
    3. Lit Search -> Termination Check
    4. Lit Search -> Data Search
    5. Lit Search -> Code Search
    6. Planner -> FM Inference
    7. Lit Search -> Gap Analysis
    8. Lit Search -> Lit Search 1 | Lit Search 2 | ... [Multiple Lit Search]
    9. Termination Check -> Planner | Human
    10. Termination Check -> Report Generation
    11. Report Generation [END]


    # Planner design

    - Planner has input and output view for each tool
    - Planner can watch the global state
    - Planner defines the initial node states
    - Planner initially parses user message

    # Worflow Samples Scenarios

    ## 1. Planner  -> Lit Search

    - User -> Planner
    - Planner -> Initial State
    - Initial State -> Lit Search [queries, ]
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# langraph imports""")
    return


@app.cell
def _():
    from langgraph.graph import StateGraph, MessagesState, START, END
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Create guardrails

    Use tool_wrapper
    """
    )
    return


app._unparsable_cell(
    r"""
    def always_bad(inp: str) -> str:
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    def always_good(inp: str) -> str:
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    def search_result_min_length(results: list, min_length: int = 10) -> bool:
    """,
    name="_"
)


@app.cell
async def _(always_bad):
    await always_bad.arun(dict(inp="a"))
    return


@app.cell
def _(mo):
    mo.md(r"""## Create node templates""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Use Custom Node Template implementation""")
    return


@app.cell
def _(
    BaseTool,
    BaseToolConfig,
    Field,
    InputSchema,
    OutputSchema,
    SearchResultItem,
):
    class DummyCodeSearchInputSchema(InputSchema):
        """
        Dummy input schema for code search
        """

        queries: list[str] = Field(
            ..., description="Input queries for code search"
        )


    class DummyCodeSearchOutputSchema(OutputSchema):
        """
        Dummy output schema for code search
        """

        results: list[SearchResultItem] = Field(
            ..., description="List of search result items"
        )


    class DummyCodeSearchToolConfig(BaseToolConfig):
        max_results: int = Field(
            default=10, description="Maximum number of search result items"
        )


    class DummyCodeSearchTool(BaseTool):
        input_schema = DummyCodeSearchInputSchema
        output_schema = DummyCodeSearchOutputSchema
        config_schema = DummyCodeSearchToolConfig

        async def _arun(
            self, params: DummyCodeSearchInputSchema
        ) -> DummyCodeSearchOutputSchema:
            res = []
            for query in params.queries:
                res.append(
                    SearchResultItem(
                        url="https://github.com/NASA-IMPACT/accelerated-discovery",
                        content=" A multiagent framework to augment research worflow ",
                        title="NASA-IMPACT/accelerated-discovery",
                        query=query,
                    )
                )
            return DummyCodeSearchOutputSchema(results=res)
    return DummyCodeSearchInputSchema, DummyCodeSearchTool


@app.cell
async def _(DummyCodeSearchInputSchema, DummyCodeSearchTool):
    _tool = DummyCodeSearchTool()

    (
        await _tool.arun(DummyCodeSearchInputSchema(queries=["dummy input"]))
    ).model_dump()
    return


@app.cell
def _(AbstractNodeTemplate, GlobalState, NodeState, SEARCH_TOOL):
    class LiteratureSearchNode(AbstractNodeTemplate):
        async def _execute(
            self,
            node_state: NodeState,
            global_state: GlobalState,
        ) -> NodeState:
            # node_state = node_state.model_copy()
            query = node_state.inputs.get("query")
            node_state.outputs["search"] = await SEARCH_TOOL.arun(
                SEARCH_TOOL.input_schema(queries=[query])
            )
    return


@app.cell
def _(AbstractNodeTemplate, DummyCodeSearchTool, GlobalState, NodeState):
    class CodeSearchNode(AbstractNodeTemplate):
        async def _execute(
            self,
            node_state: NodeState,
            global_state: GlobalState,
        ) -> NodeState:
            # node_state = node_state.model_copy()
            search_tool = DummyCodeSearchTool()
            query = global_state.node_states.get("lit_search").inputs.get("query")
            node_state.outputs["search"] = await search_tool.arun(
                search_tool.input_schema(queries=[query])
            )
    return


@app.cell
async def _(SEARCH_NODE):
    async def _(GlobalState,
        LiteratureSearchNode,
        NodeTemplateState,
        always_bad,
        always_good,
        search_result_min_length,
    ):
        SEARCH_NODE = LiteratureSearchNode(
            node_id="lit_search",
            input_guardrails=[always_bad],
            output_guardrails=[
                (always_good, {"inp": "results"}),
                search_result_min_length,
            ],
            debug=True,
            mutation=True,
        )

        _global_state = GlobalState(
            node_states={
                "lit_search": NodeTemplateState(
                    inputs={
                        "query": "landslide casualties in Nepal last year",  # Same query or different
                    }
                ),
            }
        )

    (await SEARCH_NODE.arun(_global_state)).model_dump()
    return


@app.cell
def _():
    async def _(CodeSearchNode, GlobalState, NodeTemplateState):
        CODE_SEARCH_NODE = CodeSearchNode(
            node_id="code_search",
            debug=True,
            mutation=True,
        )

        _global_state = GlobalState(
            node_states={
                "lit_search": NodeTemplateState(
                    inputs={
                        "query": "landslide casualties in Nepal last year",
                    }
                ),
                "code_search": NodeTemplateState(
                    inputs={
                        "query": "landslide casualties in Nepal last year",
                    }
                ),
            }
        )

        (await CODE_SEARCH_NODE.arun(_global_state)).model_dump()
        return
    return


@app.cell
def _(mo):
    mo.md(r"""### Option 2: Use Supervisor-based template""")
    return


@app.cell
def _(DummyCodeSearchTool, GlobalState, ManualSupervisor, NodeState):
    class CodeSearchSupervisor(ManualSupervisor):
        async def _arun(
            self,
            node_state: NodeState,
            global_state: GlobalState,
        ) -> NodeState:
            # node_state = node_state.model_copy()
            search_tool = DummyCodeSearchTool()
            query = global_state.node_states.get("lit_search").inputs.get("query")
            node_state.outputs["search"] = await search_tool.arun(
                search_tool.input_schema(queries=[query])
            )
    return


@app.cell
def _():
    # _node = SupervisedNodeTemplate(
    #     node_id="code_search",
    #     supervisor=CodeSearchSupervisor(),
    #     debug=True,
    #     mutation=True,
    # )

    # _global_state = GlobalState(
    #     node_states={
    #         "lit_search": NodeTemplateState(
    #             inputs={
    #                 "query": "landslide casualties in Nepal last year",
    #             }
    #         ),
    #         "code_search": NodeTemplateState(
    #             inputs={
    #                 "query": "landslide casualties in Nepal last year",
    #             }
    #         ),
    #     }
    # )

    # (await _node.arun(_global_state)).model_dump()
    return


@app.cell
def _(mo):
    mo.md(r"""# e2e""")
    return


@app.cell
def _():
    from langgraph.checkpoint.memory import InMemorySaver
    from akd.serializers import AKDSerializer
    return AKDSerializer, InMemorySaver


@app.cell
def _(AKDSerializer, InMemorySaver):
    MEMORY = InMemorySaver(serde=AKDSerializer())
    CONFIG = {"configurable": {"thread_id": "123"}}
    return


@app.cell
def _():
    async def _(
        CODE_SEARCH_NODE,
        CONFIG,
        END,
        GlobalState,
        MEMORY,
        NodeTemplateState,
        SEARCH_NODE,
        StateGraph,
    ):
        _graph = StateGraph(GlobalState)


        # _node = SupervisedNodeTemplate(
        #     node_id="code_search",
        #     supervisor=CodeSearchSupervisor(),
        #     debug=True,
        #     mutation=True,
        # )

        # Add nodes
        _graph.add_node("lit_search", SEARCH_NODE.to_langgraph_node(key="lit_search"))
        _graph.add_node(
            "code_search", CODE_SEARCH_NODE.to_langgraph_node(key="code_search")
        )

        _graph.set_entry_point("lit_search")
        _graph.add_edge("lit_search", "code_search")
        _graph.add_edge("code_search", END)

        GLOBAL_STATE = GlobalState(
            node_states={
                "lit_search": NodeTemplateState(
                    inputs={
                        "query": "landslide casualties in Nepal last year",
                    }
                ),
                "code_search": NodeTemplateState(),
            }
        )


        WORKFLOW = _graph.compile(checkpointer=MEMORY)


        result_graph = await WORKFLOW.ainvoke(GLOBAL_STATE, config=CONFIG)
        return GLOBAL_STATE, result_graph
    return


@app.cell
def _(result_graph):
    result_graph
    return


@app.cell
def _(GLOBAL_STATE):
    GLOBAL_STATE.model_dump()
    return


@app.cell
async def _(GLOBAL_STATE, workflow):
    async for chunk in workflow.astream(GLOBAL_STATE):
        print(f"Step: {chunk}")
        print("-" * 50)
    return


if __name__ == "__main__":
    app.run()
