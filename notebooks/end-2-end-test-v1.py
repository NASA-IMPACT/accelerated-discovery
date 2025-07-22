import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    """
    # AKD Research Workflow Demo

    This notebook demonstrates a complete research workflow using the AKD framework.
    It combines literature search, conditional code search, and content processing
    in a LangGraph workflow.
    """

    import marimo as mo

    mo.md("""
    # AKD Research Workflow Demo

    This notebook demonstrates a complete research workflow using the AKD framework.
    It combines literature search, conditional code search, and content processing 
    in a LangGraph workflow.

    ## Features:
    - Literature search using SearxNG
    - Conditional code search based on content analysis
    - Custom NodeTemplate implementations
    - LangGraph workflow orchestration
    - Interactive testing interface
    """)
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Imports""")
    return


@app.cell
def _():
    """Setup and imports"""

    import asyncio
    import logging
    import re
    from typing import Any, Dict, List, Optional
    return (List,)


@app.cell
def _():
    """Core imports and dependencies"""

    from loguru import logger
    from pydantic import BaseModel, Field

    # AKD Core Components
    from akd._base import InputSchema, OutputSchema
    from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
    from akd.nodes.templates import AbstractNodeTemplate
    from akd.nodes.states import GlobalState, NodeState
    from akd.tools.utils import tool_wrapper
    from akd.serializers import AKDSerializer

    # Agents and Tools
    from akd.agents.query import (
        QueryAgent,
        QueryAgentInputSchema,
        QueryAgentOutputSchema,
    )
    from akd.tools.search import (
        SearxNGSearchTool,
        SearxNGSearchToolInputSchema,
        SearxNGSearchToolConfig,
    )
    from akd.tools.code_search import (
        LocalRepoCodeSearchTool,
        LocalRepoCodeSearchToolInputSchema,
    )
    from akd.agents.extraction import ExtractionInputSchema
    from akd.structures import SearchResultItem

    # LangGraph Components
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    return (
        AKDSerializer,
        AbstractNodeTemplate,
        BaseAgentConfig,
        BaseModel,
        END,
        Field,
        GlobalState,
        InMemorySaver,
        InputSchema,
        InstructorBaseAgent,
        LocalRepoCodeSearchTool,
        LocalRepoCodeSearchToolInputSchema,
        NodeState,
        OutputSchema,
        QueryAgent,
        QueryAgentInputSchema,
        START,
        SearxNGSearchTool,
        SearxNGSearchToolConfig,
        SearxNGSearchToolInputSchema,
        StateGraph,
        logger,
        tool_wrapper,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data Models

    TODO

    - [ ] Build content analysis agent for conditional routing
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Custom anlaysis""")
    return


@app.cell
def _(BaseModel, Field, List):
    class ResearchSummary(BaseModel):
        """Final research summary combining all findings."""

        query: str = Field(..., description="Original research query")
        literature_findings: List[str] = Field(
            default_factory=list, description="Key literature findings"
        )
        code_findings: List[str] = Field(
            default_factory=list, description="Key code findings"
        )
        summary: str = Field(..., description="Overall research summary")
        sources: List[str] = Field(default_factory=list, description="Source URLs")
    return (ResearchSummary,)


@app.cell
def _(mo):
    mo.md(r"""## Define Guardrails""")
    return


@app.cell
def _(List, tool_wrapper):
    @tool_wrapper
    def validate_queries(queries: List[str]) -> bool:
        """Validate that we have minimum number of queries."""
        return len(queries) >= 1


    @tool_wrapper
    def validate_search_results(results: List) -> bool:
        """Validate that we have minimum number of search results."""
        return len(results) >= 1


    @tool_wrapper
    def validate_content_length(final_report: str) -> bool:
        """Validate that content has minimum length."""
        return len(final_report) >= 50
    return validate_content_length, validate_queries, validate_search_results


@app.cell
def _(mo):
    mo.md(r"""## Create agents/tools""")
    return


@app.cell
def _(BaseAgentConfig, QueryAgent):
    _cfg = BaseAgentConfig(
        model_name="gpt-4.1-nano",
        # system_prompt="Help formulate queries for general literature search. Make sure it respects the original intent. Be very scientific.",
    )
    QUERY_AGENT = QueryAgent(_cfg, debug=True)
    return (QUERY_AGENT,)


@app.cell
async def _(QUERY_AGENT, QueryAgentInputSchema):
    QUERY_AGENT.reset_memory()
    (
        await QUERY_AGENT.arun(
            QueryAgentInputSchema(
                query="PLEAESE DONT SEARCH FOR CODE", num_queries=10
            )
        )
    ).model_dump()
    return


@app.cell
def _(LocalRepoCodeSearchTool):
    CODE_SEARCH_TOOL = LocalRepoCodeSearchTool(debug=True)
    return (CODE_SEARCH_TOOL,)


@app.cell
async def _(CODE_SEARCH_TOOL, LocalRepoCodeSearchToolInputSchema):
    (
        await CODE_SEARCH_TOOL.arun(
            LocalRepoCodeSearchToolInputSchema(queries=["landslide"])
        )
    ).model_dump()
    return


@app.cell
def _(SearxNGSearchTool, SearxNGSearchToolConfig):
    _cfg = SearxNGSearchToolConfig(
        engines=["arxiv", "google_scholar"], max_results=50
    )
    SEARCH_TOOL = SearxNGSearchTool(_cfg, debug=False)
    return (SEARCH_TOOL,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Needs code search agent

    Needs code search?
    """
    )
    return


@app.cell
def _(Field, InputSchema, InstructorBaseAgent, OutputSchema):
    class NeedsCodeSearchInputSchema(InputSchema):
        """Input schema for determining whether code repo search is needed"""

        content: str = Field(
            ...,
            description="Input research content from literature search which affects the decision.",
        )
        original_query: str | None = Field(
            None,
            description="Original query that led to this content from literature search",
        )


    class NeedsCodeSearchOutputSchema(OutputSchema):
        """Output schema for dtermining whether code repo searhc is needed"""

        needs_code_search: bool = Field(
            ..., description="Boolean output for requiring code search or not"
        )
        confidence: float = Field(
            ..., description="Confidence value assigned for the decision"
        )
        reasoning: list[str] = Field(
            ..., description="Concise reasoning traces for the decision"
        )


    class NeedsCodeSearchAgent(InstructorBaseAgent):
        input_schema = NeedsCodeSearchInputSchema
        output_schema = NeedsCodeSearchOutputSchema
    return (
        NeedsCodeSearchAgent,
        NeedsCodeSearchInputSchema,
        NeedsCodeSearchOutputSchema,
    )


@app.cell
def _(BaseAgentConfig, NeedsCodeSearchAgent):
    NEEDS_CODE_SEARCH_AGENT = NeedsCodeSearchAgent(
        BaseAgentConfig(
            model_name="gpt-4o-mini",
            system_prompt="""
            It's applicable to do code search when the input content (which is the result from literature search) entails for a code search that potentially can add more evidence and values to the literature search. If it does, there needs to be a code search.
            If the content has self-sufficient evidence and need any code assiciated with it, it does not need code search.
            Primary purpose is to enhance literature review process, in assisting the user/scientist to help discover related tools and code base for the research topic.
        """,
        )
    )
    return (NEEDS_CODE_SEARCH_AGENT,)


@app.cell
async def _(NEEDS_CODE_SEARCH_AGENT, NeedsCodeSearchInputSchema):
    await NEEDS_CODE_SEARCH_AGENT.arun(NeedsCodeSearchInputSchema(content="LOL"))
    return


@app.cell
def _(mo):
    mo.md(r"""## Node Templates""")
    return


@app.cell
def _(
    AbstractNodeTemplate,
    GlobalState,
    NodeState,
    QUERY_AGENT,
    QueryAgentInputSchema,
    logger,
    validate_queries,
):
    class QueryGenerationNode(AbstractNodeTemplate):
        """Node that generates search queries from user input using QueryAgent."""

        def __init__(self, **kwargs):
            super().__init__(
                node_id="query_generation",
                input_guardrails=[],
                output_guardrails=[(validate_queries, {"queries": "queries"})],
                **kwargs,
            )
            self.query_agent = QUERY_AGENT
            self.query_agent.reset_memory()

        async def _execute(
            self, node_state: NodeState, global_state: GlobalState
        ) -> NodeState:
            """Generate search queries from user input."""
            user_query = node_state.inputs.get("user_query", "")
            num_queries = node_state.inputs.get("num_queries", 3)

            if self.debug:
                logger.info(f"Generating {num_queries} queries for: {user_query}")

            # Generate queries using QueryAgent
            query_input = QueryAgentInputSchema(
                query=user_query, num_queries=num_queries
            )

            query_result = await self.query_agent.arun(query_input)

            # Store results in node state
            node_state.outputs["queries"] = query_result.queries
            node_state.outputs["category"] = query_result.category
            node_state.outputs["original_query"] = user_query

            if self.debug:
                logger.info(f"Generated queries: {query_result.queries}")

            return node_state
    return (QueryGenerationNode,)


@app.cell
def _(
    AbstractNodeTemplate,
    GlobalState,
    NodeState,
    SEARCH_TOOL,
    SearxNGSearchToolInputSchema,
    logger,
    validate_search_results,
):
    class LiteratureSearchNode(AbstractNodeTemplate):
        """Node that searches literature using SearxNGSearchTool."""

        def __init__(self, **kwargs):
            super().__init__(
                node_id="literature_search",
                input_guardrails=[],
                output_guardrails=[
                    (validate_search_results, {"results": "literature_results"})
                ],
                **kwargs,
            )
            self.search_tool = SEARCH_TOOL

        async def _execute(
            self, node_state: NodeState, global_state: GlobalState
        ) -> NodeState:
            """Search literature using generated queries."""
            # Get queries from query generation node
            query_node = global_state.node_states.get(
                "query_generation", NodeState()
            )
            queries = query_node.outputs.get("queries", [])
            category = query_node.outputs.get("category", "science")

            if not queries:
                queries = [node_state.inputs.get("user_query", "")]

            if self.debug:
                logger.info(f"Searching literature with queries: {queries}")

            # Search literature
            search_input = SearxNGSearchToolInputSchema(
                queries=queries, category=category, max_results=50
            )

            search_result = await self.search_tool.arun(search_input)
            logger.debug(f"Search {len(search_result.results)} items")

            # Store results in node state
            node_state.outputs["literature_results"] = search_result.results
            node_state.outputs["num_results"] = len(search_result.results)

            if self.debug:
                logger.info(
                    f"Found {len(search_result.results)} literature results"
                )

            return node_state
    return (LiteratureSearchNode,)


@app.cell
def _(
    AbstractNodeTemplate,
    CODE_SEARCH_TOOL,
    GlobalState,
    LocalRepoCodeSearchToolInputSchema,
    NodeState,
    logger,
    validate_search_results,
):
    class CodeSearchNode(AbstractNodeTemplate):
        """Node that searches code repositories using LocalRepoCodeSearchTool."""

        def __init__(self, **kwargs):
            super().__init__(
                node_id="code_search",
                input_guardrails=[],
                output_guardrails=[
                    (validate_search_results, {"results": "code_results"})
                ],
                **kwargs,
            )
            self.code_search_tool = CODE_SEARCH_TOOL

        async def _execute(
            self, node_state: NodeState, global_state: GlobalState
        ) -> NodeState:
            """Search code repositories using generated queries."""
            # Get queries from query generation node
            query_node = global_state.node_states.get(
                "query_generation", NodeState()
            )
            queries = query_node.outputs.get("queries", [])

            if not queries:
                queries = [node_state.inputs.get("user_query", "")]

            if self.debug:
                logger.info(f"Searching code with queries: {queries}")

            # Search code repositories
            search_input = LocalRepoCodeSearchToolInputSchema(
                queries=queries, max_results=10
            )

            search_result = await self.code_search_tool.arun(search_input)

            # Store results in node state
            node_state.outputs["code_results"] = search_result.results
            node_state.outputs["num_results"] = len(search_result.results)

            if self.debug:
                logger.info(f"Found {len(search_result.results)} code results")

            return node_state
    return (CodeSearchNode,)


@app.cell
def _(
    AbstractNodeTemplate,
    GlobalState,
    NEEDS_CODE_SEARCH_AGENT,
    NeedsCodeSearchInputSchema,
    NeedsCodeSearchOutputSchema,
    NodeState,
    logger,
):
    class ContentProcessingNode(AbstractNodeTemplate):
        """Node that processes literature content and determines if code search is needed."""

        def __init__(self, **kwargs):
            super().__init__(
                node_id="content_processing",
                input_guardrails=[],
                output_guardrails=[],
                **kwargs,
            )
            self.needs_code_search_agent = NEEDS_CODE_SEARCH_AGENT
            self.needs_code_search_agent.reset_memory()

        async def _analyze_content_for_code_relevance(
            self, content: str
        ) -> NeedsCodeSearchOutputSchema:
            return await self.needs_code_search_agent.arun(
                NeedsCodeSearchInputSchema(content=content)
            )

        async def _execute(
            self, node_state: NodeState, global_state: GlobalState
        ) -> NodeState:
            """Process literature content and analyze for code relevance."""
            # Get literature results
            lit_node = global_state.node_states.get(
                "literature_search", NodeState()
            )
            lit_results = lit_node.outputs.get("literature_results", [])

            if self.debug:
                logger.info(f"Processing {len(lit_results)} literature results")

            # Extract key findings
            literature_findings = []

            # Analyze content for code relevance
            combined_content = " ".join(
                [f"{r.title} {r.content}" for r in lit_results]
            )

            analysis = await self._analyze_content_for_code_relevance(
                combined_content
            )
            logger.info(f"Content analysis for code search :: {analysis}")

            # Store results in node state
            node_state.outputs["literature_findings"] = literature_findings
            node_state.outputs["content_analysis"] = analysis
            node_state.outputs["processed_content"] = combined_content[
                :500
            ]  # Truncate for storage

            return node_state
    return (ContentProcessingNode,)


@app.cell
def _(
    AbstractNodeTemplate,
    GlobalState,
    NodeState,
    ResearchSummary,
    logger,
    validate_content_length,
):
    class ReportGenerationNode(AbstractNodeTemplate):
        """Node that generates final research summary combining all findings."""

        def __init__(self, **kwargs):
            super().__init__(
                node_id="report_generation",
                input_guardrails=[],
                output_guardrails=[
                    (validate_content_length, {"final_report": "final_report"})
                ],
                **kwargs,
            )

        async def _execute(
            self, node_state: NodeState, global_state: GlobalState
        ) -> NodeState:
            """Generate final research summary."""
            # Get data from other nodes
            query_node = global_state.node_states.get(
                "query_generation", NodeState()
            )
            lit_node = global_state.node_states.get(
                "literature_search", NodeState()
            )
            content_node = global_state.node_states.get(
                "content_processing", NodeState()
            )
            code_node = global_state.node_states.get("code_search", NodeState())

            original_query = query_node.outputs.get("original_query", "")
            literature_findings = content_node.outputs.get(
                "literature_findings", []
            )
            lit_results = lit_node.outputs.get("literature_results", [])
            code_results = code_node.outputs.get("code_results", [])

            if self.debug:
                logger.info("Generating final research summary")

            # Generate code findings if available
            code_findings = []
            if code_results:
                code_findings = [
                    f"{r.title}: {r.content[:100]}..." if r.content else r.title
                    for r in code_results[:3]
                ]

            # Generate summary
            summary_parts = []
            summary_parts.append(f"Research Query: {original_query}")
            summary_parts.append(
                f"Literature Sources: {len(lit_results)} papers/articles found"
            )

            if literature_findings:
                summary_parts.append("Key Literature Findings:")
                for finding in literature_findings[:10]:
                    summary_parts.append(f"- {finding}")

            if code_findings:
                summary_parts.append(
                    f"Code Sources: {len(code_results)} repositories found"
                )
                summary_parts.append("Key Code Findings:")
                for finding in code_findings:
                    summary_parts.append(f"- {finding}")

            summary = "\n".join(summary_parts)

            # Collect source URLs
            sources = []
            for result in lit_results + code_results:
                if result.url:
                    sources.append(str(result.url))

            # Create final summary
            research_summary = ResearchSummary(
                query=original_query,
                literature_findings=literature_findings,
                code_findings=code_findings,
                summary=summary,
                sources=sources,
            )

            # Store results in node state
            node_state.outputs["research_summary"] = research_summary
            node_state.outputs["final_report"] = summary

            if self.debug:
                logger.info("Research summary generated successfully")

            return node_state
    return (ReportGenerationNode,)


@app.cell
def _(GlobalState, NodeState):
    def should_search_code(state: GlobalState) -> str:
        """Determine if code search should be performed based on content analysis."""
        # Handle both dict and GlobalState formats
        if isinstance(state, dict):
            node_states = state.get("node_states", {})
        else:
            node_states = state.node_states

        content_node = node_states.get("content_processing", NodeState())
        analysis = content_node.outputs.get("content_analysis")

        if analysis and analysis.needs_code_search:
            return "code_search"
        else:
            return "report_generation"


    def code_search_complete(state: GlobalState) -> str:
        """Route to report generation after code search."""
        return "report_generation"
    return (should_search_code,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Define Workflow

    Use the nodes and create workflow

    - Planner will do it IRL
    """
    )
    return


@app.cell
def _(
    AKDSerializer,
    CodeSearchNode,
    ContentProcessingNode,
    END,
    GlobalState,
    InMemorySaver,
    LiteratureSearchNode,
    NodeState,
    QueryGenerationNode,
    ReportGenerationNode,
    START,
    StateGraph,
    logger,
    should_search_code,
):
    class AKDWorkflow:
        """Main workflow orchestrator for the AKD research pipeline."""

        def __init__(self, debug: bool = False):
            self.debug = debug
            self.memory = InMemorySaver(serde=AKDSerializer())
            self.workflow = self._create_workflow()

        def _create_workflow(self) -> StateGraph:
            """Create the LangGraph workflow."""
            # Create workflow graph
            workflow = StateGraph(GlobalState)

            # Create node instances
            query_node = QueryGenerationNode(debug=self.debug, mutation=False)
            literature_node = LiteratureSearchNode(
                debug=self.debug, mutation=False
            )
            content_node = ContentProcessingNode(debug=self.debug, mutation=False)
            code_node = CodeSearchNode(debug=self.debug, mutation=False)
            report_node = ReportGenerationNode(debug=self.debug, mutation=False)

            # Add nodes to workflow
            workflow.add_node("query_generation", query_node.to_langgraph_node())
            workflow.add_node(
                "literature_search", literature_node.to_langgraph_node()
            )
            workflow.add_node(
                "content_processing", content_node.to_langgraph_node()
            )
            workflow.add_node("code_search", code_node.to_langgraph_node())
            workflow.add_node("report_generation", report_node.to_langgraph_node())

            # Define workflow edges
            workflow.add_edge(START, "query_generation")
            workflow.add_edge("query_generation", "literature_search")
            workflow.add_edge("literature_search", "content_processing")

            # Conditional routing based on content analysis
            workflow.add_conditional_edges(
                "content_processing",
                should_search_code,
                {
                    "code_search": "code_search",
                    "report_generation": "report_generation",
                },
            )

            workflow.add_edge("code_search", "report_generation")
            workflow.add_edge("report_generation", END)

            return workflow.compile(checkpointer=self.memory)

        async def arun(self, query: str, num_queries: int = 3) -> GlobalState:
            """Run the complete research workflow."""
            # Create initial state
            initial_state = GlobalState(
                node_states={
                    "query_generation": NodeState(
                        inputs={
                            "user_query": query,
                            "query": query,
                            "num_queries": num_queries,
                        }
                    )
                }
            )

            # Run workflow
            config = {"configurable": {"thread_id": "research_session"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)

            return final_state

        async def stream_run(self, query: str, num_queries: int = 3):
            """Run workflow with streaming output."""
            # Create initial state
            initial_state = GlobalState(
                node_states={
                    "query_generation": NodeState(
                        inputs={
                            "user_query": query,
                            "query": query,
                            "num_queries": num_queries,
                        }
                    )
                }
            )

            # Stream workflow execution
            config = {"configurable": {"thread_id": "research_session"}}
            async for chunk in self.workflow.astream(initial_state, config=config):
                node_name = list(chunk.keys())[0] if chunk else "unknown"
                logger.info(f"Completed: {node_name}")
                yield chunk
    return (AKDWorkflow,)


@app.cell
def _(QUERY_AGENT):
    QUERY_AGENT.memory
    return


@app.cell
async def _(AKDWorkflow):
    akd_workflow = AKDWorkflow(debug=False)
    workflow_output = await akd_workflow.arun(
        # query="population growth in urban areas risk assessment for nepal throughout the decades.",
        query="PLEASE DONT SEARCH FOR CODE",
        num_queries=5,
    )
    return (workflow_output,)


@app.cell
def _(workflow_output):
    workflow_output
    return


@app.cell
def _(mo):
    mo.md(r"""# Interactive""")
    return


@app.cell
def _():
    # """Interactive query input"""

    # # Create interactive input form
    # query_input = mo.ui.text_area(
    #     placeholder="Enter your research query here...",
    #     label="Research Query",
    #     value="landslide risk assessment in nepal throught the years",
    # )

    # num_queries_input = mo.ui.number(
    #     start=1, stop=10, value=3, label="Number of queries to generate"
    # )

    # debug_checkbox = mo.ui.checkbox(value=True, label="Enable debug output")

    # mo.md(f"""
    # ## Interactive Research Query

    # {query_input}

    # {num_queries_input}

    # {debug_checkbox}
    # """)
    return


@app.cell
def _():
    # """Run workflow buttons"""

    # run_button = mo.ui.run_button(label="🔍 Run Research Workflow")
    # stream_button = mo.ui.run_button(label="📡 Stream Workflow")

    # mo.md(f"""
    # ## Execute Research

    # {run_button} {stream_button}
    # """)
    return


@app.cell
def _():
    # """Execute workflow when run button is clicked"""

    # result = None

    # if run_button.value:
    #     if query_input.value:
    #         try:
    #             # Create workflow instance
    #             workflow = AKDWorkflow(debug=debug_checkbox.value)

    #             # Run the workflow
    #             result = await workflow.arun(
    #                 user_query=query_input.value,
    #                 num_queries=num_queries_input.value,
    #             )

    #             status = "✅ Research completed successfully!"

    #         except Exception as e:
    #             status = f"❌ Error: {str(e)}"
    #             result = None
    #     else:
    #         status = "⚠️ Please enter a research query"
    #         result = None
    # else:
    #     status = "Ready to run research workflow"

    # status
    return


@app.cell
def _():
    # result.model_dump()
    return


@app.cell
def _():
    # """Display results"""

    # if result:
    #     mo.md(f"""
    #     ## Research Results

    #     **Query:** {result.query}

    #     **Summary:**
    #     ```
    #     {result.summary}
    #     ```

    #     **Literature Findings:** {len(result.literature_findings)} items

    #     **Code Findings:** {len(result.code_findings)} items

    #     **Sources:** {len(result.sources)} URLs
    #     """)
    # else:
    #     mo.md("No results yet. Run the workflow above to see results.")
    return


@app.cell
def _():
    # """Streaming workflow execution"""

    # stream_output = []

    # if stream_button.value:
    #     if query_input.value:
    #         try:
    #             # Create workflow instance
    #             stream_workflow = AKDWorkflow(debug=debug_checkbox.value)

    #             # Run streaming workflow
    #             async def run_stream():
    #                 steps = []
    #                 async for chunk in stream_workflow.stream_run(
    #                     user_query=query_input.value,
    #                     num_queries=num_queries_input.value,
    #                 ):
    #                     node_name = list(chunk.keys())[0] if chunk else "unknown"
    #                     steps.append(f"Step completed: {node_name}")
    #                 return steps

    #             stream_output = asyncio.run(run_stream())
    #             stream_status = "✅ Streaming workflow completed!"

    #         except Exception as e:
    #             stream_status = f"❌ Streaming error: {str(e)}"
    #             stream_output = []
    #     else:
    #         stream_status = "⚠️ Please enter a research query"
    #         stream_output = []
    # else:
    #     stream_status = "Ready to stream workflow"

    # stream_status
    return


@app.cell
def _():
    # """Display streaming results"""

    # if stream_output:
    #     steps_text = "\n".join(stream_output)
    #     mo.md(f"""
    #     ## Streaming Progress

    #     ```
    #     {steps_text}
    #     ```
    #     """)
    # else:
    #     mo.md(
    #         "No streaming output yet. Click 'Stream Workflow' to see live progress."
    #     )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    """Workflow visualization and documentation"""

    mo.md("""
    ## Workflow Architecture

    The AKD research workflow follows this structure:

    ```
    START
      ↓
    Query Generation (QueryAgent)
      ↓
    Literature Search (SearxNG)
      ↓
    Content Processing & Analysis
      ↓
    Decision Point: Code Search Needed?
      ├─ Yes → Code Search (LocalRepo) → Report Generation
      └─ No  → Report Generation
      ↓
    END
    ```

    ### Node Descriptions:

    1. **Query Generation**: Uses QueryAgent to generate multiple search queries from user input
    2. **Literature Search**: Searches academic/web literature using SearxNG
    3. **Content Processing**: Analyzes literature content and determines if code search is needed
    4. **Code Search** (Conditional): Searches local repositories for relevant code
    5. **Report Generation**: Combines all findings into a comprehensive research summary

    ### Key Features:

    - **Conditional Routing**: Code search only runs if content analysis indicates it's needed
    - **Guardrails**: Each node has input/output validation
    - **State Management**: Global state maintains data flow between nodes
    - **Streaming Support**: Real-time progress monitoring
    - **Error Handling**: Robust error handling and recovery
    """)
    return


@app.cell
def _(mo, result):
    """Advanced result analysis"""

    if result:
        # Calculate some metrics
        total_sources = len(result.sources)
        lit_findings_count = len(result.literature_findings)
        code_findings_count = len(result.code_findings)

        # Determine search path taken
        search_path = (
            "Literature + Code Search"
            if code_findings_count > 0
            else "Literature Only"
        )

        mo.md(f"""
        ## Advanced Result Analysis

        ### Search Metrics
        - **Search Path**: {search_path}
        - **Total Sources**: {total_sources}
        - **Literature Findings**: {lit_findings_count}
        - **Code Findings**: {code_findings_count}

        ### Source Breakdown
        """)

        if result.sources:
            sources_list = "\n".join(
                [f"- {source}" for source in result.sources[:5]]
            )
            mo.md(f"""
            **Top 5 Sources:**
            {sources_list}

            {f"... and {total_sources - 5} more sources" if total_sources > 5 else ""}
            """)
    else:
        mo.md("Run a research workflow to see advanced analysis.")
    return


@app.cell
def _(mo):
    """Configuration and settings"""

    mo.md("""
    ## Configuration & Troubleshooting

    ### Required Dependencies

    Make sure you have these packages installed:

    ```bash
    pip install marimo
    pip install akd  # Your AKD framework
    pip install loguru
    pip install pydantic
    pip install langgraph
    ```

    ### Environment Setup

    1. Ensure SearxNG search service is running and accessible
    2. Configure local repository paths for code search
    3. Set up any required API keys in environment variables

    ### Common Issues

    - **Import Errors**: Check that all AKD components are properly installed
    - **Search Failures**: Verify SearxNG service connectivity
    - **Memory Issues**: Adjust batch sizes for large result sets
    - **Timeout Errors**: Increase timeout settings for slow searches

    ### Performance Tips

    - Use fewer queries for faster execution
    - Enable debug mode for detailed logging
    - Consider caching for repeated queries
    - Monitor memory usage with large datasets
    """)
    return


@app.cell
def _(mo):
    """Export and save functionality"""

    export_button = mo.ui.button(label="📄 Export Results")

    mo.md(f"""
    ## Export & Save

    {export_button}

    Export your research results for external use or further analysis.
    """)
    return (export_button,)


@app.cell
def _(export_button, mo, result):
    """Handle export functionality"""

    if export_button.value and result:
        # Create exportable format
        export_data = {
            "query": result.query,
            "summary": result.summary,
            "literature_findings": result.literature_findings,
            "code_findings": result.code_findings,
            "sources": result.sources,
            "timestamp": "2024-01-01",  # You could use datetime.now()
        }

        # Convert to JSON string for display
        import json

        export_json = json.dumps(export_data, indent=2)

        mo.md(f"""
        ### Exported Research Data

        ```json
        {export_json}
        ```

        Copy the JSON above to save your research results.
        """)
    elif export_button.value:
        mo.md("No results to export. Run a research workflow first.")
    else:
        mo.md("Click 'Export Results' to generate exportable data.")
    return


@app.cell
def _(mo):
    """Footer and additional resources"""

    mo.md("""
    ---

    ## Additional Resources

    - **AKD Framework Documentation**: [Link to docs]
    - **LangGraph Documentation**: [Link to LangGraph]
    - **Marimo Documentation**: [Link to Marimo]

    ### Next Steps

    1. Experiment with different query types
    2. Customize node implementations for your use case
    3. Add new search tools or agents
    4. Implement result caching and persistence
    5. Create custom visualizations for results

    ---

    *Built with AKD Framework, LangGraph, and Marimo*
    """)
    return


if __name__ == "__main__":
    app.run()
