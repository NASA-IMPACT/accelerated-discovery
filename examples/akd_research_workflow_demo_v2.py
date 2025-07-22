#!/usr/bin/env python3
"""
AKD Research Workflow Demo

This script demonstrates a complete research workflow using the AKD framework.
It combines literature search, conditional code search, and content processing
in a LangGraph workflow.

Features:
- Literature search using SearxNG
- Conditional code search based on content analysis
- Custom NodeTemplate implementations
- LangGraph workflow orchestration
- Interactive testing interface

Converted from marimo notebook to standalone Python script.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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


# Data Models
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


# Guardrails
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


# Custom Agents
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


# Node Templates
class QueryGenerationNode(AbstractNodeTemplate):
    """Node that generates search queries from user input using QueryAgent."""

    def __init__(self, query_agent: QueryAgent, **kwargs):
        super().__init__(
            node_id="query_generation",
            input_guardrails=[],
            output_guardrails=[(validate_queries, {"queries": "queries"})],
            **kwargs,
        )
        self.query_agent = query_agent
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


class LiteratureSearchNode(AbstractNodeTemplate):
    """Node that searches literature using SearxNGSearchTool."""

    def __init__(self, search_tool: SearxNGSearchTool, **kwargs):
        super().__init__(
            node_id="literature_search",
            input_guardrails=[],
            output_guardrails=[
                (validate_search_results, {"results": "literature_results"})
            ],
            **kwargs,
        )
        self.search_tool = search_tool

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


class CodeSearchNode(AbstractNodeTemplate):
    """Node that searches code repositories using LocalRepoCodeSearchTool."""

    def __init__(self, code_search_tool: LocalRepoCodeSearchTool, **kwargs):
        super().__init__(
            node_id="code_search",
            input_guardrails=[],
            output_guardrails=[
                (validate_search_results, {"results": "code_results"})
            ],
            **kwargs,
        )
        self.code_search_tool = code_search_tool

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


class ContentProcessingNode(AbstractNodeTemplate):
    """Node that processes literature content and determines if code search is needed."""

    def __init__(self, needs_code_search_agent: NeedsCodeSearchAgent, **kwargs):
        super().__init__(
            node_id="content_processing",
            input_guardrails=[],
            output_guardrails=[],
            **kwargs,
        )
        self.needs_code_search_agent = needs_code_search_agent
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


# Workflow routing functions
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


# Main Workflow Class
class AKDWorkflow:
    """Main workflow orchestrator for the AKD research pipeline."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.memory = InMemorySaver(serde=AKDSerializer())
        
        # Initialize agents and tools
        self.query_agent = QueryAgent(BaseAgentConfig(
            model_name="gpt-4.1-nano",
        ), debug=debug)
        
        self.search_tool = SearxNGSearchTool(
            SearxNGSearchToolConfig(
                engines=["arxiv", "google_scholar"], 
                max_results=50
            ), 
            debug=debug
        )
        
        self.code_search_tool = LocalRepoCodeSearchTool(debug=debug)
        
        self.needs_code_search_agent = NeedsCodeSearchAgent(
            BaseAgentConfig(
                model_name="gpt-4o-mini",
                system_prompt="""
                It's applicable to do code search when the input content (which is the result from literature search) entails for a code search that potentially can add more evidence and values to the literature search. If it does, there needs to be a code search.
                If the content has self-sufficient evidence and need any code assiciated with it, it does not need code search.
                Primary purpose is to enhance literature review process, in assisting the user/scientist to help discover related tools and code base for the research topic.
                """,
            )
        )
        
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create workflow graph
        workflow = StateGraph(GlobalState)

        # Create node instances
        query_node = QueryGenerationNode(
            query_agent=self.query_agent,
            debug=self.debug, 
            mutation=False
        )
        literature_node = LiteratureSearchNode(
            search_tool=self.search_tool,
            debug=self.debug, 
            mutation=False
        )
        content_node = ContentProcessingNode(
            needs_code_search_agent=self.needs_code_search_agent,
            debug=self.debug, 
            mutation=False
        )
        code_node = CodeSearchNode(
            code_search_tool=self.code_search_tool,
            debug=self.debug, 
            mutation=False
        )
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


async def main():
    """Main function to demonstrate the research workflow."""
    print("=" * 80)
    print("AKD Research Workflow Demo")
    print("=" * 80)
    
    # Create workflow instance
    workflow = AKDWorkflow(debug=True)
    
    # Example queries
    test_queries = [
        "population growth in urban areas risk assessment for nepal throughout the decades",
        "landslide risk assessment in nepal throughout the years",
        "climate change impact on himalayan glaciers",
    ]
    
    # Run a test query
    query = test_queries[1]  # Using the landslide query
    print(f"\nRunning research workflow for: '{query}'")
    print("-" * 80)
    
    try:
        # Run the workflow
        result = await workflow.arun(query=query, num_queries=5)
        
        # Extract the research summary from the final state
        report_node = result["node_states"].get("report_generation", NodeState())
        research_summary = report_node.outputs.get("research_summary")
        
        if research_summary:
            print("\n" + "=" * 80)
            print("RESEARCH SUMMARY")
            print("=" * 80)
            print(research_summary.summary)
            print("\n" + "-" * 80)
            print(f"Total Sources: {len(research_summary.sources)}")
            print(f"Literature Findings: {len(research_summary.literature_findings)}")
            print(f"Code Findings: {len(research_summary.code_findings)}")
            print("-" * 80)
            
            # Show some sources
            if research_summary.sources:
                print("\nTop 5 Sources:")
                for i, source in enumerate(research_summary.sources[:5], 1):
                    print(f"{i}. {source}")
        else:
            print("No research summary generated")
            
    except Exception as e:
        print(f"Error running workflow: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("Workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())