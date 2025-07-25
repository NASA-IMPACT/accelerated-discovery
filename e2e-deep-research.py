#!/usr/bin/env python3
"""
Deep Research Workflow Demo

- DeepResearchAgent for Agentic Research
- Combined search using SearxNG
- Agentic quality validation with ReportQualityValidatorAgent
- Guardrails integration
- LangGraph workflow

Configuration:
- Set GUARDRAILS_ENABLED = True to enable Granite Guardian validation
"""

import asyncio
from typing import List

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

# AKD Core Components
from akd.agents.deep_research import DeepResearchAgent, DeepResearchAgentConfig
from akd.agents.guardrails import add_guardrails
from akd.agents.report_quality_validator import (
    ReportQualityValidatorAgent,
    ReportQualityValidatorAgentConfig,
)
from akd.agents.schemas import (
    DeepResearchInputSchema,
    ReportQualityValidatorInputSchema,
)
from akd.configs.guardrails_config import GrandrailsAgentConfig
from akd.nodes.states import GlobalState, NodeState
from akd.nodes.templates import AbstractNodeTemplate
from akd.serializers import AKDSerializer
from akd.tools.granite_guardian_tool import RiskDefinition
from akd.tools.search.agentic_search import DeepLitSearchTool, DeepLitSearchToolConfig
from akd.tools.search.searxng_search import SearxNGSearchTool, SearxNGSearchToolConfig
from akd.tools.utils import tool_wrapper

# Load environment variables
load_dotenv()


# Global Configuration
GUARDRAILS_ENABLED = False  # guardrails config for the entire workflow


# Data Models
class DeepResearchSummary(BaseModel):
    """Final deep research summary."""

    original_query: str = Field(..., description="Original research query")
    research_report: str = Field(..., description="Comprehensive research report")
    key_findings: List[str] = Field(
        default_factory=list, description="Key research findings"
    )
    sources_consulted: List[str] = Field(
        default_factory=list, description="Sources consulted"
    )
    evidence_quality_score: float = Field(
        ..., description="Evidence quality score (0-1)"
    )
    iterations_performed: int = Field(..., description="Number of research iterations")


# Enhanced Guardrails
@tool_wrapper
def validate_research_quality(
    report: str, findings: List[str], quality_score: float
) -> bool:
    """Comprehensive validation of research quality with guardrails."""
    return len(report) >= 200 and len(findings) >= 1 and quality_score >= 0.5


@tool_wrapper
def validate_source_credibility(sources: List[str]) -> bool:
    """Validate that sources meet credibility standards."""
    return len(sources) >= 3  # Minimum source count for credibility


# Node Templates
@add_guardrails(
    config=GrandrailsAgentConfig(
        enabled=GUARDRAILS_ENABLED,  # toggle for guardrails
        input_risk_types=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
        output_risk_types=[RiskDefinition.RELEVANCE, RiskDefinition.GROUNDEDNESS],
        fail_on_risk=False,  # Just log warnings, don't fail workflow
    )
)
class ResearchNode(AbstractNodeTemplate):
    """Node that performs research using ResearchAgent and combined search tools."""

    def __init__(self, **kwargs):
        super().__init__(
            node_id="deep_research",
            input_guardrails=[],
            output_guardrails=[
                (
                    validate_research_quality,
                    {
                        "report": "research_report",
                        "findings": "key_findings",
                        "quality_score": "evidence_quality_score",
                    },
                ),
                (validate_source_credibility, {"sources": "sources_consulted"}),
            ],
            **kwargs,
        )

        # Initialize search tools
        # Primary search tool (SearxNG)
        searxng_config = SearxNGSearchToolConfig(
            engines=["crossref", "google_scholar", "semantic_scholar"]
        )
        searxng_tool = SearxNGSearchTool(searxng_config)

        # Initialize DeepLitSearchTool with combined search capabilities
        deep_search_config = DeepLitSearchToolConfig(
            max_research_iterations=5,
            quality_threshold=0.7,
            use_semantic_scholar=True,
            # Relevancy assessment configuration
            enable_per_link_assessment=True,
            min_relevancy_score=0.3,
            full_content_threshold=0.7,
            enable_full_content_scraping=True,
            debug=self.debug,
        )

        self.deep_search_tool = DeepLitSearchTool(
            config=deep_search_config,
            search_tool=searxng_tool,  # Primary tool
            debug=self.debug,
        )

        # Initialize DeepResearchAgent
        research_config = DeepResearchAgentConfig(
            model_name="gpt-4o",
            temperature=0.2,
            max_tokens=4000,
        )

        self.deep_research_agent = DeepResearchAgent(
            config=research_config,
            debug=self.debug,
        )

        # Initialize ReportQualityValidatorAgent
        validator_config = ReportQualityValidatorAgentConfig(
            model_name="gpt-4o",
            temperature=0.1,
            minimum_overall_score=0.8,
            query_alignment_threshold=0.8,
        )

        self.quality_validator = ReportQualityValidatorAgent(
            config=validator_config,
            debug=self.debug,
        )

    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,  # noqa: ARG002
    ) -> NodeState:
        """Execute research workflow."""
        user_query = node_state.inputs.get("user_query", "")
        max_results = node_state.inputs.get("max_results", 50)

        if self.debug:
            logger.info(f"Starting deep research for query: {user_query}")

        # Step 1: Use DeepLitSearchTool for comprehensive search
        from akd.tools.search.searxng_search import SearxNGSearchToolInputSchema

        search_input = SearxNGSearchToolInputSchema(
            queries=[user_query],
            max_results=max_results,
            category="science",
        )

        if self.debug:
            logger.info("Executing literature search...")
            logger.info(f"🔍 INITIAL QUERY BEING SENT TO DeepLitSearchTool: '{user_query}'")

        search_results = await self.deep_search_tool.arun(search_input)

        if self.debug:
            logger.info(f"Found {len(search_results.results)} search results")
            
            # Extract and log all search results with their relevancy scores
            logger.info("\n" + "═" * 80)
            logger.info("📊 SEARCH RESULTS WITH RELEVANCY SCORES:")
            logger.info("═" * 80)
            for i, result in enumerate(search_results.results, 1):
                relevancy_info = ""
                if hasattr(result, 'relevancy_score') and result.relevancy_score is not None:
                    score = result.relevancy_score
                    if score >= 0.7:
                        relevancy_info = f"🟢 {score:.3f}"
                    elif score >= 0.4:
                        relevancy_info = f"🟡 {score:.3f}"
                    else:
                        relevancy_info = f"🔴 {score:.3f}"
                    
                    # Add query alignment info
                    if hasattr(result, 'query_alignment_details') and result.query_alignment_details:
                        best_query = result.query_alignment_details.get('best_query')
                        if best_query == 'original':
                            relevancy_info += " 🎯"
                        elif best_query == 'reformulated':
                            relevancy_info += " 🔄"
                else:
                    relevancy_info = "❓ No score"
                
                title_preview = (result.title or "Untitled")[:100]
                if len(result.title or "") > 100:
                    title_preview += "..."
                    
                logger.info(f"  {i:2d}. {relevancy_info} | {title_preview}")
                
                # Log problematic results (landslide example)
                if result.title and "landslide" in result.title.lower() and hasattr(result, 'relevancy_score'):
                    logger.warning(f"⚠️  POTENTIAL RELEVANCY ISSUE: Landslide article scored {result.relevancy_score:.3f}")
                    logger.warning(f"     Title: {result.title}")
                    if hasattr(result, 'query_alignment_details'):
                        logger.warning(f"     Query Alignment: {result.query_alignment_details}")
            logger.info("═" * 80)

        # Step 2: Extract research report and findings from search results
        # The DeepLitSearchTool returns a research report as the first result
        research_report = ""
        key_findings = []
        sources = []
        quality_score = 0.5
        iterations = 1
        weak_rubrics = []
        strong_rubrics = []

        if search_results.results:
            # First result should be the research report
            first_result = search_results.results[0]
            if first_result.url == "research report":
                research_report = first_result.content
                if first_result.extra:
                    key_findings = first_result.extra.get("key_findings", [])
                    quality_score = first_result.extra.get("quality_score", 0.5)
                    iterations = first_result.extra.get("iterations", 1)
                    weak_rubrics = first_result.extra.get("weak_rubrics", [])
                    strong_rubrics = first_result.extra.get("strong_rubrics", [])

            # Collect source information with relevancy data, skipping the first "research report" entry
            sources = []
            source_details = []
            for result in search_results.results[
                1:
            ]:  # Skip first result (research report)
                if result.url and result.url != "research report":
                    sources.append(str(result.url))

                    # Create source detail with relevancy information
                    source_detail = {
                        "url": str(result.url),
                        "title": result.title or "Untitled",
                        "content_preview": (result.content or "")[:200] + "..."
                        if result.content
                        else "No preview available",
                    }

                    # Add relevancy information if available
                    if (
                        hasattr(result, "relevancy_score")
                        and result.relevancy_score is not None
                    ):
                        source_detail["relevancy_score"] = result.relevancy_score
                        source_detail["should_fetch_full_content"] = getattr(
                            result, "should_fetch_full_content", False
                        )
                        source_detail["query_alignment_details"] = getattr(
                            result, "query_alignment_details", None
                        )

                    source_details.append(source_detail)

        # Step 3: Use agentic quality validation to determine if DeepResearchAgent is needed
        should_use_deep_research = False
        validation_reasoning = "No research report to validate"

        if research_report:
            # Validate research report quality using agentic approach
            validation_input = ReportQualityValidatorInputSchema(
                original_query=user_query,
                research_report=research_report,
                weak_rubrics=weak_rubrics,
                strong_rubrics=strong_rubrics,
                evidence_quality_score=quality_score,
                sources_consulted=sources,
            )

            if self.debug:
                logger.info("Running agentic quality validation on research report...")

            validation_result = await self.quality_validator.arun(validation_input)
            should_use_deep_research = validation_result.trigger_deep_research
            validation_reasoning = validation_result.decision_reasoning

            if self.debug:
                logger.info(f"Quality validation result: {validation_reasoning}")
                logger.info(
                    f"Overall quality score: {validation_result.overall_quality_score:.2f}"
                )
                logger.info(f"Trigger deep research: {should_use_deep_research}")
        else:
            should_use_deep_research = True
            validation_reasoning = "No research report generated from initial search"

        if should_use_deep_research:
            if self.debug:
                logger.info(f"Using DeepResearchAgent: {validation_reasoning}")

            # Create research instructions from the search results
            research_content = "\n\n".join(
                [
                    f"Title: {result.title}\nContent: {result.content}"
                    for result in search_results.results[:10]
                    if result.content
                ]
            )

            research_instructions = f"""
            Conduct comprehensive research on: {user_query}
            
            Based on the following literature findings:
            {research_content}
            
            Provide a detailed research report with:
            1. Executive summary
            2. Key findings and insights
            3. Analysis of evidence quality
            4. Identification of gaps and limitations
            5. Recommendations for future research
            """

            deep_research_input = DeepResearchInputSchema(
                research_instructions=research_instructions,
                original_query=user_query,
                max_iterations=5,
                quality_threshold=0.7,
            )

            deep_research_output = await self.deep_research_agent.arun(
                deep_research_input
            )

            # Update results with DeepResearchAgent output
            research_report = deep_research_output.research_report
            key_findings = deep_research_output.key_findings
            quality_score = deep_research_output.evidence_quality_score
            iterations = deep_research_output.iterations_performed

            if not sources and deep_research_output.sources_consulted:
                sources = deep_research_output.sources_consulted

        # Create final summary
        research_summary = DeepResearchSummary(
            original_query=user_query,
            research_report=research_report,
            key_findings=key_findings,
            sources_consulted=sources,
            evidence_quality_score=quality_score,
            iterations_performed=iterations,
        )

        # Collect relevancy assessment statistics from all search results
        relevancy_stats = self._analyze_relevancy_results(search_results.results)

        # Store results in node state
        node_state.outputs["research_summary"] = research_summary
        node_state.outputs["research_report"] = research_report
        node_state.outputs["key_findings"] = key_findings
        node_state.outputs["sources_consulted"] = sources
        node_state.outputs["source_details"] = (
            source_details if "source_details" in locals() else []
        )
        node_state.outputs["sources_count"] = len(sources)
        node_state.outputs["evidence_quality_score"] = quality_score
        node_state.outputs["validation_reasoning"] = (
            validation_reasoning
            if "validation_reasoning" in locals()
            else "Quality validation completed"
        )
        node_state.outputs["relevancy_stats"] = relevancy_stats
        node_state.outputs["all_search_results"] = search_results.results

        if self.debug:
            logger.info(f"Lit research completed with {len(key_findings)} findings")

        return node_state

    def _analyze_relevancy_results(self, search_results):
        """Analyze relevancy assessment results from search results."""
        stats = {
            "total_results": len(search_results),
            "assessed_results": 0,
            "high_relevancy": 0,
            "medium_relevancy": 0,
            "low_relevancy": 0,
            "full_content_fetched": 0,
            "query_alignment_original": 0,
            "query_alignment_reformulated": 0,
            "average_relevancy_score": 0.0,
            "relevancy_scores": [],
            "filtered_results": [],
        }

        for result in search_results:
            # Skip the special research report result
            if hasattr(result, "url") and result.url == "deep-research://report":
                continue

            if (
                hasattr(result, "relevancy_score")
                and result.relevancy_score is not None
            ):
                stats["assessed_results"] += 1
                stats["relevancy_scores"].append(result.relevancy_score)

                # Categorize by relevancy level
                if result.relevancy_score >= 0.7:
                    stats["high_relevancy"] += 1
                elif result.relevancy_score >= 0.4:
                    stats["medium_relevancy"] += 1
                else:
                    stats["low_relevancy"] += 1

                # Track full content fetching
                if (
                    hasattr(result, "should_fetch_full_content")
                    and result.should_fetch_full_content
                ):
                    stats["full_content_fetched"] += 1

                # Track query alignment
                if (
                    hasattr(result, "query_alignment_details")
                    and result.query_alignment_details
                ):
                    best_query = result.query_alignment_details.get("best_query")
                    if best_query == "original":
                        stats["query_alignment_original"] += 1
                    elif best_query == "reformulated":
                        stats["query_alignment_reformulated"] += 1

        # Calculate average relevancy score
        if stats["relevancy_scores"]:
            stats["average_relevancy_score"] = sum(stats["relevancy_scores"]) / len(
                stats["relevancy_scores"]
            )

        return stats


# Main Workflow Class
class DeepResearchWorkflow:
    """workflow orchestrator using DeepResearchAgent."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.memory = InMemorySaver(serde=AKDSerializer())
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create workflow graph
        workflow = StateGraph(GlobalState)

        # Create node instances
        deep_research_node = ResearchNode(debug=self.debug, mutation=False)

        # Add nodes to workflow
        workflow.add_node("deep_research", deep_research_node.to_langgraph_node())

        # Define workflow edges - simple linear flow
        workflow.add_edge(START, "deep_research")
        workflow.add_edge("deep_research", END)

        return workflow.compile(checkpointer=self.memory)

    async def arun(self, query: str, max_results: int = 50) -> GlobalState:
        """Run the complete deep research workflow."""
        # Create initial state
        initial_state = GlobalState(
            node_states={
                "deep_research": NodeState(
                    inputs={
                        "user_query": query,
                        "max_results": max_results,
                    }
                )
            }
        )

        # Run workflow
        config = {"configurable": {"thread_id": "deep_research_session"}}
        final_state = await self.workflow.ainvoke(initial_state, config=config)

        return final_state

    async def stream_run(self, query: str, max_results: int = 50):
        """Run workflow with streaming output."""
        # Create initial state
        initial_state = GlobalState(
            node_states={
                "deep_research": NodeState(
                    inputs={
                        "user_query": query,
                        "max_results": max_results,
                    }
                )
            }
        )

        # Stream workflow execution
        config = {"configurable": {"thread_id": "deep_research_session"}}
        async for chunk in self.workflow.astream(initial_state, config=config):
            node_name = list(chunk.keys())[0] if chunk else "unknown"
            logger.info(f"Completed: {node_name}")
            yield chunk


async def main():
    """Main function to demonstrate the deep research workflow."""
    print("═" * 100)
    print("🔬 Deep Research Workflow Demo")
    print("═" * 100)

    # Create workflow instance
    workflow = DeepResearchWorkflow(debug=True)

    # Example queries for deep research
    test_queries = [
        "literature search for oil plantation classification using remote sensing data with machine learning methods",
    ]

    # Run a test query
    query = test_queries[0]
    print(f"\nRunning deep research workflow for: '{query}'")
    print("─" * 100)

    try:
        # Run the workflow
        result = await workflow.arun(query=query, max_results=30)

        # Extract the research summary from the final state
        research_node = result["node_states"].get("deep_research", NodeState())
        research_summary = research_node.outputs.get("research_summary")

        if research_summary:
            print("\n" + "═" * 100)
            print("LIT RESEARCH SUMMARY")
            print("═" * 100)
            print(f"Query: {research_summary.original_query}")
            print(f"Quality Score: {research_summary.evidence_quality_score:.2f}/1.0")
            print(f"Iterations: {research_summary.iterations_performed}")
            print(f"Sources: {len(research_summary.sources_consulted)}")

            # Display validation reasoning if available
            validation_reasoning = research_node.outputs.get("validation_reasoning", "")
            if validation_reasoning:
                print(f"Validation: {validation_reasoning}")

            # Display all search results for debugging
            all_search_results = research_node.outputs.get("all_search_results", [])
            if all_search_results:
                print("\n" + "─" * 100)
                print("🔍 ALL SEARCH RESULTS (for debugging):")
                print("─" * 100)
                for i, result in enumerate(all_search_results[:10], 1):  # Show first 10
                    relevancy_info = "❓ No score"
                    if hasattr(result, 'relevancy_score') and result.relevancy_score is not None:
                        score = result.relevancy_score
                        if score >= 0.7:
                            relevancy_info = f"🟢 {score:.3f}"
                        elif score >= 0.4:
                            relevancy_info = f"🟡 {score:.3f}"
                        else:
                            relevancy_info = f"🔴 {score:.3f}"
                    
                    title = (result.title or "Untitled")[:80]
                    if len(result.title or "") > 80:
                        title += "..."
                    print(f"  {i:2d}. {relevancy_info} | {title}")
                    
                    # Highlight problematic results
                    if result.title and "landslide" in result.title.lower():
                        print(f"      ⚠️  LANDSLIDE ARTICLE - Query was about oil plantation!")
                        if hasattr(result, 'relevancy_score'):
                            print(f"      ⚠️  Relevancy Score: {result.relevancy_score:.3f} (Should be low!)")
            
            # Display relevancy assessment statistics
            relevancy_stats = research_node.outputs.get("relevancy_stats", {})
            if relevancy_stats and relevancy_stats.get("assessed_results", 0) > 0:
                print("\n" + "─" * 100)
                print("🎯 RELEVANCY ASSESSMENT & FILTERING PROCESS")
                print("─" * 100)
                print(f"Total search results found: {relevancy_stats['total_results']}")
                print(
                    f"Results with relevancy assessment: {relevancy_stats['assessed_results']}"
                )
                print(
                    f"Average relevancy score of included results: {relevancy_stats['average_relevancy_score']:.3f}/1.0"
                )
                print()
                print("📊 Relevancy Distribution:")
                print(
                    f"  🟢 High relevancy (≥0.7): {relevancy_stats['high_relevancy']} results"
                )
                print(
                    f"  🟡 Medium relevancy (0.4-0.7): {relevancy_stats['medium_relevancy']} results"
                )
                print(
                    f"  🔴 Low relevancy (<0.4): {relevancy_stats['low_relevancy']} results"
                )
                print()
                print(
                    f"📄 Full content fetched: {relevancy_stats['full_content_fetched']} high-relevancy sources"
                )

                # Show query alignment if available
                total_aligned = (
                    relevancy_stats["query_alignment_original"]
                    + relevancy_stats["query_alignment_reformulated"]
                )
                if total_aligned > 0:
                    print()
                    print("🔄 Query Alignment Analysis:")
                    print(
                        f"  🎯 Better aligned with original query: {relevancy_stats['query_alignment_original']}"
                    )
                    print(
                        f"  🔄 Better aligned with reformulated queries: {relevancy_stats['query_alignment_reformulated']}"
                    )
                    effectiveness = (
                        relevancy_stats["query_alignment_reformulated"] / total_aligned
                    ) * 100
                    print(f"  📈 Query refinement effectiveness: {effectiveness:.1f}%")

            print("\n" + "─" * 100)
            print("RESEARCH REPORT")
            print("─" * 100)
            # Format the research report with better line breaks and indentation
            formatted_report = research_summary.research_report.replace(
                "\n\n", "\n\n  "
            )
            print(f"  {formatted_report.replace(chr(10), chr(10) + '  ')}")

            if research_summary.key_findings:
                print("\n" + "─" * 100)
                print("KEY FINDINGS")
                print("─" * 100)
                for i, finding in enumerate(research_summary.key_findings, 1):
                    # Wrap long findings for better readability
                    wrapped_finding = (
                        finding[:500] + "..." if len(finding) > 500 else finding
                    )
                    print(f"  {i}. {wrapped_finding}")

            # Enhanced source display with relevancy information
            source_details = research_node.outputs.get("source_details", [])

            if source_details:
                print("\n" + "─" * 100)
                print("📋 DETAILED SOURCES WITH RELEVANCY INFORMATION")
                print("─" * 100)

                # Debug: Check what's actually in source_details
                print(
                    "DEBUG: First source_detail keys:",
                    list(source_details[0].keys()) if source_details else "No sources",
                )
                print(
                    "DEBUG: Sample source_detail:",
                    source_details[0] if source_details else "No sources",
                )

                sources_shown = 0
                max_sources_to_show = 100

                # Display sources with relevancy information
                for i, source_detail in enumerate(source_details, 1):
                    if sources_shown >= max_sources_to_show:
                        break

                    url = source_detail.get("url", "")
                    title = source_detail.get("title", "Untitled")
                    preview = source_detail.get(
                        "content_preview", "No preview available"
                    )

                    # Get relevancy information directly from source_detail
                    relevancy_info = ""
                    relevancy_score = source_detail.get("relevancy_score")

                    # Debug: Check relevancy score
                    print(f"DEBUG: Source {i} relevancy_score: {relevancy_score}")

                    if relevancy_score is not None:
                        score = relevancy_score

                        # Add relevancy indicator
                        if score >= 0.7:
                            relevancy_info = f"🟢 {score:.2f}"
                        elif score >= 0.4:
                            relevancy_info = f"🟡 {score:.2f}"
                        else:
                            relevancy_info = f"🔴 {score:.2f}"

                        # Add full content indicator
                        if source_detail.get("should_fetch_full_content", False):
                            relevancy_info += " 📄"

                        # Add query alignment indicator
                        query_alignment = source_detail.get("query_alignment_details")
                        if query_alignment:
                            best_query = query_alignment.get("best_query")
                            if best_query == "original":
                                relevancy_info += " 🎯"
                            elif best_query == "reformulated":
                                relevancy_info += " 🔄"

                    print(f"  {i}. {title}")
                    if relevancy_info:
                        print(f"     Relevancy: {relevancy_info}")
                    print(f"     URL: {url}")
                    print(f"     Preview: {preview}")
                    print()
                    sources_shown += 1

                # Show legend
                print("─" * 100)
                print("📖 LEGEND:")
                print(
                    "  🟢 High relevancy (≥0.7)  🟡 Medium relevancy (0.4-0.7)  🔴 Low relevancy (<0.4)"
                )
                print("  📄 Full content fetched")

            elif research_summary.sources_consulted:
                print("\n" + "─" * 100)
                print("🔗 SOURCES CONSULTED")
                print("─" * 100)
                for i, source in enumerate(research_summary.sources_consulted[:100], 1):
                    print(f"  {i}. {source}")
        else:
            print("No research summary generated")

    except Exception as e:
        print(f"Error running workflow: {e}")
        raise

    print("\n" + "═" * 100)
    print("✅ Deep Research Workflow completed successfully!")
    print("═" * 100)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
