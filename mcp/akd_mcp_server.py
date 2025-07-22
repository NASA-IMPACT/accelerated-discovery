#!/usr/bin/env python3
"""
FastMCP Server for AKD Research Workflow

This server exposes the AKD research workflow and agents via MCP (Model Context Protocol).
It provides tools for:
- Running the complete research workflow
- Generating search queries
- Performing literature searches
- Analyzing content for code search needs
- Searching code repositories
- Generating research reports
"""

import asyncio
from typing import Dict, List, Any, Optional
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from AKD framework sources
from akd.agents.query import QueryAgent, QueryAgentInputSchema
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolInputSchema, SearxNGSearchToolConfig
from akd.tools.code_search import LocalRepoCodeSearchTool, LocalRepoCodeSearchToolInputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.nodes.states import NodeState
from akd._base import InputSchema, OutputSchema
from loguru import logger

# Custom schemas for NeedsCodeSearchAgent
class NeedsCodeSearchInputSchema(InputSchema):
    """Input schema for determining whether code repo search is needed"""
    
    content: str = Field(
        ...,
        description="Input research content from literature search which affects the decision.",
    )
    original_query: Optional[str] = Field(
        None,
        description="Original query that led to this content from literature search",
    )


class NeedsCodeSearchOutputSchema(OutputSchema):
    """Output schema for determining whether code repo search is needed"""
    
    needs_code_search: bool = Field(
        ..., description="Boolean output for requiring code search or not"
    )
    confidence: float = Field(
        ..., description="Confidence value assigned for the decision"
    )
    reasoning: List[str] = Field(
        ..., description="Concise reasoning traces for the decision"
    )


class NeedsCodeSearchAgent(InstructorBaseAgent):
    """Agent that determines if code search is needed based on literature search results."""
    input_schema = NeedsCodeSearchInputSchema
    output_schema = NeedsCodeSearchOutputSchema


class AKDWorkflow:
    """Simplified workflow orchestrator for MCP server."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Initialize agents and tools
        self.query_agent = QueryAgent(
            BaseAgentConfig(model_name="gpt-4-turbo-preview"),
            debug=debug
        )
        
        self.search_tool = SearxNGSearchTool(
            SearxNGSearchToolConfig(
                engines=["arxiv", "google_scholar"],
                max_results=50
            ),
            debug=debug
        )
        
        self.code_search_tool = LocalRepoCodeSearchTool(debug=debug)
        
        self.needs_code_agent = NeedsCodeSearchAgent(
            BaseAgentConfig(
                model_name="gpt-4-turbo-preview",
                system_prompt=(
                    "You are an expert at determining whether a code repository search would be beneficial "
                    "based on literature search results. Analyze the content and decide if looking at actual "
                    "code implementations would provide additional value."
                )
            ),
            debug=debug
        )
    
    async def run_workflow(self, research_question: str, max_sources: int = 10) -> Dict[str, Any]:
        """Run the complete research workflow."""
        results = {
            "query": research_question,
            "generated_queries": [],
            "literature_results": [],
            "code_results": [],
            "needs_code_search": False,
            "summary": ""
        }
        
        try:
            # Step 1: Generate queries
            query_input = QueryAgentInputSchema(query=research_question, num_queries=3)
            query_result = await self.query_agent.arun(query_input)
            results["generated_queries"] = query_result.queries
            
            # Step 2: Search literature
            search_input = SearxNGSearchToolInputSchema(
                queries=query_result.queries,
                category=query_result.category,
                max_results=max_sources
            )
            search_result = await self.search_tool.arun(search_input)
            results["literature_results"] = [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet
                }
                for item in search_result.results[:max_sources]
            ]
            
            # Step 3: Determine if code search is needed
            if results["literature_results"]:
                content = "\n".join([
                    f"{r['title']}: {r['snippet']}"
                    for r in results["literature_results"]
                ])
                
                needs_code_input = NeedsCodeSearchInputSchema(
                    content=content,
                    original_query=research_question
                )
                needs_code_result = await self.needs_code_agent.arun(needs_code_input)
                results["needs_code_search"] = needs_code_result.needs_code_search
                
                # Step 4: Optional code search
                if needs_code_result.needs_code_search:
                    code_input = LocalRepoCodeSearchToolInputSchema(
                        queries=[research_question],
                        category="technology",
                        max_results=5
                    )
                    code_result = await self.code_search_tool.arun(code_input)
                    results["code_results"] = [
                        {
                            "file": item.file_path,
                            "content": item.content[:200] + "..."
                        }
                        for item in code_result.results
                    ]
            
            # Generate summary
            results["summary"] = self._generate_summary(results)
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the research findings."""
        summary_parts = [
            f"Research on: {results['query']}",
            f"\nGenerated {len(results['generated_queries'])} search queries.",
            f"\nFound {len(results['literature_results'])} literature sources."
        ]
        
        if results["needs_code_search"]:
            summary_parts.append(f"\nCode search was performed and found {len(results['code_results'])} relevant code examples.")
        else:
            summary_parts.append("\nCode search was not deemed necessary for this query.")
        
        return "\n".join(summary_parts)


# Initialize FastMCP server
mcp = FastMCP("akd-research-workflow")

# Initialize shared instances
workflow_instance: Optional[AKDWorkflow] = None
query_agent: Optional[QueryAgent] = None
search_tool: Optional[SearxNGSearchTool] = None
code_search_tool: Optional[LocalRepoCodeSearchTool] = None
needs_code_agent: Optional[NeedsCodeSearchAgent] = None


def initialize_components():
    """Initialize all workflow components."""
    global workflow_instance, query_agent, search_tool, code_search_tool, needs_code_agent
    
    # Initialize workflow
    workflow_instance = AKDWorkflow(debug=True)
    
    # Initialize individual components
    query_agent = QueryAgent(
        BaseAgentConfig(model_name="gpt-4o-mini"),
        debug=True
    )
    
    search_tool = SearxNGSearchTool(
        SearxNGSearchToolConfig(
            engines=["arxiv", "google_scholar"],
            max_results=50
        ),
        debug=False
    )
    
    code_search_tool = LocalRepoCodeSearchTool(debug=True)
    
    needs_code_agent = NeedsCodeSearchAgent(
        BaseAgentConfig(
            model_name="gpt-4o-mini",
            system_prompt="""
            It's applicable to do code search when the input content (which is the result from literature search) 
            entails for a code search that potentially can add more evidence and values to the literature search. 
            If it does, there needs to be a code search.
            If the content has self-sufficient evidence and need any code associated with it, it does not need code search.
            Primary purpose is to enhance literature review process, in assisting the user/scientist to help discover 
            related tools and code base for the research topic.
            """
        )
    )


# Tool 1: Run complete research workflow
@mcp.tool()
async def run_research_workflow(
    query: str = Field(..., description="Research query to investigate"),
    num_queries: int = Field(default=5, description="Number of search queries to generate"),
    max_sources: int = Field(default=10, description="Maximum number of sources to retrieve")
) -> Dict[str, Any]:
    """
    Run the complete AKD research workflow.
    
    This executes the full pipeline:
    1. Query generation
    2. Literature search
    3. Content analysis
    4. Conditional code search
    5. Report generation
    """
    if not workflow_instance:
        initialize_components()
    
    try:
        # Run the workflow
        result = await workflow_instance.run_workflow(research_question=query, max_sources=max_sources)
        
        # Return the workflow results
        if "error" not in result:
            return {
                "status": "success",
                "query": result["query"],
                "summary": result["summary"],
                "generated_queries": result["generated_queries"],
                "literature_findings": [r["title"] for r in result["literature_results"]],
                "code_findings": [r["file"] for r in result["code_results"]],
                "sources": [r["url"] for r in result["literature_results"]],
                "total_sources": len(result["literature_results"]),
                "needs_code_search": result["needs_code_search"]
            }
        else:
            return {
                "status": "error",
                "message": result.get("error", "Unknown error occurred")
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Tool 2: Generate search queries
@mcp.tool()
async def generate_search_queries(
    query: str = Field(..., description="Original research query"),
    num_queries: int = Field(default=3, description="Number of queries to generate")
) -> Dict[str, Any]:
    """
    Generate multiple search queries from a single research question.
    Uses the QueryAgent to expand and refine the original query.
    """
    if not query_agent:
        initialize_components()
    
    try:
        query_agent.reset_memory()
        result = await query_agent.arun(
            QueryAgentInputSchema(query=query, num_queries=num_queries)
        )
        
        return {
            "status": "success",
            "original_query": query,
            "generated_queries": result.queries,
            "category": result.category
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Tool 3: Search literature
@mcp.tool()
async def search_literature(
    queries: List[str] = Field(..., description="List of search queries"),
    category: str = Field(default="science", description="Search category"),
    max_results: int = Field(default=50, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search academic literature and web sources using SearxNG.
    Returns research papers, articles, and other relevant sources.
    """
    if not search_tool:
        initialize_components()
    
    try:
        result = await search_tool.arun(
            SearxNGSearchToolInputSchema(
                queries=queries,
                category=category,
                max_results=max_results
            )
        )
        
        # Format results
        formatted_results = []
        for r in result.results[:10]:  # Limit to first 10 for readability
            formatted_results.append({
                "title": r.title,
                "url": str(r.url),
                "content": r.content[:200] + "..." if r.content else "",
                "published_date": str(r.published_date) if r.published_date else None
            })
        
        return {
            "status": "success",
            "total_results": len(result.results),
            "results": formatted_results,
            "queries_used": queries
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Tool 4: Analyze content for code search need
@mcp.tool()
async def analyze_content_for_code_search(
    content: str = Field(..., description="Literature content to analyze"),
    original_query: Optional[str] = Field(None, description="Original research query")
) -> Dict[str, Any]:
    """
    Analyze literature content to determine if code search would be beneficial.
    Returns a decision with confidence and reasoning.
    """
    if not needs_code_agent:
        initialize_components()
    
    try:
        needs_code_agent.reset_memory()
        result = await needs_code_agent.arun(
            NeedsCodeSearchInputSchema(
                content=content,
                original_query=original_query
            )
        )
        
        return {
            "status": "success",
            "needs_code_search": result.needs_code_search,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Tool 5: Search code repositories
@mcp.tool()
async def search_code_repositories(
    queries: List[str] = Field(..., description="Search queries for code"),
    max_results: int = Field(default=10, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search local code repositories for relevant implementations.
    Useful for finding code examples, tools, and libraries related to research.
    """
    if not code_search_tool:
        initialize_components()
    
    try:
        result = await code_search_tool.arun(
            LocalRepoCodeSearchToolInputSchema(
                queries=queries,
                max_results=max_results
            )
        )
        
        # Format results
        formatted_results = []
        for r in result.results:
            formatted_results.append({
                "title": r.title,
                "url": str(r.url),
                "content": r.content[:200] + "..." if r.content else "",
            })
        
        return {
            "status": "success",
            "total_results": len(result.results),
            "results": formatted_results,
            "queries_used": queries
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Tool 6: Get workflow status
@mcp.tool()
async def get_workflow_status() -> Dict[str, Any]:
    """
    Get the current status of the workflow components.
    Useful for debugging and monitoring.
    """
    return {
        "workflow_initialized": workflow_instance is not None,
        "query_agent_initialized": query_agent is not None,
        "search_tool_initialized": search_tool is not None,
        "code_search_tool_initialized": code_search_tool is not None,
        "needs_code_agent_initialized": needs_code_agent is not None,
        "debug_mode": workflow_instance.debug if workflow_instance else False
    }


# Tool 7: Stream workflow execution
@mcp.tool()
async def stream_research_workflow(
    query: str = Field(..., description="Research query to investigate"),
    num_queries: int = Field(default=5, description="Number of search queries to generate")
) -> Dict[str, Any]:
    """
    Run the research workflow with step-by-step updates.
    Returns progress information as the workflow executes.
    """
    if not workflow_instance:
        initialize_components()
    
    try:
        steps = []
        
        # Simulate streaming by calling the regular workflow and returning step info
        steps.append("Starting research workflow...")
        steps.append("Generating search queries...")
        
        result = await workflow_instance.run_workflow(research_question=query, max_sources=10)
        
        steps.append(f"Generated {len(result.get('generated_queries', []))} queries")
        steps.append("Searching literature...")
        steps.append(f"Found {len(result.get('literature_results', []))} sources")
        
        if result.get("needs_code_search"):
            steps.append("Performing code search...")
            steps.append(f"Found {len(result.get('code_results', []))} code examples")
        
        steps.append("Generating summary...")
        
        return {
            "status": "success",
            "steps": steps,
            "summary": result.get("summary", "No summary generated"),
            "total_sources": len(result.get("literature_results", [])),
            "query": query
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "steps": steps if 'steps' in locals() else []
        }


# Initialize components when the module is imported
initialize_components()
logger.info("AKD Research Workflow MCP Server initialized")


if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()