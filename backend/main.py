"""
Backend main module for AKD framework.
Creates LangGraph workflows using existing NodeTemplate architecture.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

# Add parent directory to path to import AKD modules
sys.path.append(str(Path(__file__).parent.parent))

from akd.agents.extraction import EstimationExtractionAgent
from akd.agents.litsearch import LiteratureSearchAgent
from akd.agents.planner import PlannerAgent
from akd.agents.relevancy import RelevancyAgent
from akd.configs.project import ProjectSettings
from akd.nodes.states import GlobalState
from akd.nodes.supervisor import LLMSupervisor, ManualSupervisor, ReActLLMSupervisor
from akd.nodes.templates import AbstractNodeTemplate
from akd.tools.relevancy import RelevancyTool
from akd.tools.scrapers.pdf import PDFScraperTool
from akd.tools.scrapers.web import WebScraperTool
from akd.tools.search import SearxNGSearchTool, SemanticScholarSearchTool


def create_plan_node() -> AbstractNodeTemplate:
    """Create a planning node using existing PlannerAgent."""
    settings = ProjectSettings()
    
    # Create planner agent
    planner = PlannerAgent(
        model=settings.get_model_config("planner_agent")
    )
    
    # Create supervisor for the planning node
    supervisor = LLMSupervisor(
        agent=planner,
        tools=[],  # Planner doesn't need tools
        name="PlannerSupervisor",
        description="Supervisor for research planning"
    )
    
    # Create node template
    node = AbstractNodeTemplate(
        supervisor=supervisor,
        input_guardrails=[],
        output_guardrails=[],
        name="plan_node",
        description="Research planning node that creates execution plans"
    )
    
    return node


def create_literature_search_node() -> AbstractNodeTemplate:
    """Create a literature search node with search tools."""
    settings = ProjectSettings()
    
    # Create search tools
    searxng_tool = SearxNGSearchTool(
        base_url=settings.tools.searxng_url,
        api_key=settings.tools.searxng_api_key
    )
    
    semantic_scholar_tool = SemanticScholarSearchTool()
    
    # Create literature search agent
    lit_agent = LiteratureSearchAgent(
        model=settings.get_model_config("lit_search_agent")
    )
    
    # Create ReAct supervisor with tools
    supervisor = ReActLLMSupervisor(
        agent=lit_agent,
        tools=[searxng_tool, semantic_scholar_tool],
        name="LiteratureSearchSupervisor",
        description="Supervisor for literature search with web search capabilities"
    )
    
    # Create node template
    node = AbstractNodeTemplate(
        supervisor=supervisor,
        input_guardrails=[],
        output_guardrails=[],
        name="literature_search_node",
        description="Literature search node that finds relevant papers"
    )
    
    return node


def create_relevancy_check_node() -> AbstractNodeTemplate:
    """Create a relevancy checking node."""
    settings = ProjectSettings()
    
    # Create relevancy tool
    relevancy_tool = RelevancyTool()
    
    # Create relevancy agent
    relevancy_agent = RelevancyAgent(
        model=settings.get_model_config("relevancy_agent")
    )
    
    # Create supervisor
    supervisor = LLMSupervisor(
        agent=relevancy_agent,
        tools=[relevancy_tool],
        name="RelevancySupervisor",
        description="Supervisor for checking content relevancy"
    )
    
    # Create node template
    node = AbstractNodeTemplate(
        supervisor=supervisor,
        input_guardrails=[],
        output_guardrails=[],
        name="relevancy_check_node",
        description="Node that checks relevancy of found literature"
    )
    
    return node


def create_extraction_node() -> AbstractNodeTemplate:
    """Create an extraction node with PDF/Web scraping tools."""
    settings = ProjectSettings()
    
    # Create scraping tools
    pdf_tool = PDFScraperTool()
    web_tool = WebScraperTool()
    
    # Create extraction agent
    extraction_agent = EstimationExtractionAgent(
        model=settings.get_model_config("extraction_agent")
    )
    
    # Create supervisor
    supervisor = ReActLLMSupervisor(
        agent=extraction_agent,
        tools=[pdf_tool, web_tool],
        name="ExtractionSupervisor",
        description="Supervisor for extracting data from documents"
    )
    
    # Create node template
    node = AbstractNodeTemplate(
        supervisor=supervisor,
        input_guardrails=[],
        output_guardrails=[],
        name="extraction_node",
        description="Node that extracts structured data from documents"
    )
    
    return node


def create_human_review_node() -> AbstractNodeTemplate:
    """Create a human-in-the-loop review node."""
    # Create manual supervisor for human interaction
    supervisor = ManualSupervisor(
        name="HumanReviewSupervisor",
        description="Supervisor for human review and approval"
    )
    
    # Create node template
    node = AbstractNodeTemplate(
        supervisor=supervisor,
        input_guardrails=[],
        output_guardrails=[],
        name="human_review_node",
        description="Human review checkpoint for validation"
    )
    
    return node


def create_plan() -> StateGraph:
    """
    Create a research planning workflow using LangGraph and NodeTemplate.
    
    This function creates a complete research workflow that:
    1. Plans the research approach
    2. Searches for relevant literature
    3. Checks relevancy of found papers
    4. Extracts data from relevant sources
    5. Allows human review at key checkpoints
    
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    # Initialize the graph with GlobalState
    workflow = StateGraph(GlobalState)
    
    # Create nodes
    plan_node = create_plan_node()
    lit_search_node = create_literature_search_node()
    relevancy_node = create_relevancy_check_node()
    extraction_node = create_extraction_node()
    human_review_node = create_human_review_node()
    
    # Convert nodes to LangGraph format
    nodes: List[Tuple[str, AbstractNodeTemplate]] = [
        ("planner", plan_node),
        ("literature_search", lit_search_node),
        ("relevancy_check", relevancy_node),
        ("extraction", extraction_node),
        ("human_review", human_review_node)
    ]
    
    # Add nodes to graph with retry policies
    for name, node in nodes:
        workflow.add_node(
            name,
            node.to_langgraph_node(key=name),
            retry=RetryPolicy(max_attempts=3)
        )
    
    # Define workflow edges
    workflow.set_entry_point("planner")
    
    # Sequential flow with conditional branches
    workflow.add_edge("planner", "literature_search")
    workflow.add_edge("literature_search", "relevancy_check")
    
    # Conditional edge based on relevancy
    def route_after_relevancy(state: GlobalState) -> str:
        """Route based on relevancy check results."""
        # Check if we have relevant results
        relevancy_state = state.node_states.get("relevancy_check", {})
        if relevancy_state.get("has_relevant_results", False):
            return "extraction"
        else:
            # Go back to search with refined query
            return "literature_search"
    
    workflow.add_conditional_edges(
        "relevancy_check",
        route_after_relevancy,
        {
            "extraction": "extraction",
            "literature_search": "literature_search"
        }
    )
    
    # Human review after extraction
    workflow.add_edge("extraction", "human_review")
    
    # End after human review
    workflow.add_edge("human_review", END)
    
    return workflow


def create_plan_with_checkpointing() -> StateGraph:
    """
    Create a plan with checkpointing enabled for persistence.
    
    Returns:
        Compiled StateGraph with memory checkpointing
    """
    # Create the base workflow
    workflow = create_plan()
    
    # Compile with memory checkpointer
    # In production, this would be replaced with RedisCheckpointer
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


def create_custom_plan(
    node_configs: List[Dict[str, any]],
    edge_configs: List[Tuple[str, str]],
    conditional_edges: Optional[Dict[str, any]] = None
) -> StateGraph:
    """
    Create a custom plan with specified nodes and edges.
    
    Args:
        node_configs: List of node configurations
        edge_configs: List of edge tuples (from_node, to_node)
        conditional_edges: Optional conditional routing configuration
        
    Returns:
        StateGraph: Custom workflow graph
    """
    workflow = StateGraph(GlobalState)
    
    # Add nodes based on configuration
    for config in node_configs:
        node_type = config.get("type")
        node_name = config.get("name")
        
        # Factory pattern for node creation
        if node_type == "planner":
            node = create_plan_node()
        elif node_type == "literature_search":
            node = create_literature_search_node()
        elif node_type == "relevancy_check":
            node = create_relevancy_check_node()
        elif node_type == "extraction":
            node = create_extraction_node()
        elif node_type == "human_review":
            node = create_human_review_node()
        else:
            raise ValueError(f"Unknown node type: {node_type}")
        
        workflow.add_node(
            node_name,
            node.to_langgraph_node(key=node_name),
            retry=RetryPolicy(max_attempts=config.get("retry_attempts", 3))
        )
    
    # Set entry point (first node)
    if node_configs:
        workflow.set_entry_point(node_configs[0]["name"])
    
    # Add edges
    for from_node, to_node in edge_configs:
        if to_node == "END":
            workflow.add_edge(from_node, END)
        else:
            workflow.add_edge(from_node, to_node)
    
    # Add conditional edges if provided
    if conditional_edges:
        for source, config in conditional_edges.items():
            workflow.add_conditional_edges(
                source,
                config["function"],
                config["mapping"]
            )
    
    return workflow


if __name__ == "__main__":
    # Example usage
    print("Creating research planning workflow...")
    
    # Create basic plan
    workflow = create_plan()
    compiled_workflow = workflow.compile()
    
    print("Workflow created successfully!")
    print(f"Nodes: {list(workflow.nodes.keys())}")
    print(f"Edges: {workflow.edges}")
    
    # Create workflow with checkpointing
    print("\nCreating workflow with checkpointing...")
    checkpointed_workflow = create_plan_with_checkpointing()
    print("Checkpointed workflow created!")
    
    # Example of custom workflow
    print("\nCreating custom workflow...")
    custom_config = [
        {"type": "planner", "name": "research_planner"},
        {"type": "literature_search", "name": "paper_finder"},
        {"type": "extraction", "name": "data_extractor"}
    ]
    
    custom_edges = [
        ("research_planner", "paper_finder"),
        ("paper_finder", "data_extractor"),
        ("data_extractor", "END")
    ]
    
    custom_workflow = create_custom_plan(custom_config, custom_edges)
    print("Custom workflow created!")