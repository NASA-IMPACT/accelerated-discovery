"""
Demonstration of LLM-based Workflow Planning in AKD

This script shows how to use the new LLM-based planning capabilities to:
1. Generate intelligent workflows from research queries
2. Analyze agent capabilities and compatibility
3. Execute workflows with reasoning traces
4. Compare with manual workflow creation

Enhanced with verbose output and debug mode for full LLM response visibility.
"""

import asyncio
import logging
import os
from typing import Dict, Any

from akd.planner import (AgentAnalyzer, PlannerConfig, ResearchPlanner,
                         ResearchPlanningInput)
from akd.configs.project import get_project_settings

# Configure logging for verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_planning_demo.log')
    ]
)

logger = logging.getLogger(__name__)


def check_api_keys():
    """Check and report on API key configuration"""
    print("\n🔑 API Key Configuration:")
    
    settings = get_project_settings()
    api_keys = settings.model_config_settings.api_keys
    
    keys_status = {
        "OpenAI": api_keys.openai is not None,
        "Anthropic": api_keys.anthropic is not None,
        "Ollama": api_keys.ollama is not None,
        "VLLM": api_keys.vllm is not None,
    }
    
    for provider, has_key in keys_status.items():
        status = "✓ Configured" if has_key else "❌ Missing"
        print(f"   {provider}: {status}")
    
    # Check environment variables directly
    env_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    print("\n   Environment Variables:")
    for key, value in env_keys.items():
        status = "✓ Set" if value else "❌ Not Set"
        print(f"     {key}: {status}")
    
    return any(keys_status.values())


async def demonstrate_llm_planning():
    """Demonstrate LLM-based workflow planning with verbose output"""
    
    print("=" * 60)
    print("AKD LLM-based Workflow Planning Demo")
    print("=" * 60)
    
    # Check API key configuration
    has_api_keys = check_api_keys()
    if not has_api_keys:
        print("\n⚠️  Warning: No API keys detected. LLM functionality may be limited.")
    
    # Configure planner for optimal reasoning and debug output
    config = PlannerConfig(
        auto_discover_agents=True,
        planning_model="gpt-4o-mini",  # Use o3-mini when available
        enable_reasoning_traces=True,
        enable_agent_analysis=True,
        max_workflow_nodes=10,
        enable_monitoring=True,
        enable_streaming=True,
        stream_intermediate_steps=True,
        enable_time_travel=True,
        enable_conditional_edges=True
    )
    
    # Add verbose logging
    logger.info("Initializing planner with configuration:")
    logger.info(f"  - Planning model: {config.planning_model}")
    logger.info(f"  - Reasoning traces: {config.enable_reasoning_traces}")
    logger.info(f"  - Agent analysis: {config.enable_agent_analysis}")
    logger.info(f"  - Monitoring: {config.enable_monitoring}")
    logger.info(f"  - Streaming: {config.enable_streaming}")
    
    # Initialize research planner
    print("\n1. Initializing Research Planner...")
    planner = ResearchPlanner(config)
    
    # Show discovered agents
    available_agents = planner.get_available_agents()
    print(f"   Discovered {len(available_agents)} agents:")
    for agent_name, agent_info in list(available_agents.items())[:5]:  # Show first 5
        print(f"   - {agent_name}: {agent_info.get('module', 'unknown')}")
    
    if len(available_agents) > 5:
        print(f"   ... and {len(available_agents) - 5} more")
    
    # Research query for planning
    research_query = "Conduct a comprehensive literature review on carbon capture materials, focusing on efficiency improvements and cost reduction strategies published in the last 3 years"
    
    print("\n2. Research Query:")
    print(f"   '{research_query}'")
    
    # Generate workflow plan
    print("\n3. Generating Workflow Plan...")
    try:
        planning_input = ResearchPlanningInput(
            research_query=research_query,
            requirements={
                "depth": "comprehensive",
                "time_constraint": "2_weeks",
                "focus_areas": ["efficiency", "cost_reduction"],
                "time_filter": "2021-2024"
            },
            constraints={
                "max_papers": 100,
                "preferred_sources": ["peer_reviewed_journals"]
            },
            session_id="demo_session"
        )
        
        result = await planner.execute_with_error_handling(planning_input)
        
        print("   ✓ Plan generated successfully!")
        print(f"   ✓ Confidence Score: {result.confidence_score:.2f}")
        print(f"   ✓ Estimated Duration: {result.estimated_duration} seconds")
        print(f"   ✓ Execution Ready: {result.execution_ready}")
        
        # Show workflow structure
        nodes = result.workflow_plan.get("nodes", [])
        edges = result.workflow_plan.get("edges", [])
        
        print("\n4. Generated Workflow Structure:")
        print(f"   Nodes: {len(nodes)}")
        for i, node in enumerate(nodes):
            print(f"   {i+1}. {node['agent_name']}: {node['description']}")
            print(f"      Duration: {node.get('estimated_duration', 'unknown')}s, "
                  f"Confidence: {node.get('confidence', 'unknown')}")
        
        print(f"\n   Edges: {len(edges)}")
        for i, edge in enumerate(edges):
            print(f"   {i+1}. {edge['source']} → {edge['target']}")
            if edge.get('data_mapping'):
                print(f"      Data mapping required: {edge['data_mapping'].get('mapping_required', False)}")
        
        # Show reasoning trace - FULL TRACE with verbose output
        print("\n5. LLM Reasoning Trace (FULL):")
        print("   " + "=" * 50)
        reasoning_lines = result.reasoning_trace.split('\n')
        for i, line in enumerate(reasoning_lines):
            if line.strip():
                print(f"   {i+1:3d}: {line.strip()}")
        print("   " + "=" * 50)
        
        logger.info("Full LLM reasoning trace captured:")
        logger.info(result.reasoning_trace)
        
        # Show agent capabilities used - FULL DETAILS
        print("\n6. Agent Capabilities Analysis (FULL):")
        print("   " + "-" * 50)
        for i, capability in enumerate(result.agent_capabilities):
            cap_dict = capability if isinstance(capability, dict) else capability
            agent_name = cap_dict.get('agent_name', 'Unknown')
            domain = cap_dict.get('domain', 'Unknown')
            capabilities = cap_dict.get('capabilities', [])
            confidence = cap_dict.get('confidence_score', 0.0)
            
            print(f"   Agent {i+1}: {agent_name}")
            print(f"     Domain: {domain}")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Capabilities: {', '.join(capabilities)}")
            if cap_dict.get('description'):
                print(f"     Description: {cap_dict['description']}")
            print()
        
        # Demonstrate execution if ready
        if result.execution_ready:
            print("\n7. Executing Workflow...")
            print("   Note: This will execute the actual agents with verbose output.")
            
            logger.info("Starting workflow execution...")
            
            # Demo execution
            try:
                execution_result = await planner.execute_workflow(
                    result.workflow_plan,
                    session_id="demo_execution"
                )
                
                print(f"   ✓ Execution Status: {execution_result.get('status', 'unknown')}")
                
                # Show detailed execution results
                if execution_result.get('status') == 'success':
                    print(f"   ✓ Final State: {execution_result.get('final_state', {}).get('workflow_status', 'unknown')}")
                    results = execution_result.get('results', {})
                    print(f"   ✓ Node Results: {len(results)} nodes completed")
                    
                    # Show results for each node
                    print("\n   Node Execution Details:")
                    for node_id, node_result in results.items():
                        print(f"     Node {node_id}:")
                        print(f"       Status: {node_result.get('status', 'unknown')}")
                        print(f"       Duration: {node_result.get('duration', 'unknown')}s")
                        if node_result.get('output'):
                            print(f"       Output: {str(node_result['output'])[:100]}...")
                        if node_result.get('error'):
                            print(f"       Error: {node_result['error']}")
                        print()
                else:
                    print(f"   ⚠ Execution failed: {execution_result.get('error', 'unknown error')}")
                    
                # Log full execution result
                logger.info("Full execution result:")
                logger.info(execution_result)
                    
            except Exception as e:
                print(f"   ⚠ Execution failed: {e}")
                logger.error(f"Execution error: {e}", exc_info=True)
                print("   This might be due to missing API keys or configuration issues.")
        
        else:
            print("\n7. Workflow Not Ready for Execution")
            print("   Reason: Missing agents or configuration issues")
            
            # Show validation results
            validation = await planner.validate_plan(result.workflow_plan)
            if not validation["valid"]:
                print("   Validation Errors:")
                for error in validation["errors"]:
                    print(f"     - {error}")
        
    except Exception as e:
        print(f"   ❌ Planning failed: {e}")
        print("   This might be due to missing API keys or model access.")
        
        # Show fallback to simple workflow
        print("\n   Demonstrating fallback to simple workflow...")
        await demonstrate_simple_workflow(planner, research_query)


async def demonstrate_simple_workflow(planner, research_query):
    """Demonstrate simple workflow creation as fallback"""
    
    print("\n8. Creating Simple Linear Workflow...")
    
    # Get available agents
    available_agents = planner.get_available_agents()
    agent_names = list(available_agents.keys())
    
    if len(agent_names) >= 2:
        # Create simple sequence
        simple_sequence = agent_names[:2]  # Use first 2 agents
        print(f"   Using agents: {' → '.join(simple_sequence)}")
        
        try:
            result = await planner.create_simple_workflow(
                research_query=research_query,
                agent_sequence=simple_sequence,
                session_id="demo_simple"
            )
            
            print("   ✓ Simple workflow executed successfully!")
            print(f"   ✓ Status: {result.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"   ❌ Simple workflow failed: {e}")
    else:
        print("   ⚠ Not enough agents available for simple workflow")


async def demonstrate_agent_analysis():
    """Demonstrate agent capability analysis with verbose output"""
    
    print("\n9. Agent Capability Analysis Demo...")
    
    config = PlannerConfig(
        auto_discover_agents=True,
        enable_agent_analysis=True,
        agent_analysis_model="gpt-4o-mini",
        enable_monitoring=True
    )
    
    logger.info("Starting agent analysis demonstration...")
    
    analyzer = AgentAnalyzer(config)
    
    # Get available agents from planner
    planner = ResearchPlanner(config)
    available_agents = planner.get_available_agents()
    
    if available_agents:
        print(f"   Found {len(available_agents)} agents to analyze")
        
        # Analyze multiple agents for comprehensive output
        for i, agent_name in enumerate(list(available_agents.keys())[:3]):  # Analyze first 3
            agent_class = planner.discovered_agents.get(agent_name)
            
            if agent_class:
                print(f"\n   Analyzing Agent {i+1}: {agent_name}")
                logger.info(f"Analyzing agent: {agent_name}")
                
                try:
                    capability = await analyzer.analyze_agent(agent_class)
                    
                    print("   ✓ Analysis complete!")
                    print(f"     Full Description: {capability.description}")
                    print(f"     Domain: {capability.domain}")
                    print(f"     All Capabilities: {', '.join(capability.capabilities)}")
                    print(f"     Confidence Score: {capability.confidence_score:.3f}")
                    
                    # Show additional details if available
                    if hasattr(capability, 'requirements'):
                        print(f"     Requirements: {capability.requirements}")
                    if hasattr(capability, 'output_types'):
                        print(f"     Output Types: {capability.output_types}")
                    if hasattr(capability, 'compatibility_notes'):
                        print(f"     Compatibility Notes: {capability.compatibility_notes}")
                    
                    # Log full capability analysis
                    logger.info(f"Agent {agent_name} analysis result:")
                    logger.info(f"  Description: {capability.description}")
                    logger.info(f"  Domain: {capability.domain}")
                    logger.info(f"  Capabilities: {capability.capabilities}")
                    logger.info(f"  Confidence: {capability.confidence_score}")
                    
                except Exception as e:
                    print(f"   ❌ Analysis failed: {e}")
                    logger.error(f"Agent analysis failed for {agent_name}: {e}", exc_info=True)
            else:
                print(f"   ⚠ Could not find agent class for {agent_name}")
                
        # Show compatibility analysis between agents
        print("\n   Agent Compatibility Analysis:")
        agent_names = list(available_agents.keys())[:2]  # Take first 2
        if len(agent_names) >= 2:
            try:
                # First analyze the individual agents to get AgentCapability objects
                agent1_class = planner.discovered_agents.get(agent_names[0])
                agent2_class = planner.discovered_agents.get(agent_names[1])
                
                agent1_capability = await analyzer.analyze_agent(agent1_class)
                agent2_capability = await analyzer.analyze_agent(agent2_class)
                
                # Now analyze compatibility
                compatibility = await analyzer.analyze_compatibility(
                    agent1_capability,
                    agent2_capability
                )
                
                print(f"   {agent_names[0]} ↔ {agent_names[1]}:")
                print(f"     Compatibility Score: {compatibility.compatibility_score:.3f}")
                print(f"     Data Mapping Required: {compatibility.data_mapping_required}")
                print(f"     Explanation: {compatibility.explanation}")
                print(f"     Compatibility Notes: {compatibility.compatibility_notes}")
                print(f"     Mapping Suggestions: {compatibility.mapping_suggestions}")
                
                logger.info(f"Compatibility analysis: {agent_names[0]} ↔ {agent_names[1]}")
                logger.info(f"  Score: {compatibility.compatibility_score}")
                logger.info(f"  Notes: {compatibility.compatibility_notes}")
                
            except Exception as e:
                print(f"   ❌ Compatibility analysis failed: {e}")
                logger.error(f"Compatibility analysis failed: {e}", exc_info=True)
        
    else:
        print("   ⚠ No agents available for analysis")


async def demonstrate_workflow_comparison():
    """Compare manual vs LLM-generated workflows"""
    
    print("\n10. Workflow Comparison: Manual vs LLM-Generated")
    
    # Manual workflow example
    manual_workflow = {
        "nodes": [
            {
                "node_id": "query_node",
                "agent_name": "QueryAgent",
                "description": "Process research query",
                "inputs": {"query": "carbon capture materials"},
                "config": {},
                "dependencies": [],
                "estimated_duration": 30,
                "confidence": 0.8
            },
            {
                "node_id": "search_node",
                "agent_name": "LiteratureSearchAgent",
                "description": "Search literature",
                "inputs": {},
                "config": {},
                "dependencies": ["query_node"],
                "estimated_duration": 120,
                "confidence": 0.7
            }
        ],
        "edges": [
            {
                "source": "query_node",
                "target": "search_node",
                "data_mapping": None,
                "condition": None
            }
        ]
    }
    
    print("   Manual Workflow:")
    print(f"     Nodes: {len(manual_workflow['nodes'])}")
    print(f"     Edges: {len(manual_workflow['edges'])}")
    print(f"     Total Duration: {sum(n['estimated_duration'] for n in manual_workflow['nodes'])}s")
    
    print("\n   LLM-Generated Workflow (from previous demo):")
    print("     ✓ Automatically selected appropriate agents")
    print("     ✓ Optimized for research query specifics")
    print("     ✓ Included reasoning trace for transparency")
    print("     ✓ Provided confidence scores")
    print("     ✓ Suggested data mappings between agents")


async def main():
    """Main demo function with comprehensive verbose output"""
    
    print("Starting AKD LLM-based Planning Demo...")
    print("Enhanced with verbose output, debug logging, and full LLM response visibility.")
    print("Note: This demo requires OpenAI API access for full functionality.")
    print("Some features may not work without proper API keys.\n")
    
    logger.info("Starting comprehensive LLM planning demonstration")
    
    try:
        # Core planning demo
        logger.info("Running core LLM planning demonstration...")
        await demonstrate_llm_planning()
        
        # Agent analysis demo
        logger.info("Running agent analysis demonstration...")
        await demonstrate_agent_analysis()
        
        # Comparison demo
        logger.info("Running workflow comparison demonstration...")
        await demonstrate_workflow_comparison()
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        
        print("\nKey Features Demonstrated:")
        print("✓ LLM-based workflow generation with FULL reasoning traces")
        print("✓ Agent capability analysis and compatibility assessment")
        print("✓ Intelligent agent selection based on research requirements")
        print("✓ Automatic data mapping between agents")
        print("✓ Fallback to simple workflows when needed")
        print("✓ Integration with existing AKD agent ecosystem")
        print("✓ Comprehensive debug logging and verbose output")
        print("✓ Full LLM response visibility and error tracking")
        
        print("\nDebug Information:")
        print("✓ Log file created: llm_planning_demo.log")
        print("✓ Verbose output enabled throughout execution")
        print("✓ All LLM responses captured and logged")
        print("✓ Error traces available for debugging")
        
        print("\nNext Steps:")
        print("1. Configure your OpenAI API key for full functionality")
        print("2. Try with different research queries")
        print("3. Experiment with custom requirements and constraints")
        print("4. Register your own agents with the planner")
        print("5. Use streaming execution for real-time updates")
        print("6. Review the log file for detailed LLM interactions")
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        print("This is likely due to missing dependencies or API configuration.")
        print("Check the log file for detailed error information.")


if __name__ == "__main__":
    asyncio.run(main())