#!/usr/bin/env python3
"""
Demo script for DeepLitSearchAgent - End-to-End Research Workflow

This script demonstrates the complete research workflow of the DeepLitSearchAgent,
including real LLM calls, search execution, and comprehensive research report generation.

Usage:
    python demo_deep_search.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from akd.configs.project import get_project_settings

# Add project root to path if needed
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from akd.agents.search import (
    DeepLitSearchAgent,
    DeepLitSearchAgentConfig,
    LitSearchAgentInputSchema,
)

def print_header():
    """Print demo header."""
    print("=" * 80)
    print("🔬 DeepLitSearchAgent - End-to-End Research Demo")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_section(title: str):
    """Print section header."""
    print("\n" + "─" * 60)
    print(f"📋 {title}")
    print("─" * 60)

def print_subsection(title: str):
    """Print subsection header."""
    print(f"\n🔹 {title}")
    print("-" * 40)

async def demo_basic_research():
    """Demo basic research workflow with minimal configuration."""
    
    print_section("BASIC RESEARCH DEMO")
    
    # Check API keys
    config = get_project_settings()
    if not config.model_config_settings.api_keys.openai:
        print("❌ No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    print(f"✅ API Key configured: {config.model_config_settings.api_keys.openai[:15]}...")
    
    # Configure agent for demo (cost-optimized)
    print_subsection("Agent Configuration")
    
    agent_config = DeepLitSearchAgentConfig(
        max_research_iterations=2,      # Limit iterations for demo
        quality_threshold=0.6,          # Reasonable threshold
        auto_clarify=False,             # Keep simple for demo
        use_semantic_scholar=False,     # Focus on primary search
        enable_per_link_assessment=False,  # Simplify for demo
        enable_full_content_scraping=False,  # Reduce complexity
        debug=True                      # Show debug info
    )
    
    print("Configuration:")
    print(f"  • Max iterations: {agent_config.max_research_iterations}")
    print(f"  • Quality threshold: {agent_config.quality_threshold}")
    print(f"  • Auto clarify: {agent_config.auto_clarify}")
    print(f"  • Semantic Scholar: {agent_config.use_semantic_scholar}")
    print(f"  • Link assessment: {agent_config.enable_per_link_assessment}")
    
    # Initialize agent
    print_subsection("Agent Initialization")
    print("🤖 Initializing DeepLitSearchAgent...")
    
    try:
        agent = DeepLitSearchAgent(config=agent_config)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Define research query
    research_query = "recent advances in transformer architectures for natural language processing"
    
    print_subsection("Research Query")
    print(f"Query: '{research_query}'")
    print(f"Max results: 8")
    
    # Prepare input
    input_params = LitSearchAgentInputSchema(
        query=research_query,
        max_results=8
    )
    
    # Execute research
    print_section("RESEARCH EXECUTION")
    print("🔍 Starting comprehensive literature research...")
    print("This may take 30-60 seconds as we make real LLM calls...\n")
    
    try:
        start_time = datetime.now()
        result = await agent._arun(input_params)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        print(f"✅ Research completed in {execution_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display results
    print_section("RESEARCH RESULTS")
    
    print(f"📊 Total results found: {len(result.results)}")
    print(f"🔄 Research iterations performed: {getattr(result, 'iterations_performed', 'N/A')}")
    
    if len(result.results) == 0:
        print("⚠️  No results returned")
        return False
    
    # Display the research report (first result)
    first_result = result.results[0]
    
    if first_result.get("url") == "deep-research://report":
        print_section("📑 COMPREHENSIVE RESEARCH REPORT")
        
        report_content = first_result.get("content", "")
        key_findings = first_result.get("key_findings", [])
        quality_score = first_result.get("quality_score", "N/A")
        sources_consulted = first_result.get("sources_consulted", [])
        citations = first_result.get("citations", [])
        
        print(f"📈 Quality Score: {quality_score}")
        print(f"📚 Sources Consulted: {len(sources_consulted)}")
        print(f"📝 Citations: {len(citations)}")
        
        print_subsection("Research Report Content")
        print(report_content[:1000] + "..." if len(report_content) > 1000 else report_content)
        
        if key_findings:
            print_subsection("Key Findings")
            for i, finding in enumerate(key_findings[:5], 1):
                print(f"{i}. {finding}")
        
        if sources_consulted:
            print_subsection("Sources Consulted")
            for i, source in enumerate(sources_consulted[:5], 1):
                print(f"{i}. {source}")
        
        if citations:
            print_subsection("Citations")
            for i, citation in enumerate(citations[:3], 1):
                print(f"{i}. {citation}")
    
    # Display additional search results
    if len(result.results) > 1:
        print_section("📄 ADDITIONAL SEARCH RESULTS")
        
        search_results = result.results[1:]  # Skip the report
        print(f"Found {len(search_results)} additional research papers/sources:")
        
        for i, item in enumerate(search_results[:5], 1):  # Show first 5
            title = item.get("title", "N/A")
            url = item.get("url", "N/A") 
            source = item.get("source", "N/A")
            
            print(f"\n{i}. {title}")
            print(f"   Source: {source}")
            print(f"   URL: {url}")
            
            content = item.get("content", "")
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Preview: {preview}")
    
    print_section("✨ DEMO COMPLETE")
    print("🎉 Successfully demonstrated end-to-end research workflow!")
    print(f"⚡ Total execution time: {execution_time:.1f} seconds")
    print("💡 The agent successfully:")
    print("   • Generated research queries using LLM")
    print("   • Executed web searches")
    print("   • Assessed result quality using LLM")
    print("   • Generated comprehensive research report")
    
    return True

async def demo_query_generation():
    """Demo just the query generation functionality."""
    
    print_section("QUERY GENERATION DEMO")
    
    config = get_project_settings()
    if not config.model_config_settings.api_keys.openai:
        print("❌ No OpenAI API key found.")
        return False
    
    agent_config = DeepLitSearchAgentConfig(debug=False)
    agent = DeepLitSearchAgent(config=agent_config)
    
    instructions = "Research applications of artificial intelligence in climate change mitigation and adaptation"
    
    print(f"Research Instructions: '{instructions}'")
    print("\n🤖 Generating research queries using LLM...")
    
    try:
        queries = await agent._generate_initial_queries(instructions)
        
        print(f"✅ Generated {len(queries)} research queries:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
            
        return True
        
    except Exception as e:
        print(f"❌ Query generation failed: {e}")
        return False

async def main():
    """Main demo function."""
    
    print_header()
    
    print("This demo will showcase the DeepLitSearchAgent's capabilities:")
    print("• Real LLM calls for query generation and analysis")
    print("• Web search execution and result processing")
    print("• Comprehensive research report generation")
    print("\nNote: This demo makes real API calls and may take 30-60 seconds to complete.")
    
    # Check if user wants to continue
    try:
        response = input("\nProceed with demo? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Demo cancelled.")
            return
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    # Run basic research demo
    success = await demo_basic_research()
    
    if success:
        print("\n" + "=" * 80)
        print("Would you like to see a quick query generation demo?")
        try:
            response = input("Run query generation demo? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                await demo_query_generation()
        except KeyboardInterrupt:
            pass
    
    print("\n" + "=" * 80)
    print("🙏 Thank you for trying the DeepLitSearchAgent demo!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)