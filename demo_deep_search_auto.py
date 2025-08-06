#!/usr/bin/env python3
"""
Automatic Demo script for DeepLitSearchAgent - No interaction required

This script runs the complete research workflow automatically without user input.
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
    print("🔬 DeepLitSearchAgent - Automatic Research Demo")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_section(title: str):
    """Print section header."""
    print("\n" + "─" * 60)
    print(f"📋 {title}")
    print("─" * 60)

async def main():
    """Main demo function."""
    
    print_header()
    
    # Check API keys
    config = get_project_settings()
    if not config.model_config_settings.api_keys.openai:
        print("❌ No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    print(f"✅ API Key configured: {config.model_config_settings.api_keys.openai[:15]}...")
    
    # Configure agent for demo (cost-optimized)
    print_section("Agent Configuration")
    
    agent_config = DeepLitSearchAgentConfig(
        max_research_iterations=1,      # Single iteration for demo
        quality_threshold=0.6,          # Reasonable threshold
        auto_clarify=False,             # Keep simple for demo
        use_semantic_scholar=False,     # Focus on primary search
        enable_per_link_assessment=False,  # Simplify for demo
        enable_full_content_scraping=False,  # Reduce complexity
        debug=False                     # Reduce noise
    )
    
    print("Configuration:")
    print(f"  • Max iterations: {agent_config.max_research_iterations}")
    print(f"  • Quality threshold: {agent_config.quality_threshold}")
    print(f"  • Auto clarify: {agent_config.auto_clarify}")
    
    # Initialize agent
    print_section("Agent Initialization")
    print("🤖 Initializing DeepLitSearchAgent...")
    
    try:
        agent = DeepLitSearchAgent(config=agent_config)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Define research query
    research_query = "quantum computing error correction recent advances"
    
    print_section("Research Query")
    print(f"Query: '{research_query}'")
    print(f"Max results: 6")
    
    # Prepare input
    input_params = LitSearchAgentInputSchema(
        query=research_query,
        max_results=6
    )
    
    # Execute research
    print_section("Research Execution")
    print("🔍 Starting comprehensive literature research...")
    print("⏳ Making real LLM calls and web searches...\n")
    
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
    print_section("Research Results")
    
    print(f"📊 Total results found: {len(result.results)}")
    print(f"🔄 Research iterations: {getattr(result, 'iterations_performed', 'N/A')}")
    
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
        
        print("\n🔹 Research Report Content")
        print("-" * 40)
        if len(report_content) > 1200:
            print(report_content[:1200] + "\n\n... [Report continues] ...")
        else:
            print(report_content)
        
        if key_findings:
            print("\n🔹 Key Findings")
            print("-" * 40)
            for i, finding in enumerate(key_findings[:4], 1):
                print(f"{i}. {finding}")
        
        if sources_consulted:
            print("\n🔹 Sources Consulted")
            print("-" * 40)
            for i, source in enumerate(sources_consulted[:3], 1):
                print(f"{i}. {source}")
    
    # Display additional search results
    if len(result.results) > 1:
        print_section("📄 Additional Search Results")
        
        search_results = result.results[1:]  # Skip the report
        print(f"Found {len(search_results)} additional research papers/sources:\n")
        
        for i, item in enumerate(search_results[:4], 1):  # Show first 4
            title = item.get("title", "N/A")
            url = item.get("url", "N/A") 
            
            print(f"{i}. {title}")
            print(f"   URL: {url}")
            
            content = item.get("content", "")
            if content:
                preview = content[:180] + "..." if len(content) > 180 else content
                print(f"   Preview: {preview}")
            print()
    
    print_section("✨ Demo Complete")
    print("🎉 Successfully demonstrated end-to-end research workflow!")
    print(f"⚡ Total execution time: {execution_time:.1f} seconds")
    print("\n💡 The DeepLitSearchAgent successfully:")
    print("   • Generated research queries using real LLM")
    print("   • Executed web searches through SearxNG")
    print("   • Assessed result quality using LLM")
    print("   • Generated comprehensive research report")
    print("   • Provided structured findings and citations")
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)