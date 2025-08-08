#!/usr/bin/env python
# coding: utf-8

"""
Deep Search Agent Testing & Configuration Tool

This script provides a testing environment for scientists to experiment with and tweak 
the functionality of the DeepLitSearchAgent. It focuses on making configuration and 
prompts easily visible and adjustable while keeping the implementation details contained.

Features:
- Configurable Parameters: Easily adjust search behavior, thresholds, and agent settings
- Visible & Editable Prompts: View and modify system prompts used by embedded components  
- Interactive Testing: Run searches with different configurations and see results
- Result Analysis: Visualize and analyze search results and quality metrics
"""

import os
import sys
import tempfile
from pathlib import Path


import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# AKD imports
from akd.agents.search.deep_search import DeepLitSearchAgent, DeepLitSearchAgentConfig
from akd.agents.search._base import LitSearchAgentInputSchema
from akd.configs.project import get_project_settings
from akd.configs.prompts import (
    TRIAGE_AGENT_PROMPT,
    CLARIFYING_AGENT_PROMPT,
    RESEARCH_INSTRUCTION_AGENT_PROMPT,
    DEEP_RESEARCH_AGENT_PROMPT,
    MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT
)

# Custom prompts storage - can be modified during runtime
CUSTOM_PROMPTS = {
    "TRIAGE_AGENT_PROMPT": TRIAGE_AGENT_PROMPT,
    "CLARIFYING_AGENT_PROMPT": CLARIFYING_AGENT_PROMPT,
    "RESEARCH_INSTRUCTION_AGENT_PROMPT": RESEARCH_INSTRUCTION_AGENT_PROMPT,
    "DEEP_RESEARCH_AGENT_PROMPT": DEEP_RESEARCH_AGENT_PROMPT,
    "MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT": MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT
}

print("✅ Setup complete!")


# ## Configuration Panel
# 
# Adjust these parameters to experiment with different search behaviors:

# In[ ]:


# =============================================================================
# DEEP SEARCH AGENT CONFIGURATION
# =============================================================================

# Research Parameters
MAX_RESEARCH_ITERATIONS = 3  # Reduce for faster testing
QUALITY_THRESHOLD = 0.7      # Quality score threshold (0-1)
MAX_CLARIFYING_ROUNDS = 1    # Number of clarification rounds

# Search Behavior
AUTO_CLARIFY = True          # Automatically ask clarifying questions
USE_SEMANTIC_SCHOLAR = True  # Include Semantic Scholar searches
ENABLE_STREAMING = False     # Disable for notebook testing

# Link Assessment
ENABLE_PER_LINK_ASSESSMENT = True   # Enable relevancy assessment per link
MIN_RELEVANCY_SCORE = 0.3           # Minimum score to include results
FULL_CONTENT_THRESHOLD = 0.7        # Score threshold for full content fetch
ENABLE_FULL_CONTENT_SCRAPING = True # Enable full content scraping

# Debug Settings
DEBUG_MODE = True            # Enable detailed logging

print(f"📊 Configuration loaded:")
print(f"   Max Iterations: {MAX_RESEARCH_ITERATIONS}")
print(f"   Quality Threshold: {QUALITY_THRESHOLD}")
print(f"   Debug Mode: {DEBUG_MODE}")
print(f"   Semantic Scholar: {USE_SEMANTIC_SCHOLAR}")
print(f"   Per-link Assessment: {ENABLE_PER_LINK_ASSESSMENT}")


# ## System Prompts Viewer
# 
# View and optionally modify the system prompts used by the agent components:

# In[ ]:


# =============================================================================
# SYSTEM PROMPTS - VISIBLE AND CONFIGURABLE
# =============================================================================

def display_prompt(name: str, prompt: str, max_lines: int = 20):
    """Display a system prompt in a readable format."""
    lines = prompt.strip().split('\n')
    truncated = len(lines) > max_lines
    displayed_lines = lines[:max_lines] if truncated else lines

    print(f"\n{'='*60}")
    print(f"📝 {name}")
    print(f"{'='*60}")
    print('\n'.join(displayed_lines))

    if truncated:
        print(f"\n... ({len(lines) - max_lines} more lines) ...")
    print(f"\n{'='*60}")

# Display all system prompts
prompts = {
    "TRIAGE AGENT": TRIAGE_AGENT_PROMPT,
    "CLARIFICATION AGENT": CLARIFYING_AGENT_PROMPT, 
    "INSTRUCTION BUILDER": RESEARCH_INSTRUCTION_AGENT_PROMPT,
    "DEEP RESEARCH AGENT": DEEP_RESEARCH_AGENT_PROMPT,
    "RELEVANCY ASSESSOR": MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT
}

print("🎯 System Prompts Overview (first 10 lines each):")
for name, prompt in prompts.items():
    display_prompt(name, prompt, max_lines=10)


# ## Agent Initialization
# 
# Initialize the DeepLitSearchAgent with your configured parameters:

# In[ ]:


# =============================================================================
# AGENT INITIALIZATION
# =============================================================================

# Create configuration object
config = DeepLitSearchAgentConfig(
    max_research_iterations=MAX_RESEARCH_ITERATIONS,
    quality_threshold=QUALITY_THRESHOLD,
    auto_clarify=AUTO_CLARIFY,
    max_clarifying_rounds=MAX_CLARIFYING_ROUNDS,
    enable_streaming=ENABLE_STREAMING,
    use_semantic_scholar=USE_SEMANTIC_SCHOLAR,
    enable_per_link_assessment=ENABLE_PER_LINK_ASSESSMENT,
    min_relevancy_score=MIN_RELEVANCY_SCORE,
    full_content_threshold=FULL_CONTENT_THRESHOLD,
    enable_full_content_scraping=ENABLE_FULL_CONTENT_SCRAPING,
    debug=DEBUG_MODE
)

# Initialize the agent
print("🚀 Initializing DeepLitSearchAgent...")
agent = DeepLitSearchAgent(config=config, debug=DEBUG_MODE)
print("✅ Agent initialized successfully!")

# Display configuration summary
print("\n📋 Active Configuration:")
config_dict = config.model_dump()
for key, value in config_dict.items():
    print(f"   {key}: {value}")


# ## Testing Interface
# 
# Test the agent with different queries and see how configuration changes affect results:

# In[ ]:


# =============================================================================
# TEST QUERIES - MODIFY THESE TO TEST DIFFERENT SCENARIOS
# =============================================================================

# Predefined test queries for different complexity levels
TEST_QUERIES = {
    "simple": "machine learning in climate science",
    "specific": "deep learning approaches for satellite-based precipitation estimation accuracy improvements",
    "broad": "artificial intelligence applications in environmental science",
    "complex": "multi-modal deep learning for remote sensing data fusion in agricultural monitoring systems",
    "vague": "AI and environment"  # This should trigger clarification
}

# Display available test queries
print("🎯 Available Test Queries:")
for level, query in TEST_QUERIES.items():
    print(f"   {level.upper()}: {query}")

# Select query to test (modify this)
SELECTED_QUERY_TYPE = "specific"  # Change this to test different queries
CUSTOM_QUERY = None  # Set this to test your own query

# Use custom query if provided, otherwise use selected test query
query_to_test = CUSTOM_QUERY if CUSTOM_QUERY else TEST_QUERIES[SELECTED_QUERY_TYPE]

print(f"\n🔍 Selected Query ({SELECTED_QUERY_TYPE}): {query_to_test}")


# ## Run Deep Search
# 
# Execute the deep search with your selected query and configuration:

# In[ ]:


# =============================================================================
# EXECUTE DEEP SEARCH
# =============================================================================

async def run_deep_search(query: str, agent: DeepLitSearchAgent):
    """Run deep search and capture results."""
    print(f"🔍 Starting deep search for: '{query}'")
    print(f"⏱️  Max iterations: {agent.config.max_research_iterations}")
    print(f"📊 Quality threshold: {agent.config.quality_threshold}")
    print("-" * 60)

    # Create input schema
    input_schema = LitSearchAgentInputSchema(
        query=query,
        category="science",
        max_results=20
    )

    try:
        # Execute search
        result = await agent.arun(input_schema)

        print(f"\n✅ Search completed!")
        print(f"📈 Results found: {len(result.results)}")
        print(f"🔄 Iterations performed: {result.iterations_performed}")

        return result

    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
        raise e

# Note: This will be executed in main() function for proper async handling


# ## Results Analysis
# 
# Analyze and visualize the search results:

# In[ ]:


# =============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_search_results(results):
    """Analyze and visualize search results."""

    if not results or not results.results:
        print("❌ No results to analyze")
        return

    print(f"\n📊 SEARCH RESULTS ANALYSIS")
    print(f"={'='*50}")

    # Basic statistics
    total_results = len(results.results)
    print(f"\n📈 Basic Statistics:")
    print(f"   Total results: {total_results}")
    print(f"   Iterations: {results.iterations_performed}")

    # Check if first result is the research report
    has_research_report = False
    research_report = None
    actual_results = results.results

    if actual_results and actual_results[0].get('url') == 'deep-research://report':
        has_research_report = True
        research_report = actual_results[0]
        actual_results = actual_results[1:]  # Exclude report from analysis
        print(f"   Research report: ✅ Generated")
        print(f"   Search results: {len(actual_results)}")

    # Display research report if available
    if has_research_report and research_report:
        print(f"\n📋 RESEARCH REPORT SUMMARY:")
        print(f"   Quality Score: {research_report.get('quality_score', 'N/A')}")
        if 'key_findings' in research_report:
            print(f"   Key Findings: {len(research_report['key_findings'])} items")

    # Analyze result types and categories
    if actual_results:
        categories = [r.get('category', 'unknown') for r in actual_results]
        category_counts = pd.Series(categories).value_counts()

        print(f"\n📂 Result Categories:")
        for cat, count in category_counts.items():
            print(f"   {cat}: {count}")

        # Analyze content lengths
        content_lengths = [len(r.get('content', '')) for r in actual_results if r.get('content')]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            print(f"\n📝 Content Analysis:")
            print(f"   Average content length: {avg_length:.0f} characters")
            print(f"   Max content length: {max(content_lengths)}")
            print(f"   Min content length: {min(content_lengths)}")

        # Show relevancy scores if available
        relevancy_scores = [r.get('relevancy_score') for r in actual_results if r.get('relevancy_score')]
        if relevancy_scores:
            avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
            print(f"\n🎯 Relevancy Analysis:")
            print(f"   Average relevancy: {avg_relevancy:.3f}")
            print(f"   Max relevancy: {max(relevancy_scores):.3f}")
            print(f"   Min relevancy: {min(relevancy_scores):.3f}")

            # Plot relevancy distribution
            plt.figure(figsize=(10, 4))
            plt.hist(relevancy_scores, bins=10, alpha=0.7, color='skyblue')
            plt.axvline(MIN_RELEVANCY_SCORE, color='red', linestyle='--', label=f'Min Threshold ({MIN_RELEVANCY_SCORE})')
            plt.axvline(FULL_CONTENT_THRESHOLD, color='orange', linestyle='--', label=f'Full Content Threshold ({FULL_CONTENT_THRESHOLD})')
            plt.xlabel('Relevancy Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Relevancy Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

# Note: This will be executed in main() function


# ## Detailed Results Display
# 
# View detailed information about individual results:

# In[ ]:


# =============================================================================
# DETAILED RESULTS DISPLAY
# =============================================================================

def display_detailed_results(results, max_results: int = 5, show_content: bool = False):
    """Display detailed information about search results."""

    if not results or not results.results:
        print("❌ No results to display")
        return

    print(f"\n📋 DETAILED RESULTS (showing top {max_results})")
    print(f"={'='*60}")

    # Check if first result is research report
    start_idx = 0
    if results.results[0].get('url') == 'deep-research://report':
        research_report = results.results[0]
        print(f"\n🎯 RESEARCH REPORT")
        print(f"   Quality Score: {research_report.get('quality_score', 'N/A')}")
        print(f"   Iterations: {research_report.get('iterations', 'N/A')}")

        if show_content and research_report.get('content'):
            content = research_report['content']
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"\n📄 Report Preview:")
            print(preview)

        start_idx = 1
        print(f"\n{'-'*60}")

    # Display individual results
    actual_results = results.results[start_idx:start_idx + max_results]

    for i, result in enumerate(actual_results, 1):
        print(f"\n🔍 RESULT {i}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Category: {result.get('category', 'N/A')}")

        if 'relevancy_score' in result:
            score = result['relevancy_score']
            print(f"   Relevancy: {score:.3f} {'🟢' if score >= FULL_CONTENT_THRESHOLD else '🟡' if score >= MIN_RELEVANCY_SCORE else '🔴'}")

        if 'full_content_fetched' in result:
            print(f"   Full Content: {'✅ Fetched' if result['full_content_fetched'] else '❌ Not fetched'}")

        content = result.get('content', '')
        if content:
            print(f"   Content Length: {len(content)} chars")
            if show_content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Preview: {preview}")

        print(f"   {'-'*40}")

# Configuration for display
MAX_RESULTS_TO_SHOW = 3  # Adjust this number
SHOW_CONTENT_PREVIEW = True  # Set to False to hide content previews

# Note: This will be executed in main() function


# ## Configuration Comparison Tool
# 
# Compare results with different configurations to understand the impact of parameter changes:

# In[ ]:


# =============================================================================
# CONFIGURATION COMPARISON
# =============================================================================

async def compare_configurations(query: str, configs: Dict[str, DeepLitSearchAgentConfig]):
    """Compare results across different configurations."""

    results = {}

    for config_name, config in configs.items():
        print(f"\n🔧 Testing configuration: {config_name}")

        # Create agent with this config
        test_agent = DeepLitSearchAgent(config=config, debug=False)  # Disable debug for comparison

        # Run search
        input_schema = LitSearchAgentInputSchema(
            query=query,
            category="science",
            max_results=20
        )

        try:
            result = await test_agent.arun(input_schema)
            results[config_name] = {
                'total_results': len(result.results),
                'iterations': result.iterations_performed,
                'has_report': result.results[0].get('url') == 'deep-research://report' if result.results else False,
                'quality_score': result.results[0].get('quality_score') if result.results and result.results[0].get('url') == 'deep-research://report' else None
            }
            print(f"   ✅ {len(result.results)} results in {result.iterations_performed} iterations")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[config_name] = {'error': str(e)}

    return results

# Define configurations to compare
COMPARISON_CONFIGS = {
    "Fast (1 iter)": DeepLitSearchAgentConfig(
        max_research_iterations=1,
        quality_threshold=0.5,
        enable_per_link_assessment=False,
        enable_full_content_scraping=False,
        debug=False
    ),
    "Standard (2 iter)": DeepLitSearchAgentConfig(
        max_research_iterations=2,
        quality_threshold=0.7,
        enable_per_link_assessment=True,
        enable_full_content_scraping=True,
        debug=False
    ),
    "Thorough (3 iter)": DeepLitSearchAgentConfig(
        max_research_iterations=3,
        quality_threshold=0.8,
        enable_per_link_assessment=True,
        enable_full_content_scraping=True,
        min_relevancy_score=0.2,
        debug=False
    )
}

# Uncomment to run comparison (can be slow)
# print("🔄 Running configuration comparison...")
# comparison_results = await compare_configurations(query_to_test, COMPARISON_CONFIGS)

# # Display comparison
# print(f"\n📊 CONFIGURATION COMPARISON")
# print(f"={'='*60}")
# comparison_df = pd.DataFrame(comparison_results).T
# print(comparison_df)

print("💡 Uncomment the code above to run configuration comparison")


# ## Export Results
# 
# Export your results for further analysis:

# In[ ]:


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(results, query: str, config: DeepLitSearchAgentConfig, filename: str = None):
    """Export search results to JSON file."""

    if not filename:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deep_search_results_{timestamp}.json"

    export_data = {
        'query': query,
        'config': config.model_dump(),
        'results': results.model_dump() if hasattr(results, 'model_dump') else results,
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': {
            'total_results': len(results.results) if results and results.results else 0,
            'iterations_performed': results.iterations_performed if results else 0
        }
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Results exported to: {filename}")
        print(f"📁 File size: {os.path.getsize(filename)} bytes")

    except Exception as e:
        print(f"❌ Export failed: {str(e)}")

# Note: This will be executed in main() function


# ## Troubleshooting & Tips
# 
# ### Common Issues:
# 
# 1. **No results found**: Try reducing `MIN_RELEVANCY_SCORE` or increasing `MAX_RESEARCH_ITERATIONS`
# 2. **Search too slow**: Disable `ENABLE_FULL_CONTENT_SCRAPING` or reduce `MAX_RESEARCH_ITERATIONS`
# 3. **Poor quality results**: Increase `QUALITY_THRESHOLD` or enable `USE_SEMANTIC_SCHOLAR`
# 4. **Memory issues**: Reduce the number of results or disable content scraping
# 
# ### Optimization Tips:
# 
# - Use `DEBUG_MODE = True` to see detailed agent reasoning
# - Start with simple, specific queries for testing
# - Adjust relevancy thresholds based on your domain requirements
# - Consider the trade-off between search depth and execution time
# 
# ### For Scientists:
# 
# - **Domain-specific tuning**: Adjust `MIN_RELEVANCY_SCORE` based on your field's standards
# - **Source preferences**: Modify prompts to emphasize certain types of publications
# - **Quality metrics**: Experiment with `QUALITY_THRESHOLD` to balance comprehensiveness vs. precision
# - **Iteration tuning**: More iterations = more comprehensive but slower searches

# ## Quick Test Panel
# 
# Use this cell for quick experiments:

# In[ ]:


# =============================================================================
# QUICK TEST PANEL
# =============================================================================

# Quick configuration changes for rapid testing
QUICK_TEST_CONFIG = DeepLitSearchAgentConfig(
    max_research_iterations=1,  # Fast testing
    quality_threshold=0.5,      # Lower threshold
    enable_per_link_assessment=False,  # Disable for speed
    enable_full_content_scraping=False,  # Disable for speed
    debug=True  # See what's happening
)

QUICK_TEST_QUERY = "machine learning climate models"  # Simple test query

# Uncomment to run quick test
# print("🚀 Running quick test...")
# quick_agent = DeepLitSearchAgent(config=QUICK_TEST_CONFIG)
# quick_input = LitSearchAgentInputSchema(query=QUICK_TEST_QUERY, category="science")
# quick_results = await quick_agent.arun(quick_input)
# print(f"✅ Quick test complete: {len(quick_results.results)} results")

print("💡 Uncomment the code above to run a quick test")
print(f"📝 Test query: '{QUICK_TEST_QUERY}'")
print(f"⚡ Config: {QUICK_TEST_CONFIG.max_research_iterations} iteration(s), minimal features")


# =============================================================================
# PROMPT TWEAKING FUNCTIONALITY
# =============================================================================

def view_prompt(prompt_name: str) -> str:
    """View a specific system prompt."""
    if prompt_name in CUSTOM_PROMPTS:
        return CUSTOM_PROMPTS[prompt_name]
    else:
        available = ", ".join(CUSTOM_PROMPTS.keys())
        raise ValueError(f"Prompt '{prompt_name}' not found. Available: {available}")


def edit_prompt(prompt_name: str, new_prompt: str) -> None:
    """Edit a system prompt."""
    if prompt_name not in CUSTOM_PROMPTS:
        available = ", ".join(CUSTOM_PROMPTS.keys())
        raise ValueError(f"Prompt '{prompt_name}' not found. Available: {available}")
    
    CUSTOM_PROMPTS[prompt_name] = new_prompt
    print(f"✅ Updated prompt '{prompt_name}'")
    print(f"📊 New length: {len(new_prompt)} characters")


def save_custom_prompts(filename: Optional[str] = None) -> str:
    """Save custom prompts to a JSON file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_prompts_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(CUSTOM_PROMPTS, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Custom prompts saved to: {filename}")
    return filename


def load_custom_prompts(filename: str) -> None:
    """Load custom prompts from a JSON file."""
    global CUSTOM_PROMPTS
    
    with open(filename, 'r', encoding='utf-8') as f:
        loaded_prompts = json.load(f)
    
    CUSTOM_PROMPTS.update(loaded_prompts)
    print(f"✅ Custom prompts loaded from: {filename}")
    print(f"📊 Loaded {len(loaded_prompts)} prompts")


def reset_prompts() -> None:
    """Reset all prompts to their original values."""
    global CUSTOM_PROMPTS
    
    CUSTOM_PROMPTS = {
        "TRIAGE_AGENT_PROMPT": TRIAGE_AGENT_PROMPT,
        "CLARIFYING_AGENT_PROMPT": CLARIFYING_AGENT_PROMPT,
        "RESEARCH_INSTRUCTION_AGENT_PROMPT": RESEARCH_INSTRUCTION_AGENT_PROMPT,
        "DEEP_RESEARCH_AGENT_PROMPT": DEEP_RESEARCH_AGENT_PROMPT,
        "MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT": MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT
    }
    print("✅ All prompts reset to original values")


def interactive_prompt_editor():
    """Interactive prompt editing interface."""
    print("\n🎯 Interactive Prompt Editor")
    print("=" * 50)
    print("Available commands:")
    print("  list    - List all available prompts")
    print("  view    - View a specific prompt")
    print("  edit    - Edit a specific prompt")
    print("  save    - Save custom prompts to file")
    print("  load    - Load custom prompts from file")
    print("  reset   - Reset all prompts to original")
    print("  quit    - Exit editor")
    print()
    
    while True:
        try:
            command = input("📝 Enter command: ").strip().lower()
            
            if command == "quit":
                break
            elif command == "list":
                print("\n📋 Available Prompts:")
                for i, name in enumerate(CUSTOM_PROMPTS.keys(), 1):
                    length = len(CUSTOM_PROMPTS[name])
                    print(f"  {i}. {name} ({length} chars)")
            
            elif command == "view":
                prompt_name = input("Enter prompt name: ").strip()
                try:
                    prompt_text = view_prompt(prompt_name)
                    print(f"\n📄 {prompt_name}:")
                    print("-" * 40)
                    print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
                    print("-" * 40)
                    print(f"Total length: {len(prompt_text)} characters")
                except ValueError as e:
                    print(f"❌ {e}")
            
            elif command == "edit":
                prompt_name = input("Enter prompt name: ").strip()
                if prompt_name not in CUSTOM_PROMPTS:
                    print(f"❌ Prompt '{prompt_name}' not found")
                    continue
                    
                print(f"\n🖊️  Editing {prompt_name}")
                print("Enter new prompt (press Ctrl+D or Ctrl+Z when done):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    new_prompt = "\n".join(lines)
                    if new_prompt.strip():
                        edit_prompt(prompt_name, new_prompt)
                    else:
                        print("❌ Empty prompt not saved")
            
            elif command == "save":
                filename = input("Enter filename (or press Enter for auto-generated): ").strip()
                save_custom_prompts(filename if filename else None)
            
            elif command == "load":
                filename = input("Enter filename: ").strip()
                try:
                    load_custom_prompts(filename)
                except FileNotFoundError:
                    print(f"❌ File '{filename}' not found")
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in file '{filename}'")
            
            elif command == "reset":
                confirm = input("Reset all prompts to original? (y/N): ").strip().lower()
                if confirm == 'y':
                    reset_prompts()
                else:
                    print("❌ Reset cancelled")
            
            else:
                print(f"❌ Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting editor...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()  # Empty line for readability


def create_agent_with_custom_prompts(config: DeepLitSearchAgentConfig) -> DeepLitSearchAgent:
    """Create an agent with custom prompts injected."""
    # This is a simplified version - in practice, you'd need to modify the agent
    # to accept custom prompts or patch the prompt constants
    agent = DeepLitSearchAgent(config=config, debug=config.debug)
    
    # Store custom prompts as agent attributes for potential use
    agent._custom_prompts = CUSTOM_PROMPTS.copy()
    
    return agent


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

async def main():
    """Main execution function with proper async handling."""
    print("\n" + "=" * 70)
    print("🔬 DEEP SEARCH AGENT TESTING TOOL")
    print("=" * 70)
    
    # Show configuration
    print("\n📊 Current Configuration:")
    print(f"   Max Iterations: {MAX_RESEARCH_ITERATIONS}")
    print(f"   Quality Threshold: {QUALITY_THRESHOLD}")
    print(f"   Debug Mode: {DEBUG_MODE}")
    print(f"   Semantic Scholar: {USE_SEMANTIC_SCHOLAR}")
    print(f"   Per-link Assessment: {ENABLE_PER_LINK_ASSESSMENT}")
    
    # Show test query
    print(f"\n🔍 Test Query ({SELECTED_QUERY_TYPE}): {query_to_test}")
    
    # Prompt to continue or enter interactive mode
    print("\n🎯 Options:")
    print("  1. Run search with current configuration")
    print("  2. Enter interactive prompt editor")
    print("  3. Quick test (1 iteration, minimal features)")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Executing deep search...")
        search_results = await run_deep_search(query_to_test, agent)
        
        print("\n📊 Analyzing results...")
        analyze_search_results(search_results)
        
        print("\n📋 Detailed results:")
        display_detailed_results(search_results, MAX_RESULTS_TO_SHOW, SHOW_CONTENT_PREVIEW)
        
        # Export results
        export_results(search_results, query_to_test, config)
        
    elif choice == "2":
        interactive_prompt_editor()
        
    elif choice == "3":
        print("\n🚀 Running quick test...")
        quick_agent = DeepLitSearchAgent(config=QUICK_TEST_CONFIG)
        quick_input = LitSearchAgentInputSchema(query=QUICK_TEST_QUERY, category="science")
        quick_results = await quick_agent.arun(quick_input)
        print(f"✅ Quick test complete: {len(quick_results.results)} results")
        analyze_search_results(quick_results)
        
    elif choice == "4":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())

