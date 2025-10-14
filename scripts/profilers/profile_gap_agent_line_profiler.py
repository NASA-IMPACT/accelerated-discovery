#!/usr/bin/env python3
"""
Gap Agent Line Profiler - Use with kernprof/line_profiler

Run this script with line_profiler to get line-by-line timing:

    # Install line_profiler if not already installed
    uv pip install line_profiler

    # Run with line profiler
    kernprof -l -v scripts/profile_gap_agent_line_profiler.py

    # Or use the @profile decorator and run:
    python -m line_profiler -m scripts.profile_gap_agent_line_profiler

This will show detailed timing for each line in the @profile decorated functions.
"""

import asyncio

from pydantic import AnyUrl

from akd.agents.gap_analysis import GapAgent, GapAgentConfig, GapInputSchema
from akd.configs.project import get_project_settings
from akd.structures import SearchResultItem
from akd.tools.scrapers import DoclingScraperConfig
from akd.tools.search import SemanticScholarSearchToolConfig


async def main():
    """Run Gap Agent for line profiling."""
    print("=" * 70)
    print("🔬 Gap Agent - Line Profiler")
    print("=" * 70)

    # Setup configuration
    project_settings = get_project_settings()
    openai_key = project_settings.model_config_settings.api_keys.openai

    docling_config = DoclingScraperConfig(
        do_table_structure=True,
        pdf_mode="fast",
        export_type="html",
        debug=False,
    )

    s2_config = SemanticScholarSearchToolConfig(
        debug=False,
        external_id="ARXIV",
        fields=["paperId", "title", "externalIds", "isOpenAccess", "openAccessPdf"],
    )

    gap_agent_config = GapAgentConfig(
        docling_config=docling_config,
        s2_tool_config=s2_config,
        model_name="gpt-4o-mini",
        api_key=openai_key,
        debug=False,
        output_graph=True,
    )

    print("├── Initializing Gap Agent...")
    agent = GapAgent(gap_agent_config)

    # Create test input
    search_results = [
        SearchResultItem(
            url=AnyUrl("https://arxiv.org/abs/2411.08181"),
            title="Challenges in Guardrailing Large Language Models for Science",
            query="llm guardrails for science",
            pdf_url=AnyUrl("http://arxiv.org/pdf/2411.08181"),
            content="LLM guardrails for science",
        ),
    ]

    test_input = GapInputSchema(
        search_results=search_results,
        gap="methodology",
    )

    print("├── Running Gap Agent pipeline...")
    result = await agent.arun(test_input)

    print("├── ✓ Pipeline completed successfully")
    if result.graph:
        print(f"├── ✓ Graph nodes: {len(result.graph.get('nodes', []))}")
    print("└── Done!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
