#!/usr/bin/env python3
"""
Simple Gap Agent Profiler - Use with memray run

This is a barebones test script for Gap Agent that you run with memray directly:

    memray run --live profile_gap_agent.py
    memray run -o gap_agent.bin profile_gap_agent.py
    memray run --live-remote profile_gap_agent.py

No Tracker() needed - memray wraps the entire script automatically.
"""

import asyncio

from pydantic import AnyUrl

from akd.agents.gap_analysis import GapAgent, GapAgentConfig, GapInputSchema
from akd.configs.project import get_project_settings
from akd.structures import SearchResultItem
from akd.tools.scrapers import DoclingScraperConfig
from akd.tools.search import SemanticScholarSearchToolConfig


async def main():
    """Run Gap Agent for profiling."""
    print("=" * 70)
    print("🔬 Gap Agent - Memray Profiling")
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
