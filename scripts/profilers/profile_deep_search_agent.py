#!/usr/bin/env python3
"""
Simple DeepLitSearchAgent Profiler - Use with memray run

This is a barebones test script for DeepLitSearchAgent that you run with memray directly:

    memray run --live profile_deep_search_agent.py
    memray run -o deep_search.bin profile_deep_search_agent.py
    memray run --live-remote profile_deep_search_agent.py

No Tracker() needed - memray wraps the entire script automatically.
"""

import asyncio

from akd.agents.search import (
    DeepLitSearchAgent,
    DeepLitSearchAgentConfig,
    LitSearchAgentInputSchema,
)


async def main():
    """Run DeepLitSearchAgent for profiling."""
    print("=" * 70)
    print("🔬 DeepLitSearchAgent - Memray Profiling")
    print("=" * 70)

    # Minimal configuration for profiling
    config = DeepLitSearchAgentConfig(
        max_research_iterations=1,
        quality_threshold=0.5,
        auto_clarify=False,
        debug=False,
    )

    print("├── Initializing DeepLitSearchAgent...")
    agent = DeepLitSearchAgent(config=config)

    test_input = LitSearchAgentInputSchema(
        query="transformer neural networks attention mechanisms",
        max_results=50,
    )

    print("├── Running DeepLitSearchAgent pipeline...")
    result = await agent.arun(test_input)

    print("├── ✓ Pipeline completed successfully")
    print(f"├── ✓ Results found: {len(result.results)}")
    print(f"├── ✓ Iterations: {result.iterations_performed}")
    print("└── Done!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
