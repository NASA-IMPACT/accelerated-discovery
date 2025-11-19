#!/usr/bin/env python3
"""
Memray-based Memory Profiler for AKD Agents

A barebones profiler using Memray to track memory allocations in Gap Agent and
DeepLitSearchAgent. No code modifications needed - just run and visualize.

Usage:
    # Profile Gap Agent
    python akd_memory_profiler_memray.py --agent gap

    # Profile with LIVE MODE (real-time web interface)
    python akd_memory_profiler_memray.py --agent gap --live
    # Opens http://localhost:8080 - see class/method allocations in real-time!

    # Profile DeepLitSearchAgent
    python akd_memory_profiler_memray.py --agent deep_search

    # Profile both agents
    python akd_memory_profiler_memray.py --agent both

    # Custom output directory
    python akd_memory_profiler_memray.py --agent gap --output-dir ./profiles

After profiling (non-live mode), view results:
    memray flamegraph gap_agent_profile.bin      # Interactive visualization
    memray table gap_agent_profile.bin           # Top allocators
    memray tree gap_agent_profile.bin            # Call tree
    memray stats gap_agent_profile.bin           # Summary statistics

Requirements:
    - memray: uv pip install memray
    - Set OPENAI_API_KEY in .env file
    - Internet connection for fetching papers

Author: AKD Team
Version: 1.1.0
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from memray import Tracker
from pydantic import AnyUrl

# Configure logger
logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>", level="INFO")


# ============================================================================
# Configuration
# ============================================================================

# Test data for GapAgent (real arXiv URL from tests)
GAP_AGENT_TEST_URL = "http://arxiv.org/abs/2504.06136v1"
GAP_AGENT_PDF_URL = "http://arxiv.org/pdf/2504.06136v1"
GAP_AGENT_TITLE = "QGen Studio: An Adaptive Question-Answer Generation Platform"

# Test query for DeepLitSearchAgent
DEEP_SEARCH_TEST_QUERY = "transformer neural networks attention mechanisms"


# ============================================================================
# Gap Agent Profiling
# ============================================================================


async def profile_gap_agent(output_file: Path, live_mode: bool = False):
    """Profile Gap Agent with Memray."""
    from akd.agents.gap_analysis import GapAgent, GapAgentConfig, GapInputSchema
    from akd.configs.project import get_project_settings
    from akd.structures import SearchResultItem
    from akd.tools.scrapers import DoclingScraperConfig
    from akd.tools.search import SemanticScholarSearchToolConfig

    logger.info("=" * 70)
    logger.info("<cyan>🔬 Profiling Gap Agent with Memray</cyan>")
    logger.info("=" * 70)

    # Setup configuration
    project_settings = get_project_settings()
    openai_key = project_settings.model_config_settings.api_keys.openai

    docling_config = DoclingScraperConfig(
        do_table_structure=True,
        pdf_mode="fast",  # Use fast mode for profiling
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
    )

    logger.info("├── Initializing Gap Agent...")
    agent = GapAgent(gap_agent_config)

    # Create test input with one paper
    search_results = [
        SearchResultItem(
            url=AnyUrl(GAP_AGENT_TEST_URL),
            title=GAP_AGENT_TITLE,
            query="test query for profiling",
            pdf_url=AnyUrl(GAP_AGENT_PDF_URL),
            content="Test content for memory profiling with Memray",
        ),
    ]

    test_input = GapInputSchema(
        search_results=search_results,
        gap="methodology",
    )

    logger.info("├── Starting Memray tracking...")
    if live_mode:
        logger.info("├── <cyan>LIVE MODE: Web interface will open at http://localhost:8080</cyan>")
        logger.info("├── <cyan>Press Ctrl+C when done profiling</cyan>")
    else:
        logger.info(f"├── Output file: <green>{output_file}</green>")

    # Profile with Memray
    tracker_args = {"file_name": str(output_file)}
    if live_mode:
        tracker_args["live"] = True

    with Tracker(**tracker_args):
        logger.info("├── Running Gap Agent pipeline...")
        try:
            result = await agent.arun(test_input)
            logger.info("├── ✓ Pipeline completed successfully")
            logger.info(f"├── ✓ Graph nodes: {len(result.graph.get('nodes', []))}")
        except Exception as e:
            logger.error(f"├── ✗ Error during execution: {e}")
            raise

    logger.info("└── <green>Memray tracking complete!</green>")
    logger.info("")


# ============================================================================
# DeepLitSearchAgent Profiling
# ============================================================================


async def profile_deep_search_agent(output_file: Path, live_mode: bool = False):
    """Profile DeepLitSearchAgent with Memray."""
    from akd.agents.search import (
        DeepLitSearchAgent,
        DeepLitSearchAgentConfig,
        LitSearchAgentInputSchema,
    )

    logger.info("=" * 70)
    logger.info("<cyan>🔬 Profiling DeepLitSearchAgent with Memray</cyan>")
    logger.info("=" * 70)

    # Minimal configuration for faster profiling
    config = DeepLitSearchAgentConfig(
        max_research_iterations=1,  # Reduced for profiling
        quality_threshold=0.5,
        auto_clarify=False,  # Skip clarification for simpler profiling
        debug=False,
    )

    logger.info("├── Initializing DeepLitSearchAgent...")
    agent = DeepLitSearchAgent(config=config)

    test_input = LitSearchAgentInputSchema(
        query=DEEP_SEARCH_TEST_QUERY,
        max_results=5,
    )

    logger.info("├── Starting Memray tracking...")
    if live_mode:
        logger.info("├── <cyan>LIVE MODE: Web interface will open at http://localhost:8080</cyan>")
        logger.info("├── <cyan>Press Ctrl+C when done profiling</cyan>")
    else:
        logger.info(f"├── Output file: <green>{output_file}</green>")

    # Profile with Memray
    tracker_args = {"file_name": str(output_file)}
    if live_mode:
        tracker_args["live"] = True

    with Tracker(**tracker_args):
        logger.info("├── Running DeepLitSearchAgent pipeline...")
        try:
            result = await agent.arun(test_input)
            logger.info("├── ✓ Pipeline completed successfully")
            logger.info(f"├── ✓ Results found: {len(result.results)}")
            logger.info(f"├── ✓ Iterations: {result.iterations_performed}")
        except Exception as e:
            logger.error(f"├── ✗ Error during execution: {e}")
            raise

    logger.info("└── <green>Memray tracking complete!</green>")
    logger.info("")


# ============================================================================
# Main CLI
# ============================================================================


async def main():
    """Main entry point for the Memray profiler."""
    parser = argparse.ArgumentParser(
        description="Memray-based memory profiler for AKD agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agent",
        choices=["gap", "deep_search", "both"],
        default="gap",
        help="Which agent to profile (default: gap)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for memray output files (default: current directory)",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live mode - opens web interface at http://localhost:8080",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for API key
    try:
        from akd.configs.project import get_project_settings

        settings = get_project_settings()
        if not settings.model_config_settings.api_keys.openai:
            logger.error("⚠️  OPENAI_API_KEY not found in environment. Please set it in .env")
            sys.exit(1)
    except Exception as e:
        logger.error(f"⚠️  Error loading config: {e}")
        sys.exit(1)

    # Check if memray is installed
    try:
        import memray  # noqa
    except ImportError:
        logger.error("⚠️  Memray is not installed. Install it with: uv pip install memray")
        sys.exit(1)

    logger.info("")
    logger.info("<cyan>🚀 AKD Memory Profiler - Memray Edition</cyan>")
    logger.info("")

    try:
        if args.agent in ["gap", "both"]:
            gap_output = output_dir / "gap_agent_profile.bin"
            await profile_gap_agent(gap_output, live_mode=args.live)

        if args.agent in ["deep_search", "both"]:
            deep_output = output_dir / "deep_search_agent_profile.bin"
            await profile_deep_search_agent(deep_output, live_mode=args.live)

        # Print next steps (skip if in live mode)
        if not args.live:
            logger.info("=" * 70)
            logger.info("<green>✅ Profiling Complete!</green>")
            logger.info("=" * 70)
            logger.info("")
            logger.info("<cyan>📊 View Results:</cyan>")
            logger.info("")

            if args.agent in ["gap", "both"]:
                gap_file = output_dir / "gap_agent_profile.bin"
                logger.info("<yellow>Gap Agent:</yellow>")
                logger.info(f"  memray flamegraph {gap_file}  # Interactive visualization")
                logger.info(f"  memray table {gap_file}       # Top allocators")
                logger.info(f"  memray tree {gap_file}        # Call tree")
                logger.info(f"  memray stats {gap_file}       # Summary")
                logger.info("")

            if args.agent in ["deep_search", "both"]:
                deep_file = output_dir / "deep_search_agent_profile.bin"
                logger.info("<yellow>DeepLitSearchAgent:</yellow>")
                logger.info(f"  memray flamegraph {deep_file}  # Interactive visualization")
                logger.info(f"  memray table {deep_file}       # Top allocators")
                logger.info(f"  memray tree {deep_file}        # Call tree")
                logger.info(f"  memray stats {deep_file}       # Summary")
                logger.info("")

            logger.info("<cyan>💡 Tip:</cyan> Use flamegraph for best visual insight into memory allocations!")
            logger.info("")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error during profiling: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
