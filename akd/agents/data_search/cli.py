#!/usr/bin/env python
"""
Command-line interface for DataSearchAgent.

Usage:
    # Single path mode with fast model
    uv run akd/agents/data_search/cli.py "query" --single-path --model gpt-5-nano

    # Full workflow with output saving
    uv run akd/agents/data_search/cli.py "query" --model gpt-5-mini
"""

import argparse
import asyncio

from dotenv import load_dotenv

from akd.agents.data_search import DataSearchAgent, DataSearchAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema
from akd.agents.data_search.handlers import CMRHandlerConfig
from akd.configs.data_search_config import get_config

# Load environment variables
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(
        description="NASA Data Search Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development: Single path with fast model
  %(prog)s "atmospheric CO2 2020-2023" --single-path --model gpt-5-nano

  # Production: Full workflow with output saving
  %(prog)s "flood risk Mississippi" --model gpt-5-mini

  # Custom configuration with per-component models
  %(prog)s "MODIS SST data" --topic-model gpt-5-nano --cmr-known-model gpt-5-mini
        """,
    )

    parser.add_argument(
        "query",
        help="Natural language research query",
    )

    parser.add_argument(
        "--single-path",
        action="store_true",
        help="Execute single path [0] at each branch for fast testing",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable auto-save (default: auto-save enabled)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata capture (git, prompts, config)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use for all components (default: gpt-5-mini)",
    )

    # Per-component model overrides
    parser.add_argument(
        "--topic-model",
        help="Model for topic splitting (overrides --model)",
    )

    parser.add_argument(
        "--decomp-model",
        help="Model for scientific decomposition (overrides --model)",
    )

    parser.add_argument(
        "--routing-model",
        help="Model for repository routing (overrides --model)",
    )

    parser.add_argument(
        "--cmr-known-model",
        help="Model for CMR known parameters (overrides --model)",
    )

    parser.add_argument(
        "--cmr-searchable-model",
        help="Model for CMR searchable parameters (overrides --model)",
    )

    parser.add_argument(
        "--cmr-filtering-model",
        help="Model for CMR approach filtering (overrides --model)",
    )

    parser.add_argument(
        "--cmr-ranking-model",
        help="Model for CMR final ranking (overrides --model)",
    )

    args = parser.parse_args()

    # Load base configuration
    base_config = get_config()

    # Build CMR handler configuration
    cmr_config = CMRHandlerConfig(
        mcp_endpoint=str(base_config.mcp.endpoint),
        collection_search_page_size=20,
        granule_search_page_size=10,
        collections_per_query=5,
        max_collections_per_approach=5,
        final_collection_count=25,
        min_collection_relevance_score=0.3,
        collection_search_timeout=30.0,
        granule_search_timeout=45.0,
        enable_parallel_search=True,
        # Per-component models with fallback to --model
        known_parameters_model=args.cmr_known_model or args.model,
        searchable_parameters_model=args.cmr_searchable_model or args.model,
        approach_filtering_model=args.cmr_filtering_model or args.model,
        final_ranking_model=args.cmr_ranking_model or args.model,
    )

    # Build agent configuration
    agent_config = DataSearchAgentConfig(
        debug=args.debug,
        single_path_mode=args.single_path,
        auto_save=not args.no_save,
        capture_metadata=not args.no_metadata,
        # Universal component models with fallback to --model
        topic_splitting_model=args.topic_model or args.model,
        scientific_decomposition_model=args.decomp_model or args.model,
        repository_routing_model=args.routing_model or args.model,
        # Handler-specific configurations
        cmr=cmr_config,
    )

    # Create agent and run
    agent = DataSearchAgent(config=agent_config, debug=args.debug)

    print(f"🔍 Query: {args.query}")
    if args.single_path:
        print("⚡ Mode: Single-path execution (selects [0] at each branch)")
    if not args.no_save:
        print("💾 Auto-save: Enabled")
    print(f"🤖 Model: {args.model}")
    if args.topic_model or args.decomp_model or args.routing_model:
        print("   Per-component overrides:")
        if args.topic_model:
            print(f"   • Topic splitting: {args.topic_model}")
        if args.decomp_model:
            print(f"   • Decomposition: {args.decomp_model}")
        if args.routing_model:
            print(f"   • Routing: {args.routing_model}")
    if (
        args.cmr_known_model
        or args.cmr_searchable_model
        or args.cmr_filtering_model
        or args.cmr_ranking_model
    ):
        print("   CMR component overrides:")
        if args.cmr_known_model:
            print(f"   • Known parameters: {args.cmr_known_model}")
        if args.cmr_searchable_model:
            print(f"   • Searchable parameters: {args.cmr_searchable_model}")
        if args.cmr_filtering_model:
            print(f"   • Approach filtering: {args.cmr_filtering_model}")
        if args.cmr_ranking_model:
            print(f"   • Final ranking: {args.cmr_ranking_model}")
    print()

    try:
        result = await agent.arun(DataSearchAgentInputSchema(query=args.query))

        # Summary
        print(
            f"\n✅ Search completed in {result.search_metadata['duration_seconds']:.1f}s",
        )
        print(f"📊 Topics processed: {result.search_metadata['topics_processed']}")
        print(f"🔍 Total from CMR: {result.total_cmr_results} collections")
        print(f"📦 After filtering: {result.total_filtered_results} collections")

        if result.total_cmr_results > 0:
            retention = (result.total_filtered_results / result.total_cmr_results) * 100
            print(f"🎯 Retention rate: {retention:.1f}%")

        if not args.no_save and result.search_metadata.get("search_id"):
            from akd.agents.data_search.utils.metadata import build_output_filename

            output_file = build_output_filename(result.search_metadata["search_id"])
            print(f"💾 Saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
