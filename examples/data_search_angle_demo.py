#!/usr/bin/env python3
"""
Demo script showcasing the angle-based CMR data search pipeline.

This script demonstrates the complete workflow from a scientific query
to organized results by scientific angle, showing the output at each step.

Usage:
    cd /Users/cdavis/github/accelerated-discovery
    uv run python examples/data_search_angle_demo.py
"""

import asyncio
from datetime import datetime

from dotenv import load_dotenv

from akd.agents.data_search import DataSearchAgent, DataSearchAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema
from akd.agents.data_search.handlers import CMRHandlerConfig
from akd.configs.data_search_config import get_config

# Load environment variables from .env file
load_dotenv()


def print_step_header(step_num: int, step_name: str):
    """Print a formatted step header."""
    print(f"\n{'=' * 60}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'=' * 60}")


def print_scientific_angles(angles_data):
    """Print scientific angles in a readable format."""
    print(f"\n📐 Generated {len(angles_data)} scientific angles:")
    for i, angle in enumerate(angles_data, 1):
        print(f"\n  {i}. {angle.get('title', 'Unknown title')}")
        print(
            f"     Justification: {angle.get('scientific_justification', 'No justification')}",
        )


def print_angle_summary(angle_result, angle_num: int):
    """Print summary for a single angle's results."""
    angle = angle_result.scientific_angle
    print(f"\n🔬 ANGLE {angle_num}: {angle.get('title', 'Unknown')}")
    print(
        f"   Justification: {angle.get('scientific_justification', 'No justification')}",
    )
    print("   📊 Results:")
    print(f"      • {len(angle_result.cmr_queries)} CMR queries generated")
    print(f"      • {angle_result.total_collections_found} collections found")
    print(
        f"      • {len(angle_result.collections)} collections selected (after deduplication/ranking)",
    )
    print(f"      • {angle_result.total_granules_found} data files found")

    if angle_result.cmr_queries:
        print("\n   🔍 CMR Queries:")
        for i, query in enumerate(angle_result.cmr_queries, 1):
            params = []
            if query.get("keyword"):
                params.append(f"keyword='{query['keyword']}'")
            if query.get("platform"):
                params.append(f"platform='{query['platform']}'")
            if query.get("instrument"):
                params.append(f"instrument='{query['instrument']}'")
            if query.get("temporal"):
                params.append(f"temporal='{query['temporal']}'")
            if query.get("bounding_box"):
                params.append(f"spatial='{query['bounding_box']}'")
            print(f"      {i}. {', '.join(params) if params else 'No parameters'}")

    if angle_result.collections:
        print("\n   📚 Selected Collections:")
        for i, collection in enumerate(angle_result.collections, 1):
            print(f"      {i}. {collection.get('dataset_id', 'Unknown ID')}")
            print(f"         Title: {collection.get('title', 'No title')[:80]}...")
            if collection.get("platform"):
                print(f"         Platform: {collection['platform']}")
            if collection.get("instrument"):
                print(f"         Instrument: {collection['instrument']}")

    if angle_result.granules:
        print("\n   📁 Sample Data Files (showing first 3):")
        for i, granule in enumerate(angle_result.granules[:3], 1):
            print(
                f"      {i}. {granule.get('producer_granule_id', granule.get('title', 'Unknown'))}",
            )
            if granule.get("time_start"):
                print(f"         Date: {granule['time_start']}")
            if granule.get("file_size_mb"):
                print(f"         Size: {granule['file_size_mb']} MB")


async def run_demo():
    """Run the complete demo workflow."""
    print("🚀 CMR Data Search - Angle-Based Pipeline Demo")
    print("=" * 60)

    # Configuration
    config = get_config()

    cmr_handler_config = CMRHandlerConfig(
        mcp_endpoint=config.mcp.endpoint,
        collection_search_page_size=20,
        granule_search_page_size=10,
        final_collection_count=5,  # Limit for demo
        collection_search_timeout=30.0,
        granule_search_timeout=45.0,
        min_collection_relevance_score=0.3,
    )

    agent_config = DataSearchAgentConfig(
        debug=True,  # Enable detailed logging
        enable_parallel_search=True,
        cmr=cmr_handler_config,
    )

    # Initialize agent
    print("🔧 Initializing CMR Data Search Agent...")
    print(f"   MCP Endpoint: {agent_config.cmr.mcp_endpoint}")
    print(f"   Max Collections: {agent_config.cmr.final_collection_count}")
    print(f"   Parallel Search: {agent_config.enable_parallel_search}")

    agent = DataSearchAgent(config=agent_config, debug=True)

    # Demo query
    demo_query = (
        "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print_step_header(1, "Input Query")
    print(f"📝 Scientific Question: {demo_query}")

    # Search parameters
    search_params = DataSearchAgentInputSchema(
        query=demo_query,
        temporal_range="2020-01-01,2024-12-31",  # Recent 4 years
        spatial_bounds="-95,28,-88,35",  # Lower Mississippi region (rough bounds)
        max_results=50,
    )

    print("⚙️  Search Parameters:")
    print(f"   • Temporal Range: {search_params.temporal_range}")
    print(f"   • Spatial Bounds: {search_params.spatial_bounds}")
    print(f"   • Max Results: {search_params.max_results}")

    # Execute search
    print_step_header(2, "Executing Search Pipeline")
    start_time = datetime.now()

    try:
        result = await agent._arun(search_params)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print_step_header(3, "Search Results Summary")
        print(f"⏱️  Search Duration: {duration:.1f} seconds")
        print("📊 Overall Results:")
        print(f"   • {len(result.angles)} scientific angles processed")
        print(f"   • {result.total_results} total data files found")
        print(f"   • {len(result.collections_searched)} total collections searched")

        # Show workflow metadata
        metadata = result.search_metadata
        print("\n🔍 Search Metadata:")
        print(f"   • Search ID: {metadata.get('search_id', 'Unknown')}")
        print(f"   • Workflow Version: {metadata.get('workflow_version', 'Unknown')}")
        print(f"   • Angles Processed: {metadata.get('angles_processed', 0)}")

        print_step_header(4, "Detailed Results by Scientific Angle")

        # Display each angle's results
        for i, angle_result in enumerate(result.angles, 1):
            print_angle_summary(angle_result, i)

        print_step_header(5, "Summary Statistics")

        # Calculate summary statistics
        total_queries = sum(len(ar.cmr_queries) for ar in result.angles)
        total_collections_found = sum(
            ar.total_collections_found for ar in result.angles
        )
        total_collections_selected = sum(len(ar.collections) for ar in result.angles)

        print("📈 Pipeline Statistics:")
        print(f"   • Scientific Angles: {len(result.angles)}")
        print(f"   • CMR Queries Generated: {total_queries}")
        print(f"   • Collections Found: {total_collections_found}")
        print(f"   • Collections Selected: {total_collections_selected}")
        print(f"   • Data Files Discovered: {result.total_results}")
        print(
            f"   • Deduplication Rate: {((total_collections_found - total_collections_selected) / max(total_collections_found, 1)) * 100:.1f}%",
        )

        print_step_header(6, "Sample Data Files")

        # Show sample granules from each angle
        for i, angle_result in enumerate(result.angles, 1):
            if angle_result.granules:
                angle_title = angle_result.scientific_angle.get("title", f"Angle {i}")
                print(f"\n📁 {angle_title} - Sample Files:")
                for j, granule in enumerate(
                    angle_result.granules[:2],
                    1,
                ):  # Show first 2
                    print(
                        f"   {j}. {granule.get('producer_granule_id', granule.get('title', 'Unknown'))}",
                    )
                    if granule.get("time_start"):
                        print(f"      📅 Date: {granule['time_start']}")
                    if granule.get("file_size_mb"):
                        print(f"      💾 Size: {granule['file_size_mb']} MB")
                    if granule.get("links"):
                        download_links = [
                            link
                            for link in granule["links"]
                            if link.get("rel")
                            == "http://esipfed.org/ns/fedsearch/1.1/data#"
                        ]
                        if download_links:
                            print(
                                f"      🔗 Download: {download_links[0].get('href', 'No URL')}",
                            )

        print("\n✅ Demo completed successfully!")
        print(
            "🎯 The angle-based pipeline provides scientific transparency by showing:",
        )
        print("   • How the query was interpreted (scientific angles)")
        print("   • What searches were performed (CMR queries)")
        print("   • What datasets were found and selected (collections)")
        print("   • What data files are available (granules)")

    except Exception as e:
        print(f"\n❌ Search failed with error: {e}")
        import traceback

        print("\n🔍 Full error traceback:")
        traceback.print_exc()


def main():
    """Main entry point for the demo."""
    print("🌍 NASA CMR Data Search - Angle-Based Pipeline Demo")
    print(
        "This demo shows how scientific queries are processed through multiple angles",
    )
    print("to provide organized, transparent access to NASA Earth science data.")
    print("\nPress Ctrl+C to interrupt the demo at any time.")

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
