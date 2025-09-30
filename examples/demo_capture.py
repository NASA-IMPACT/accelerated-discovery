"""
Capture Demo - Run complete multi-path workflow and save all intermediate results.

This demo runs the entire data search workflow with ALL topics and ALL decompositions,
capturing every intermediate result from each component. The captured data can then
be used with demo_loader.py to test individual components without re-running the
entire pipeline.

Usage:
    python demo_capture.py                          # Use default query
    python demo_capture.py --query "your query"     # Custom query
    python demo_capture.py --output results.json    # Custom output file
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Import timing collector
from timing_collector import TimingCollector

from akd.agents.data_search import CMRDataSearchAgent, CMRDataSearchAgentConfig
from akd.agents.data_search.components import ScientificDecomposition, Topic
from akd.agents.data_search.components.collection_ranking import (
    CollectionRankingInputSchema,
)
from akd.agents.data_search.components.repository_router import NASARepositoryEnum
from akd.configs.data_search_config import get_config
from akd.utils.serialization import safe_model_dump, safe_model_dump_list

# Load environment variables
load_dotenv()

print("✅ Imports successful")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🔑 OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

# Model configuration for each pipeline component
MODEL_CONFIG = {
    "topic_splitting": "gpt-5-mini",
    "scientific_decomposition": "gpt-5-mini",
    "repository_routing": "gpt-5-mini",
    "collection_ranking": "gpt-5-mini",
    "cmr_query": "gpt-5-mini",
}

print("🎛️ Model Configuration:")
for component, model in MODEL_CONFIG.items():
    print(f"   • {component.replace('_', ' ').title()}: {model}")

# Load base configuration
config = get_config()

# Configure agent with model-specific settings
agent_config = CMRDataSearchAgentConfig(
    debug=True,
    mcp_endpoint=config.mcp.endpoint,
    max_collections_to_search=5,
    collection_search_page_size=20,
    granule_search_page_size=10,
    enable_parallel_search=True,
    collection_search_timeout=30.0,
    granule_search_timeout=45.0,
    min_collection_relevance_score=0.3,
    # Model configurations for components
    topic_splitting_model=MODEL_CONFIG["topic_splitting"],
    scientific_decomposition_model=MODEL_CONFIG["scientific_decomposition"],
    repository_routing_model=MODEL_CONFIG["repository_routing"],
    collection_ranking_model=MODEL_CONFIG["collection_ranking"],
    cmr_query_model=MODEL_CONFIG["cmr_query"],
    # Legacy compatibility
    angle_generation_model=MODEL_CONFIG["topic_splitting"],
)

# Initialize the agent
agent = CMRDataSearchAgent(config=agent_config, debug=True)

print("\n🤖 Agent initialized for capture workflow")


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:50]  # Limit length


async def capture_single_path(
    query: str,
    topic: Topic,
    decomposition: ScientificDecomposition,
    timing_collector: TimingCollector,
    topic_idx: int,
    decomp_idx: int,
) -> Dict[str, Any]:
    """
    Capture complete pipeline results for a single topic+decomposition path.

    Args:
        query: Original research query
        topic: Topic to process
        decomposition: Scientific decomposition to process

    Returns:
        Dictionary with all component outputs for this path
    """
    print(f"      Processing path: {topic.title} → {decomposition.title}")

    # Set timing context for this path
    timing_collector.set_context(topic_idx=topic_idx, decomp_idx=decomp_idx)

    path_data = {}

    try:
        # Known Parameters
        print("        🔍 Known Parameters...")
        async with timing_collector.measure("known_parameters") as timer:
            known_params_output = await agent.known_parameters_component.process(
                query,
                topic,
                decomposition,
            )
            timer.add_metadata(
                query_approaches_generated=len(known_params_output.query_approaches),
            )

        path_data["known_params"] = {
            "reasoning": known_params_output.reasoning,
            "query_approaches": safe_model_dump_list(
                known_params_output.query_approaches,
            ),
        }
        print(
            f"          ✅ Generated {len(known_params_output.query_approaches)} query approaches",
        )

        # Searchable Parameters
        print("        🔍 Searchable Parameters...")
        async with timing_collector.measure("searchable_parameters") as timer:
            searchable_output = await agent.searchable_parameters_component.process(
                query,
                topic,
                decomposition,
                known_params_output.query_approaches,
            )
            timer.add_metadata(
                searchable_queries_generated=len(searchable_output.searchable_queries),
                keyword_strategy=searchable_output.keyword_strategy,
            )

        path_data["searchable_params"] = {
            "keyword_strategy": searchable_output.keyword_strategy,
            "searchable_queries": safe_model_dump_list(
                searchable_output.searchable_queries,
            ),
        }
        print(
            f"          ✅ Generated {len(searchable_output.searchable_queries)} searchable queries",
        )

        # Collection Search
        print("        🔍 Collection Search...")
        async with timing_collector.measure("collection_search") as timer:
            collections = []
            successful_queries = 0
            failed_queries = 0

            for i, query_obj in enumerate(searchable_output.searchable_queries):
                try:
                    search_params = query_obj.get_mcp_parameters()
                    search_params["page_size"] = (
                        agent.config.collection_search_page_size
                    )

                    tool_input = agent.collection_search_tool.input_schema(
                        **search_params,
                    )
                    result = await agent.collection_search_tool.arun(tool_input)

                    if hasattr(result, "collections") and result.collections:
                        collections.extend(result.collections)
                        successful_queries += 1
                    else:
                        failed_queries += 1

                except Exception as e:
                    print(f"          ⚠️ Query {i + 1} failed: {e}")
                    failed_queries += 1

            timer.add_metadata(
                queries_executed=len(searchable_output.searchable_queries),
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                collections_found=len(collections),
            )

        path_data["collections_raw"] = collections
        print(f"          ✅ Found {len(collections)} total collections")

        # Collection Ranking (if needed)
        ranked_collections = collections
        if len(collections) > agent.config.max_collections_to_search:
            print("        🔍 Collection Ranking...")

            async with timing_collector.measure("collection_ranking") as timer:
                # Extract approach fields for ranking
                approach_instruments = []
                approach_platforms = []
                approach_keywords = []

                for approach in known_params_output.query_approaches:
                    if hasattr(approach, "instrument") and approach.instrument:
                        approach_instruments.append(approach.instrument)
                    if hasattr(approach, "platform") and approach.platform:
                        approach_platforms.append(approach.platform)

                for query_obj in searchable_output.searchable_queries:
                    if (
                        hasattr(query_obj, "primary_keywords")
                        and query_obj.primary_keywords
                    ):
                        approach_keywords.extend(query_obj.primary_keywords)

                # Remove duplicates
                approach_instruments = list(dict.fromkeys(approach_instruments))
                approach_platforms = list(dict.fromkeys(approach_platforms))
                approach_keywords = list(dict.fromkeys(approach_keywords))

                ranking_input = CollectionRankingInputSchema(
                    original_query=query,
                    topic_title=topic.title,
                    topic_context=topic.functional_context,
                    decomposition_title=decomposition.title,
                    decomposition_justification=decomposition.scientific_justification,
                    approach_instruments=approach_instruments,
                    approach_platforms=approach_platforms,
                    approach_keywords=approach_keywords,
                    collections=collections,
                    max_collections=agent.config.max_collections_to_search,
                )

                ranking_result = await agent.collection_ranking_component.arun(
                    ranking_input,
                )

                timer.add_metadata(
                    collections_to_rank=len(collections),
                    collections_after_ranking=len(ranking_result.ranked_collections),
                    ranking_enabled=True,
                )
            ranked_collections = [
                collections[rc.collection_index]
                for rc in ranking_result.ranked_collections
                if 0 <= rc.collection_index < len(collections)
            ]
            print(f"          ✅ Ranked to {len(ranked_collections)} top collections")

        path_data["collections_ranked"] = ranked_collections

        # Granule Search
        print("        🔍 Granule Search...")
        async with timing_collector.measure("granule_search") as timer:
            all_granules = []
            successful_collections = 0
            failed_collections = 0

            for collection in ranked_collections:
                concept_id = collection.get("concept_id")
                if not concept_id:
                    continue

                try:
                    granule_params = {
                        "collection_concept_id": concept_id,
                        "page_size": agent.config.granule_search_page_size,
                    }

                    granule_search_params = agent.granule_search_tool.input_schema(
                        **granule_params,
                    )
                    result = await agent.granule_search_tool.arun(granule_search_params)

                    if hasattr(result, "results") and result.results.get("granules"):
                        granules = result.results["granules"]
                        all_granules.extend(granules)
                        successful_collections += 1
                    else:
                        failed_collections += 1

                except Exception as e:
                    print(f"          ⚠️ Granule search failed for {concept_id}: {e}")
                    failed_collections += 1

            timer.add_metadata(
                collections_searched=len(ranked_collections),
                successful_collections=successful_collections,
                failed_collections=failed_collections,
                granules_found=len(all_granules),
            )

        path_data["granules"] = all_granules
        print(f"          ✅ Found {len(all_granules)} granules")

    except Exception as e:
        print(f"        ❌ Path processing failed: {e}")
        path_data["error"] = str(e)

    return path_data


async def capture_full_workflow(query: str, output_file: str = None) -> str:
    """
    Run complete multi-path workflow and capture all intermediate results.

    Args:
        query: Research query to process
        output_file: Optional custom output filename

    Returns:
        Path to the saved JSON file
    """
    print("\n🚀 CAPTURING FULL MULTI-PATH WORKFLOW")
    print(f"🔍 Query: '{query}'")

    start_time = datetime.now()

    # Initialize timing collector
    timing_collector = TimingCollector(
        search_id=f"capture_{slugify(query)}_{int(start_time.timestamp())}",
    )

    # Step 1: Topic Splitting
    print("\n1️⃣ Topic Splitting:")
    async with timing_collector.measure("topic_splitting") as timer:
        topics_output = await agent.topic_splitting_component.process(query)
        timer.add_metadata(topics_identified=len(topics_output.topics))

    print(f"   ✅ Identified {len(topics_output.topics)} topics:")
    for i, topic in enumerate(topics_output.topics, 1):
        print(f"   {i}. {topic.title}")

    # Step 2: Process each topic
    all_topic_results = []
    total_paths = 0

    for topic_idx, topic in enumerate(topics_output.topics):
        print(f"\n2️⃣ Processing Topic {topic_idx + 1}: {topic.title}")

        # Set timing context for this topic
        timing_collector.set_context(topic_idx=topic_idx)

        # Repository Routing
        print("   🔍 Repository Routing...")
        async with timing_collector.measure("repository_routing") as timer:
            routing_output = await agent.repository_router_component.process(
                query,
                topic,
            )
            timer.add_metadata(
                repositories_identified=len(routing_output.route.repositories),
                repositories=routing_output.route.repositories,
            )
        print(f"     ✅ Repositories: {routing_output.route.repositories}")

        # Check if CMR is selected
        has_cmr = NASARepositoryEnum.CMR in routing_output.route.repositories
        if not has_cmr:
            print("     ⚠️ CMR not selected, skipping decomposition")
            all_topic_results.append(
                {
                    "index": topic_idx,
                    "topic": safe_model_dump(topic),
                    "routing": safe_model_dump(routing_output.route),
                    "decompositions": [],
                    "note": f"Routed to {routing_output.route.repositories}; CMR not selected",
                },
            )
            continue

        # Scientific Decomposition
        print("   🔍 Scientific Decomposition...")
        async with timing_collector.measure("scientific_decomposition") as timer:
            decomp_output = await agent.scientific_decomposition_component.process(
                query,
                topic,
            )
            timer.add_metadata(
                decompositions_generated=len(decomp_output.decompositions),
            )
        print(f"     ✅ Generated {len(decomp_output.decompositions)} decompositions:")
        for i, decomp in enumerate(decomp_output.decompositions, 1):
            print(f"     {i}. {decomp.title}")

        # Process each decomposition in parallel for better performance
        print(
            f"\n   3️⃣ Processing {len(decomp_output.decompositions)} decompositions in parallel...",
        )

        # Create tasks for parallel execution
        decomp_tasks = []
        for decomp_idx, decomposition in enumerate(decomp_output.decompositions):
            print(
                f"      Queuing decomposition {decomp_idx + 1}: {decomposition.title}",
            )
            task = capture_single_path(
                query,
                topic,
                decomposition,
                timing_collector,
                topic_idx,
                decomp_idx,
            )
            decomp_tasks.append((decomp_idx, decomposition, task))

        # Execute all decompositions in parallel
        print(
            f"      🚀 Executing {len(decomp_tasks)} decomposition tasks in parallel...",
        )
        task_results = await asyncio.gather(
            *[task for _, _, task in decomp_tasks],
            return_exceptions=True,
        )

        # Process results and handle any exceptions
        decomposition_results = []
        for i, ((decomp_idx, decomposition, _), result) in enumerate(
            zip(decomp_tasks, task_results),
        ):
            if isinstance(result, Exception):
                print(f"      ❌ Decomposition {decomp_idx + 1} failed: {result}")
                # Create error entry
                decomposition_results.append(
                    {
                        "index": decomp_idx,
                        "decomposition": safe_model_dump(decomposition),
                        "error": str(result),
                        "known_params": {},
                        "searchable_params": {},
                        "collections_raw": [],
                        "collections_ranked": [],
                        "granules": [],
                    },
                )
            else:
                print(f"      ✅ Decomposition {decomp_idx + 1} completed successfully")
                decomposition_results.append(
                    {
                        "index": decomp_idx,
                        "decomposition": safe_model_dump(decomposition),
                        **result,
                    },
                )
            total_paths += 1

        all_topic_results.append(
            {
                "index": topic_idx,
                "topic": safe_model_dump(topic),
                "routing": safe_model_dump(routing_output.route),
                "decompositions": decomposition_results,
            },
        )

    # Calculate totals
    total_collections = sum(
        len(decomp.get("collections_ranked", []))
        for topic_result in all_topic_results
        for decomp in topic_result["decompositions"]
    )

    total_granules = sum(
        len(decomp.get("granules", []))
        for topic_result in all_topic_results
        for decomp in topic_result["decompositions"]
    )

    duration = (datetime.now() - start_time).total_seconds()

    # Finalize timing collection
    timing_data = timing_collector.finalize()

    # Build complete output
    output_data = {
        "query": query,
        "timestamp": start_time.isoformat(),
        "topics": all_topic_results,
        "timing_data": timing_data,
        "metadata": {
            "total_topics": len(topics_output.topics),
            "total_decompositions": sum(
                len(tr["decompositions"]) for tr in all_topic_results
            ),
            "total_paths": total_paths,
            "total_collections": total_collections,
            "total_granules": total_granules,
            "duration_seconds": duration,
            "agent_config": {
                "max_collections_to_search": agent.config.max_collections_to_search,
                "collection_search_page_size": agent.config.collection_search_page_size,
                "granule_search_page_size": agent.config.granule_search_page_size,
            },
            "model_config": MODEL_CONFIG,
        },
    }

    # Save to file
    if not output_file:
        query_slug = slugify(query)
        timestamp_slug = start_time.strftime("%Y%m%d_%H%M%S")
        output_file = f"captured_data/captured_{query_slug}_{timestamp_slug}.json"

    # Ensure captured_data directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print("\n📊 CAPTURE COMPLETE!")
    print(f"   Total Topics: {len(topics_output.topics)}")
    print(f"   Total Paths: {total_paths}")
    print(f"   Total Collections: {total_collections}")
    print(f"   Total Granules: {total_granules}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   💾 Saved to: {output_file}")

    # Print timing summary
    timing_collector.print_summary()

    return output_file


async def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Capture complete multi-path data search workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_capture.py                                    # Use default query
  python demo_capture.py --query "MODIS temperature data"  # Custom query
  python demo_capture.py --output my_results.json          # Custom output file
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        default="Help me gather data to study the flood risk of the lower Mississippi basin",
        help="Research query to process",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON filename (default: auto-generated)",
    )

    args = parser.parse_args()

    try:
        output_file = await capture_full_workflow(args.query, args.output)
        print("\n✅ Workflow capture completed successfully!")
        print(
            f"🎯 Use 'python demo_loader.py {output_file}' to test individual components",
        )

    except Exception as e:
        print(f"\n❌ Capture failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
