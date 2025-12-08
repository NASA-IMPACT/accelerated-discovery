#!/usr/bin/env python3
"""
Run data search agent on a single query.

Usage:
    python evaluations/run_single_query.py "Your query here"
    python evaluations/run_single_query.py "Your query here" --model gpt-4o
    python evaluations/run_single_query.py "Your query here" --output-dir my_results/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

from akd.agents.data_search import DataSearchAgent, DataSearchAgentConfig, DataSearchAgentInputSchema
from akd.agents.data_search.handlers import CMRHandlerConfig


async def run_single_query(
    query: str,
    output_path : str | Path,
    agent : DataSearchAgent,
):
    """Run data search on a single query."""

    # Create output directory
   

    # Initialize agent

    # for llm reranker you have apply_final_Collection_limit = False 

    print(f"\n{'='*80}")
    print(f"Running Data Search Agent")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Output path: {output_path}")
    print(f"{'='*80}\n")

    try:
        # Run the agent
        input_params = DataSearchAgentInputSchema(query=query)
        result = await agent.arun(input_params)

        # Extract metadata
        search_metadata = result.search_metadata

        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"✓ Search ID: {search_metadata.get('search_id')}")
        print(f"✓ Topics Processed: {search_metadata.get('topics_processed', 0)}")
        print(f"✓ Total CMR Results: {result.total_cmr_results}")
        print(f"✓ Total Filtered Results: {result.total_filtered_results}")
        print(f"✓ Duration: {search_metadata.get('duration_seconds', 0):.1f}s")

        # Print summary if available
        if hasattr(result, 'summary') and result.summary:
            summary = result.summary
            print(f"\n{'='*80}")
            print(f"SUMMARY")
            print(f"{'='*80}")
            print(f"Total Topics: {summary.total_topics}")

            for topic in summary.topics:
                print(f"\n {topic.title}")
                print(f"   Decompositions: {topic.total_decompositions}")

                for decomp in topic.decompositions:
                    print(f"\n   → {decomp.title}")
                    print(f"      Total Collections: {decomp.total_collections}")
                    if decomp.collections:
                        print(f"      Top {len(decomp.collections)} Collections:")
                        for concept_id, title in decomp.collections:
                            print(f"        - {concept_id}: {title[:60]}...")

        
        
        # save result to output directory as json
        output_file = output_path / f"query_results_{query[:50].replace(' ', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(result.model_dump(), f, indent=4)

        print(f"\n{'='*80}")
        print(f"✓ Results saved to: {output_file}")
        print(f"{'='*80}\n")

        return result

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(query: str):
    parser = argparse.ArgumentParser(
        description="Run data search agent on a single query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # parser.add_argument(
    #     "query",
    #     help="The query to run",
    # )

    parser.add_argument(
        "--output-dir",
        default="evaluations/single_query_results",
        help="Output directory for results (default: evaluations/single_query_results)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--cmr-endpoint",
        default="http://localhost:8080/mcp/cmr/mcp",
        help="CMR MCP endpoint URL",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug output",
    )

    args = parser.parse_args()

    cmr_config = CMRHandlerConfig(
        use_llm_reranker=True, 
        apply_final_collection_limit=False,
        llm_reranker_model="gpt-4o-mini",
        llm_reranker_temperature=0.0,
        mcp_endpoint=args.cmr_endpoint, 
    )
    

    config = DataSearchAgentConfig(
        cmr=cmr_config,
        topic_splitting_model="gpt-4o-mini",
        scientific_decomposition_model="gpt-5-mini", 
        repository_routing_model="gpt-4o-mini",
        auto_save=True,
        output_subdir=str(output_path),
        debug=True,
        temperature =0.0,
    )

    agent = DataSearchAgent(config=config, debug=True)

    asyncio.run(run_single_query(
        query=query,
        output_dir=args.output_dir,
        agent=agent,
    ))


if __name__ == "__main__":
    # run for all queries from json files in evaluations/evaluate_run_files/run_20251106_173040
    # get agent_output.search_metadata.original_query


    # files = Path("evaluations/evaluate_run_files/run_20251106_173040").glob("*.json")
    # for file in files:
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         original_query = data.get("agent_output", {}).get("search_metadata", {}).get("original_query")
    #         if original_query:
    #             print(f"\nRunning query from file {file.name}: {original_query}\n")
    #             asyncio.run(run_single_query(
    #                 query=original_query,
    #                 output_dir=f"evaluations/single_query_results/{file.stem}",
    #                 model="gpt-5-mini",
    #                 cmr_endpoint="http://localhost:8080/mcp/cmr/mcp",
    #                 debug=True,
    #             ))
    #         else:
    #             print(f"No original query found in file {file.name}")
    truth_set_path = "evaluations/create_notebook/truth_set_20251027_deduplicated.json"

    with open(truth_set_path, "r") as f:
        truth_set = json.load(f)

    queries = truth_set.get("queries", [])

    print(f"Total queries to run: {len(queries)}")


    # # query = "What datasets are available on sea surface temperature anomalies in the Pacific Ocean over the last 20 years?"

    queries = [query for query in queries if query.get("sme") and query.get("sme").lower() == "emily"]

    # LLM reranker
    # output_path = Path("evaluations/single_query_results")/"llm_reranker_fresh_run_with_decomp/version2-again"
    # output_path.mkdir(parents=True, exist_ok=True)

    # # save config of the cmr handler used
    # cmr_config = CMRHandlerConfig(
    #     use_llm_reranker=True, 
    #     apply_final_collection_limit=False,
    #     llm_reranker_model="gpt-4o-mini",
    #     llm_reranker_temperature=0.0,
    #     mcp_endpoint="http://localhost:8080/mcp/cmr/mcp",
    # )
    

    # Legacy reranker 

    output_path = Path("evaluations/single_query_results")/"legacy_reranker_fresh_run/again"
    output_path.mkdir(parents=True, exist_ok=True)

    # save config of the cmr handler used
    cmr_config = CMRHandlerConfig(
        use_llm_reranker=False, 
        apply_final_collection_limit=False, #could be true, i am trying to build equivalency
        # llm_reranker_model="gpt-4o-mini",
        llm_reranker_temperature=0.0,
        mcp_endpoint="http://localhost:8080/mcp/cmr/mcp",
    )
    

    config = DataSearchAgentConfig(
        cmr=cmr_config,
        topic_splitting_model="gpt-4o-mini",
        scientific_decomposition_model="gpt-5-mini", 
        repository_routing_model="gpt-4o-mini",
        auto_save=True,
        output_subdir=str(output_path),
        debug=True,
        temperature =0.0,
    )


    # write a json file with the cmr config used
    with open(output_path / "cmr_handler_config_used.json", "w") as f:
        json.dump(cmr_config.model_dump(), f, indent=4)

    




        
        


    agent = DataSearchAgent(config=config, debug=True)

    async def run_all_queries():
        for query_info in queries:
            query = query_info.get("query_text")
            if query:  # Fixed the method call to lower()
                output_file = output_path / f"query_results_{query[:50].replace(' ', '_')}.json"
                if output_file.exists():
                    print(f"✓ Results already exist at: {output_file}, skipping...")
                    continue

                query = query_info.get("query_text")
                print(f"\nRunning query: {query}\n")
                await run_single_query(
                            query=query,
                            output_path=output_path,
                            agent=agent,
                        )
                # break

    asyncio.run(run_all_queries()) 

            



    # for i, query_info in enumerate(queries):
    #     query = query_info.get("query_text")
    #     sme =  query_info.get("sme")
    #     if query:  # Fixed the method call to lower()
    #         print(f"\nRunning query {i+1}/{len(queries)}: {query}\n")
    #         asyncio.run(run_single_query(
    #                 query=query,
    #                 output_dir=f"evaluations/single_query_results/",
    #                 model="gpt-5-mini",
    #                 cmr_endpoint="http://localhost:8080/mcp/cmr/mcp",
    #                 debug=True,
    #             ))


    # query = "How can we assess agricultural drought severity and its impact on crop productivity in the California United States?"
    # asyncio.run(run_single_query(
    #                 query=query,
    #                 output_dir=f"evaluations/single_query_results/",
    #                 model="gpt-5-mini",
    #                 cmr_endpoint="http://localhost:8080/mcp/cmr/mcp",
    #                 debug=True,
    #             ))

    # asyncio.run(main(query=query))



# Usage Example 1: Default Config (uses built-in CMR criteria)
# from akd.agents.data_search.handlers.cmr import CMRHandler, CMRHandlerConfig

# config = CMRHandlerConfig(
#     use_llm_reranker=True,
#     llm_reranker_model="gpt-4o-mini",  # or "gpt-4o"
#     llm_reranker_temperature=0.0,
# )

# handler = CMRHandler(config=config, debug=True)
# Usage Example 2: Custom Config (custom criteria, fields, weights)
# from akd.agents.data_search.handlers.cmr import CMRHandler, CMRHandlerConfig
# from akd.tools.reranker import LLMRerankerToolConfig, ScoringCriterion, ScoringCategory

# # Create custom reranker config with your own criteria
# custom_reranker_config = LLMRerankerToolConfig(
#     model_name="gpt-4o",
#     temperature=0.0,
#     fields_to_evaluate={
#         "title": "Dataset title",
#         "content": "Dataset description",
#         "processing_level": "Data processing level",
#         # Add your custom fields
#     },
#     scoring_criteria=[
#         ScoringCriterion(
#             name="My Custom Criterion",
#             description="Your custom scoring logic",
#             weight=0.5,
#             scoring_categories=[
#                 ScoringCategory(name="Perfect", description="...", value=3.0),
#                 ScoringCategory(name="Good", description="...", value=2.0),
#                 ScoringCategory(name="Poor", description="...", value=1.0),
#             ],
#         ),
#         # Add more criteria...
#     ],
# )

# # Pass custom config to CMRHandlerConfig
# config = CMRHandlerConfig(
#     use_llm_reranker=True,
#     custom_llm_reranker_config=custom_reranker_config,
# )

# handler = CMRHandler(config=config, debug=True)
# The handler will:
# Use custom_llm_reranker_config if provided (ignores llm_reranker_model and llm_reranker_temperature)
# Otherwise, use default CMR criteria with llm_reranker_model and llm_reranker_temperature