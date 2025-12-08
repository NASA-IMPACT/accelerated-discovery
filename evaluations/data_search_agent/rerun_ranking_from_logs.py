"""
Rerun ranking/reranking on evaluation logs using reconstructed approach collections.

This script uses the handler's _rank_collections method directly, which routes to:
1. Legacy ranking: Per-approach filtering + final cross-approach ranking
2. LLM reranker: Direct reranking via adapter

Usage:
    # Run legacy ranking
    python3 evaluations/rerun_ranking_from_logs.py \
        --input evaluations/single_query_results/legacy_reranker_new_config/cpq_15_output \
        --ranking-type legacy \
        --output-dir evaluations/reranked_legacy/

    # Run LLM reranker
    python3 evaluations/rerun_ranking_from_logs.py \
        --input evaluations/single_query_results/legacy_reranker_new_config/cpq_15_output \
        --ranking-type llm \
        --output-dir evaluations/reranked_llm/
"""

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

# Add akd to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akd.agents.data_search.handlers.cmr.handler import CMRHandler
from akd.agents.data_search.handlers.cmr.config import CMRHandlerConfig
from akd.agents.data_search.handlers.cmr.schemas import Topic, ScientificDecomposition, CMRQueryApproach

# Import reconstruction utilities
from reconstruct_structure_for_reranker import (
    reconstruct_rank_collections_input,
    RankCollectionsInput,
)


async def run_ranking(
    rank_input: RankCollectionsInput,
    handler: CMRHandler,
) -> Dict[str, Any]:
    """
    Run ranking pipeline using handler's _rank_collections method.

    Args:
        rank_input: Reconstructed rank collections input
        handler: Configured CMR handler

    Returns:
        Dictionary with ranked results and metadata
    """
    # Parse topic and decomposition
    topic = Topic(**rank_input.topic)
    decomp = ScientificDecomposition(**rank_input.decomposition)

    # Parse query approaches
    query_approaches = []
    for approach_dict in rank_input.approaches_to_use:
        query_approaches.append(CMRQueryApproach(**approach_dict))

    ranking_type = "LLM reranker" if handler.config.use_llm_reranker else "legacy"
    print(f"Running {ranking_type} on {sum(len(c) for c in rank_input.approach_collections.values())} collections...")

    # Call handler's rank_collections method (routes internally)
    ranked_collections, ranking_metadata = await handler._rank_collections(
        approach_collections=rank_input.approach_collections,
        approach_collections_by_query=rank_input.approach_collections_by_query,
        original_query=rank_input.original_query,
        topic=topic,
        decomp=decomp,
        query_approaches=query_approaches,
        run_id=rank_input.run_id or "rerank_from_logs",
    )

    return {
        "ranked_collections": ranked_collections,
        "ranking_metadata": ranking_metadata,
        "ranking_type": ranking_type,
        "total_input": sum(len(c) for c in rank_input.approach_collections.values()),
        "total_output": len(ranked_collections),
    }


async def process_evaluation_file(
    input_file: Path,
    output_dir: Path,
    handler: CMRHandler,
) -> None:
    """
    Process a single evaluation file and rerank all decompositions.

    Args:
        input_file: Path to evaluation JSON
        output_dir: Directory to save reranked results
        handler: Configured CMR handler
    """
    ranking_type = "llm" if handler.config.use_llm_reranker else "legacy"
    output_file = output_dir / input_file.name
    if output_file.exists():
        print(f"✓ Skipping already processed file: {output_file}")
        return

    print(f"\n{'='*80}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*80}\n")

    # Load evaluation data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Get metadata
    original_query = data.get("search_metadata", {}).get("original_query", "")

    print(f"Original Query: {original_query}\n")


    for topic_data in data.get("topics", []):
        topic = topic_data.get("topic", {})
    
        for decomp_result in topic_data.get("decomposition_results", []):
            decomp_title = decomp_result.get("decomposition", {}).get("title", "")
            print(f"Processing decomposition: {decomp_title[:80]}...")

            # Reconstruct rank input
            rank_input = reconstruct_rank_collections_input(
                decomposition_result=decomp_result,
                original_query=original_query,
                topic=topic,
                run_id=f"rerank_{input_file.stem}",
            )

            # Run ranking using handler
            rank_result = await run_ranking(rank_input, handler)

            print(f"  Input: {rank_result['total_input']} collections")
            print(f"  Output: {rank_result['total_output']} ranked collections\n")

            # Add reranked results to existing decomposition result
            decomp_result["reranked_results"] = rank_result["ranked_collections"]
            decomp_result["reranking_metadata"] = rank_result["ranking_metadata"]

    # Save reranked results
    output_file = output_dir / input_file.name
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved reranked results to: {output_file}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Rerun ranking/reranking on evaluation logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing evaluation JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reranked results (default: creates 'reranked_{ranking_type}_results' in parent of input files)",
    )
    parser.add_argument(
        "--ranking-type",
        type=str,
        choices=["legacy", "llm"],
        required=True,
        help="Type of ranking to use (legacy: approach-aware filtering, llm: LLM reranker)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory")
        sys.exit(1)

    # Find all JSON files in the input directory
    input_files = list(args.input.glob("*.json"))
    if not input_files:
        print(f"Error: No JSON files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(input_files)} JSON files in {args.input}\n")

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = args.input / f"reranked_{args.ranking_type}_results_v2"
        print(f"No output directory specified. Using: {args.output_dir}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize handler based on ranking type
    if args.ranking_type == "legacy":
        handler_config = CMRHandlerConfig(
            use_llm_reranker=False,
            apply_final_collection_limit=True,
        )
        print(f"Using LEGACY ranking pipeline")
        print(f"  - Per-approach filtering + final cross-approach ranking")
        print(f"  - Final collection count: {handler_config.final_collection_count}")
    else:  # llm
        handler_config = CMRHandlerConfig(
            use_llm_reranker=True,
            apply_final_collection_limit=False,
            llm_reranker_model="gpt-4o-mini",
            llm_reranker_temperature=0.0,
        )
        print(f"Using LLM reranker")
        print(f"  - Model: {handler_config.llm_reranker_model}")

    # Initialize handler
    handler = CMRHandler(config=handler_config)

    print()

    # Process all input files
    for input_file in input_files:
        await process_evaluation_file(
            input_file=input_file,
            output_dir=args.output_dir,
            handler=handler,
        )

    print(f"\n{'='*80}")
    print(f"All files processed! Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
