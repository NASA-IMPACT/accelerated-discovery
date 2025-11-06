#!/usr/bin/env python3
"""
Re-execute all CMR queries from evaluation runs with full pagination.

This script reads evaluation output files, extracts all mcp_parameters_sent entries,
re-executes those queries against CMR with complete pagination, and saves all
concept IDs returned.

Usage:
    uv run python evaluations/reexecute_cmr_queries.py evaluations/run_20251027_213136
    uv run python evaluations/reexecute_cmr_queries.py evaluations/run_20251027_213136/evaluation_runs.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from akd.tools.data_search import CMRCollectionSearchTool


def extract_mcp_parameters(eval_data: dict) -> List[Tuple[Dict[str, int], dict]]:
    """
    Extract all mcp_parameters_sent from evaluation data with index information.

    Args:
        eval_data: Loaded evaluation JSON data

    Returns:
        List of (index_info, mcp_parameters) tuples where index_info contains
        topic_idx, decomp_idx, query_idx for error logging
    """
    results = []

    agent_output = eval_data.get("agent_output", {})
    topics = agent_output.get("topics", [])

    for topic_idx, topic in enumerate(topics):
        decomp_results = topic.get("decomposition_results", [])

        for decomp_idx, decomp in enumerate(decomp_results):
            searchable_queries = decomp.get("searchable_queries", [])

            for query_idx, query in enumerate(searchable_queries):
                if "mcp_parameters_sent" in query:
                    index_info = {
                        "topic_idx": topic_idx,
                        "decomp_idx": decomp_idx,
                        "query_idx": query_idx,
                    }
                    results.append((index_info, query["mcp_parameters_sent"]))

    return results


async def reexecute_query(
    tool: CMRCollectionSearchTool,
    params: dict,
) -> Tuple[List[str], Optional[str]]:
    """
    Re-execute a single CMR query with full pagination.

    Args:
        tool: Initialized CMRCollectionSearchTool instance
        params: MCP parameters from mcp_parameters_sent

    Returns:
        Tuple of (concept_ids, error_message)
        - concept_ids: List of concept IDs (empty on error)
        - error_message: None on success, error string on failure
    """
    try:
        all_collections = []
        page_num = 1

        while True:
            # Prepare search parameters for this page
            search_params = params.copy()
            search_params["page_size"] = 50  # MCP server maximum
            search_params["page_num"] = page_num

            # Fix temporal format bug: replace :59:59:59Z with :59:59Z
            if "temporal" in search_params and search_params["temporal"]:
                search_params["temporal"] = search_params["temporal"].replace(
                    ":59:59:59Z",
                    ":59:59Z",
                )

            # Create input schema and execute
            tool_input = tool.input_schema(**search_params)
            result = await tool.arun(tool_input)

            # Extract collections from this page
            if hasattr(result, "collections") and result.collections:
                all_collections.extend(result.collections)

            # Check if we're done
            total_hits = result.total_hits if hasattr(result, "total_hits") else 0

            if len(all_collections) >= total_hits or not result.collections:
                break

            # Polite delay between pages to avoid rate limiting
            await asyncio.sleep(0.1)
            page_num += 1

        # Extract concept IDs
        concept_ids = [
            c.get("concept_id") for c in all_collections if c.get("concept_id")
        ]

        return concept_ids, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return [], error_msg


async def process_evaluation_file(
    file_path: Path,
    output_path: Path,
    tool: CMRCollectionSearchTool,
) -> Dict[str, Any]:
    """
    Process a single evaluation file.

    Args:
        file_path: Path to evaluation JSON file
        output_path: Path to save concept IDs
        tool: Initialized CMRCollectionSearchTool

    Returns:
        Statistics dict with counts of queries processed, succeeded, failed
    """
    print(f"Processing: {file_path.name}")

    # Load evaluation data
    try:
        with open(file_path) as f:
            eval_data = json.load(f)
    except Exception as e:
        print(f"  ERROR: Failed to load file: {e}", file=sys.stderr)
        return {"processed": 0, "succeeded": 0, "failed": 0}

    # Extract all mcp_parameters_sent entries
    queries = extract_mcp_parameters(eval_data)
    print(f"  Found {len(queries)} queries to re-execute")

    # Re-execute all queries in parallel
    async def execute_single(index_info, mcp_params):
        """Execute single query and return result tuple."""
        concept_ids, error = await reexecute_query(tool, mcp_params)
        return (index_info, mcp_params, concept_ids, error)

    # Create tasks for all queries
    tasks = [
        execute_single(index_info, mcp_params) for index_info, mcp_params in queries
    ]

    # Execute all queries in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    all_concept_ids = []
    error_log = []

    for result in results:
        if isinstance(result, Exception):
            # Task itself failed
            error_log.append(
                {
                    "error": f"Task failed: {type(result).__name__}: {str(result)}",
                },
            )
        else:
            index_info, mcp_params, concept_ids, error = result
            if error:
                error_log.append(
                    {
                        **index_info,
                        "error": error,
                        "mcp_parameters": mcp_params,
                    },
                )
            else:
                all_concept_ids.extend(concept_ids)

    # Deduplicate concept IDs while preserving order
    seen = set()
    unique_concept_ids = []
    for cid in all_concept_ids:
        if cid not in seen:
            seen.add(cid)
            unique_concept_ids.append(cid)

    # Save results
    output_data = {
        "data": unique_concept_ids,
        "log": error_log,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    succeeded = len(queries) - len(error_log)
    print(
        f"  Completed: {succeeded}/{len(queries)} queries succeeded, "
        f"{len(unique_concept_ids)} unique concept IDs",
    )

    if error_log:
        print(f"  WARNING: {len(error_log)} queries failed (see log in output file)")

    return {
        "processed": len(queries),
        "succeeded": succeeded,
        "failed": len(error_log),
    }


async def main(run_dir: Path):
    """
    Main orchestrator: process all evaluation files in a run directory.

    Args:
        run_dir: Path to evaluation run directory
    """
    # Find evaluation_runs.json
    eval_runs_path = run_dir / "evaluation_runs.json"

    if not eval_runs_path.exists():
        print(
            f"ERROR: evaluation_runs.json not found at {eval_runs_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load evaluation runs metadata
    with open(eval_runs_path) as f:
        eval_runs = json.load(f)

    results = eval_runs.get("results", [])
    print(f"Found {len(results)} evaluation files to process\n")

    # Initialize CMR tool with MCP endpoint (same as original evaluation runs)
    tool = CMRCollectionSearchTool.from_params(
        page_size=50,
        debug=False,
        mcp_endpoint="http://localhost:8080/mcp/cmr/mcp/",
    )

    # Create output directory
    output_dir = run_dir / "reexecuted"
    output_dir.mkdir(exist_ok=True)

    # Prepare file processing tasks
    async def process_single_file(result):
        """Process a single evaluation file."""
        output_file = result.get("output_file")
        if not output_file:
            return None

        # Convert relative path to absolute
        eval_file_path = Path(output_file)
        if not eval_file_path.is_absolute():
            eval_file_path = Path.cwd() / eval_file_path

        if not eval_file_path.exists():
            print(f"WARNING: File not found: {eval_file_path}", file=sys.stderr)
            return None

        # Determine output path
        search_id = result.get("search_id", eval_file_path.stem)
        output_path = output_dir / f"{search_id}_concepts.json"

        # Process file
        stats = await process_evaluation_file(eval_file_path, output_path, tool)
        return stats

    # Process all evaluation files in parallel with polite delays per query
    file_tasks = [process_single_file(result) for result in results]
    all_stats = await asyncio.gather(*file_tasks, return_exceptions=True)

    # Calculate totals
    total_stats = {"processed": 0, "succeeded": 0, "failed": 0}
    files_processed = 0

    for stats in all_stats:
        if isinstance(stats, Exception):
            print(f"ERROR: File processing failed: {stats}", file=sys.stderr)
        elif stats is not None:
            total_stats["processed"] += stats["processed"]
            total_stats["succeeded"] += stats["succeeded"]
            total_stats["failed"] += stats["failed"]
            files_processed += 1

    # Print summary
    print("=" * 60)
    print(f"SUMMARY: Processed {files_processed}/{len(results)} files")
    print(f"  Total queries: {total_stats['processed']}")
    print(f"  Succeeded: {total_stats['succeeded']}")
    print(f"  Failed: {total_stats['failed']}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-execute all CMR queries from evaluation runs with full pagination",
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="Path to run directory or evaluation_runs.json file",
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run_path.is_file() and args.run_path.name == "evaluation_runs.json":
        run_dir = args.run_path.parent
    elif args.run_path.is_dir():
        run_dir = args.run_path
    else:
        print(f"ERROR: Invalid path: {args.run_path}", file=sys.stderr)
        print(
            "Expected: directory containing evaluation_runs.json or path to evaluation_runs.json",
        )
        sys.exit(1)

    # Run async main
    asyncio.run(main(run_dir))
