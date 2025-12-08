"""
Script to add 'collections_to_ranking_pipeline' and 'input_to_reranker' fields to evaluation JSON files.

These fields represent the collections that would be sent to the ranking pipeline
based on a configurable collections_per_query parameter.

The script:
1. Reads the evaluation JSON file
2. For each searchable_query, determines which collections from all_collections_from_cmr
   would be sent to ranking based on the query's MCP parameters
3. Adds a 'collections_to_ranking_pipeline' field to each query (per-query subset)
4. Adds an 'input_to_reranker' field to each decomposition (flat list of all collections)
5. Saves the modified JSON to a new file

Fields added:
- collections_to_ranking_pipeline: Per-query list (added to each searchable_query)
- input_to_reranker: Flat list at decomposition level (like all_collections_from_cmr but filtered)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def match_collections_to_query(
    all_collections: List[Dict[str, Any]],
    query_mcp_params: Dict[str, Any],
    collections_returned: int,
) -> List[Dict[str, Any]]:
    """
    Match collections from all_collections_from_cmr to a specific query.

    Since all_collections_from_cmr is the concatenation of results from all queries,
    we need to identify which collections came from which query. The order is preserved
    from the query execution order.

    Args:
        all_collections: The full list of collections from all_collections_from_cmr
        query_mcp_params: The MCP parameters sent for this query
        collections_returned: Number of collections returned by this query (from cmr_collections_returned)

    Returns:
        List of collections that match this query (in order)
    """
    # Since collections are concatenated in query execution order,
    # we track position across all queries
    # This is handled by the caller who maintains a running index
    return []


def add_collections_to_ranking_pipeline(
    input_file: Path,
    output_file: Path,
    collections_per_query: int | None,
) -> None:
    """
    Add collections_to_ranking_pipeline field to evaluation JSON.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        collections_per_query: Number of collections to select per query for ranking (None = all)
    """
    print(f"Loading JSON from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_topics = len(data.get('topics', []))
    print(f"Processing {total_topics} topics...")

    # Process each topic
    for topic_idx, topic in enumerate(data.get('topics', [])):
        print(f"\nTopic {topic_idx + 1}/{total_topics}: {topic['topic']['title']}")

        # Process each decomposition result
        for decomp_idx, decomp_result in enumerate(topic.get('decomposition_results', [])):
            decomp_title = decomp_result['decomposition']['title']
            print(f"  Decomposition {decomp_idx + 1}: {decomp_title[:60]}...")

            all_collections = decomp_result.get('all_collections_from_cmr', [])
            searchable_queries = decomp_result.get('searchable_queries', [])

            print(f"    Total collections in all_collections_from_cmr: {len(all_collections)}")
            print(f"    Total searchable queries: {len(searchable_queries)}")

            # Track position in all_collections_from_cmr as we iterate through queries
            collection_offset = 0

            # Accumulate all collections for ranking (flat list)
            input_to_reranker = []

            # Process each searchable query
            for query_idx, query in enumerate(searchable_queries):
                collections_returned = query.get('cmr_collections_returned', 0)

                # Get the collections for this query from all_collections_from_cmr
                # They are in order, so we slice based on offset and collections_returned
                query_collections = all_collections[
                    collection_offset : collection_offset + collections_returned
                ]

                # Apply collections_per_query limit (mimics handler.py:449)
                # If None, take all collections (k=all)
                collections_for_ranking = query_collections if collections_per_query is None else query_collections[:collections_per_query]

                # Add the new field to the query
                query['collections_to_ranking_pipeline'] = collections_for_ranking

                # Accumulate for flat list (mimics handler.py:540)
                input_to_reranker.extend(collections_for_ranking)

                print(
                    f"      Query {query_idx + 1}: {collections_returned} total, "
                    f"{len(collections_for_ranking)} to ranking (offset: {collection_offset})"
                )

                # Move offset forward
                collection_offset += collections_returned

            # Add input_to_reranker as flat list at decomposition level
            decomp_result['input_to_reranker'] = input_to_reranker

            print(f"    Total input_to_reranker: {len(input_to_reranker)} collections")

            # Verify we consumed all collections
            if collection_offset != len(all_collections):
                print(
                    f"    WARNING: Collection count mismatch! "
                    f"Consumed {collection_offset} but have {len(all_collections)} total"
                )

    # Save modified JSON
    print(f"\nSaving modified JSON to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add collections_to_ranking_pipeline and input_to_reranker fields to evaluation JSON files",
        epilog="""
Examples:
  # Process single file with collections_per_query=5
  %(prog)s input.json -c 5

  # Process entire directory (all JSON files)
  %(prog)s evaluations/single_query_results/legacy_reranker/ -c 5

  # Process single file with custom output path
  %(prog)s input.json -c 10 -o output.json

  # Process multiple files with same collections_per_query
  %(prog)s file1.json file2.json file3.json -c 5

  # Process with multiple collections_per_query values (creates multiple output files)
  %(prog)s input.json -c 5 10 15

  # Process directory with multiple cpq values including k=all
  %(prog)s evaluations/results/ -c 5 10 15 all
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_paths',
        type=Path,
        nargs='+',
        help='Path(s) to input JSON file(s) or directory containing JSON files',
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Path to output JSON file (only valid with single input file)',
    )
    parser.add_argument(
        '-c', '--collections-per-query',
        type=lambda x: None if x.lower() == 'all' else int(x),
        nargs='+',
        default=[5],
        help='Number(s) of collections to select per query for ranking pipeline (default: 5). Use "all" for all collections.',
    )

    args = parser.parse_args()

    # Expand directories to JSON files
    input_files = []
    for path in args.input_paths:
        if not path.exists():
            print(f"Error: Path not found: {path}")
            return 1

        if path.is_dir():
            # Find all JSON files in directory
            json_files = sorted(path.glob('*.json'))
            if not json_files:
                print(f"Warning: No JSON files found in directory: {path}")
            else:
                print(f"Found {len(json_files)} JSON files in {path}")
                input_files.extend(json_files)
        elif path.is_file():
            if path.suffix.lower() == '.json':
                input_files.append(path)
            else:
                print(f"Warning: Skipping non-JSON file: {path}")
        else:
            print(f"Error: Invalid path: {path}")
            return 1

    if not input_files:
        print("Error: No JSON files to process")
        return 1

    # Validate single input file if custom output specified
    if args.output is not None and len(input_files) > 1:
        print("Error: Custom output path (-o) can only be used with a single input file")
        return 1

    # Validate single collections_per_query if custom output specified
    if args.output is not None and len(args.collections_per_query) > 1:
        print("Error: Custom output path (-o) can only be used with a single collections_per_query value")
        return 1

    # Process all combinations
    total_tasks = len(input_files) * len(args.collections_per_query)
    current_task = 0

    for input_file in input_files:

        for cpq in args.collections_per_query:
            current_task += 1
            print(f"\n{'='*80}")
            print(f"Task {current_task}/{total_tasks}: {input_file.name} with cpq={cpq}")
            print(f"{'='*80}")

            # Set output file
            if args.output is not None:
                output_file = args.output
            else:
                # create a new directory in the same level of input parent
                cpq_label = "all" if cpq is None else str(cpq)
                output_dir = input_file.parent / f"cpq_{cpq_label}_output"
                output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
                output_file = output_dir / (
                    input_file.stem + input_file.suffix
                )

            # Run the processing
            add_collections_to_ranking_pipeline(
                input_file,
                output_file,
                cpq,
            )

    print(f"\n{'='*80}")
    print(f"All tasks complete! Processed {total_tasks} file(s)")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    exit(main())
