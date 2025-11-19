#!/usr/bin/env python3
"""
Calculate decomposition-level binary recall for evaluation runs.

This script calculates recall from the decomposition perspective:
- Each decomposition gets a binary score: 100% if at least one expected concept is found, 0% otherwise
- Final query score = average of all minimum decomposition scores

This differs from the concept-level recall in calculate_accuracy.py which calculates
the proportion of all concepts found across all decompositions.

Usage:
    uv run python evaluations/calculate_decomposition_recall.py evaluations/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set


def load_truth_set(path: Path) -> Dict:
    """Load SME truth set."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_runs(path: Path) -> Dict:
    """Load evaluation runs metadata."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reexecuted_concepts(run_dir: Path, search_id: str) -> Set[str]:
    """Load concept IDs from reexecuted CMR queries."""
    reexecuted_dir = run_dir / "reexecuted"
    concept_file = reexecuted_dir / f"{search_id}_concepts.json"

    if not concept_file.exists():
        return set()

    with open(concept_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return set(data.get("data", []))


def calculate_decomposition_recall(
    truth_set: Dict,
    evaluation_runs: Dict,
    run_dir: Path,
    query_range: tuple = (11, 20),
) -> Dict:
    """
    Calculate binary decomposition-level recall.

    Args:
        truth_set: SME truth set with expected concept IDs
        evaluation_runs: Evaluation runs metadata
        run_dir: Directory containing reexecuted results
        query_range: Tuple of (start, end) query numbers to process (inclusive)

    Returns:
        Dictionary with decomposition-level recall statistics
    """
    # Create mapping of query_number -> search_id
    query_to_search_id = {}
    for result in evaluation_runs.get("results", []):
        if result.get("status") == "completed" and result.get("search_id"):
            query_to_search_id[result["query_number"]] = result["search_id"]

    query_scores = []
    total_decomps = 0
    total_decomps_found = 0

    start_query, end_query = query_range

    for query in truth_set["queries"]:
        query_num = query["query_number"]

        # Filter to specified range
        if query_num < start_query or query_num > end_query:
            continue

        # Skip if no search_id
        if query_num not in query_to_search_id:
            print(f"  WARNING: No search_id for query {query_num}, skipping")
            continue

        search_id = query_to_search_id[query_num]
        print(f"Processing query {query_num}: {query['query_text'][:60]}...")

        # Load CMR concepts
        cmr_concepts = load_reexecuted_concepts(run_dir, search_id)

        if not cmr_concepts:
            print(f"  WARNING: No reexecuted results found for {search_id}")
            continue

        # Process each topic and decomposition
        decomposition_scores = []

        for topic in query.get("topics", []):
            topic_title = topic.get("topic", "Unknown")

            for decomp in topic.get("decomps", []):
                # Only score minimum=true decompositions
                if not decomp.get("minimum", False):
                    continue

                decomp_title = decomp.get("decomp", "Unknown")
                expected_concepts = set(decomp.get("cmr_concept_ids", []))

                # Find which expected concepts are in CMR results
                found_concepts = expected_concepts & cmr_concepts
                missing_concepts = expected_concepts - cmr_concepts

                # Binary scoring: 100% if at least one concept found, 0% otherwise
                binary_score = 100.0 if len(found_concepts) > 0 else 0.0

                decomposition_scores.append(
                    {
                        "topic": topic_title,
                        "decomp": decomp_title,
                        "expected_count": len(expected_concepts),
                        "found_count": len(found_concepts),
                        "binary_score": binary_score,
                        "found_concept_ids": sorted(list(found_concepts)),
                        "missing_concept_ids": sorted(list(missing_concepts)),
                    },
                )

                total_decomps += 1
                if binary_score == 100.0:
                    total_decomps_found += 1

        # Calculate query average
        if decomposition_scores:
            query_average = sum(d["binary_score"] for d in decomposition_scores) / len(decomposition_scores)
            decomps_found = sum(1 for d in decomposition_scores if d["binary_score"] == 100.0)
        else:
            query_average = 0.0
            decomps_found = 0

        query_scores.append(
            {
                "query_number": query_num,
                "query_text": query["query_text"],
                "decomposition_scores": decomposition_scores,
                "decomps_scored": len(decomposition_scores),
                "decomps_found": decomps_found,
                "query_average": query_average,
            },
        )

        print(
            f"  Decomposition recall: {query_average:.1f}% ({decomps_found}/{len(decomposition_scores)} decomps found)",
        )

    # Calculate overall average
    if query_scores:
        overall_average = sum(q["query_average"] for q in query_scores) / len(query_scores)
    else:
        overall_average = 0.0

    return {
        "methodology": "Binary decomposition scoring: 100% if ≥1 concept found, 0% otherwise",
        "query_range": f"Queries {start_query}-{end_query}",
        "overall_average": overall_average,
        "queries_processed": len(query_scores),
        "total_decompositions": total_decomps,
        "total_decompositions_found": total_decomps_found,
        "decomposition_success_rate": (total_decomps_found / total_decomps * 100) if total_decomps > 0 else 0,
        "query_scores": query_scores,
        "summary": {
            "queries_perfect": sum(1 for q in query_scores if q["query_average"] == 100.0),
            "queries_partial": sum(1 for q in query_scores if 0 < q["query_average"] < 100.0),
            "queries_zero": sum(1 for q in query_scores if q["query_average"] == 0.0),
        },
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate decomposition-level binary recall for evaluation runs",
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to evaluation run directory (e.g., evaluations/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--truth-set",
        type=str,
        default="evaluations/truth_set_20251027_deduplicated.json",
        help="Path to truth set JSON file",
    )
    parser.add_argument(
        "--query-range",
        type=str,
        default="11-20",
        help='Query range to process (e.g., "11-20" or "1-10")',
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output path (default: run_dir/decomposition_recall.json)",
    )

    args = parser.parse_args()

    # Parse paths
    run_dir = Path(args.run_dir)
    truth_set_path = Path(args.truth_set)
    evaluation_runs_path = run_dir / "evaluation_runs.json"

    # Parse query range
    start_query, end_query = map(int, args.query_range.split("-"))

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "decomposition_recall.json"

    # Validate paths
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    if not truth_set_path.exists():
        print(f"ERROR: Truth set not found: {truth_set_path}")
        return 1

    if not evaluation_runs_path.exists():
        print(f"ERROR: evaluation_runs.json not found: {evaluation_runs_path}")
        return 1

    reexecuted_dir = run_dir / "reexecuted"
    if not reexecuted_dir.exists():
        print(f"ERROR: reexecuted directory not found: {reexecuted_dir}")
        print("Run reexecute_cmr_queries.py first")
        return 1

    print(f"Calculating decomposition-level recall for: {run_dir}")
    print()

    # Load data
    print(f"Loading truth set from: {truth_set_path}")
    truth_set = load_truth_set(truth_set_path)

    print("Loading evaluation runs metadata")
    evaluation_runs = load_evaluation_runs(evaluation_runs_path)
    print()

    # Calculate recall
    results = calculate_decomposition_recall(
        truth_set,
        evaluation_runs,
        run_dir,
        query_range=(start_query, end_query),
    )

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries processed: {results['queries_processed']}")
    print(f"Query range: {start_query}-{end_query}")
    print()
    print(f"Overall decomposition recall: {results['overall_average']:.2f}%")
    print(f"Decompositions found: {results['total_decompositions_found']}/{results['total_decompositions']}")
    print(f"Decomposition success rate: {results['decomposition_success_rate']:.2f}%")
    print()
    print("Query-level breakdown:")
    print(f"  Perfect (100%): {results['summary']['queries_perfect']}")
    print(f"  Partial (>0% <100%): {results['summary']['queries_partial']}")
    print(f"  Zero (0%): {results['summary']['queries_zero']}")

    # Save results
    print()
    print(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
