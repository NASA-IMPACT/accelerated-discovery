#!/usr/bin/env python3
"""
Calculate evaluation accuracy by comparing truth set against reexecuted full CMR results.

This script scores the data search agent's performance by checking if expected
concept IDs (from SME truth set) appear anywhere in the full set of CMR results
returned for each query.

Usage:
    uv run python evaluations/calculate_accuracy.py evaluations/run_20251027_213136
    uv run python evaluations/calculate_accuracy.py evaluations/run_20251027_213136 --output my_accuracy.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def load_truth_set(truth_set_path: Path) -> dict:
    """Load truth set JSON file."""
    with open(truth_set_path) as f:
        return json.load(f)


def load_evaluation_runs(eval_runs_path: Path) -> dict:
    """Load evaluation runs JSON file."""
    with open(eval_runs_path) as f:
        return json.load(f)


def load_reexecuted_concepts(reexecuted_path: Path) -> List[str]:
    """Load concept IDs from reexecuted results file."""
    try:
        with open(reexecuted_path) as f:
            data = json.load(f)
            return data.get("data", [])
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse {reexecuted_path}: {e}", file=sys.stderr)
        return None


def find_evaluation_result(query_number: int, eval_runs: dict) -> Optional[dict]:
    """Find evaluation result entry for a given query number."""
    for result in eval_runs.get("results", []):
        if result.get("query_number") == query_number:
            return result
    return None


def score_decomposition(
    expected_ids: List[str],
    full_concept_ids: List[str],
) -> Tuple[List[str], float]:
    """
    Score a single decomposition by checking if expected concept IDs are in full results.

    Returns:
        Tuple of (found_ids, score_percent)
    """
    found_ids = [cid for cid in expected_ids if cid in full_concept_ids]

    if not expected_ids:
        return [], 100.0  # No expectations = perfect score

    # Score is percentage of expected IDs that were found
    score = (len(found_ids) / len(expected_ids)) * 100.0
    return found_ids, score


def calculate_query_accuracy(
    query: dict,
    full_concept_ids: List[str],
) -> Tuple[float, List[dict]]:
    """
    Calculate accuracy for a single query across all minimum decompositions.

    Returns:
        Tuple of (query_accuracy_percent, decomposition_details_list)
    """
    decomp_details = []
    decomp_scores = []
    total_expected = 0
    total_found = 0

    # Iterate through all topics and decompositions
    for topic in query.get("topics", []):
        topic_name = topic.get("topic")

        for decomp in topic.get("decomps", []):
            # Only score minimum decompositions
            if not decomp.get("minimum", False):
                continue

            decomp_name = decomp.get("decomp")
            expected_ids = decomp.get("cmr_concept_ids", [])

            # Score this decomposition
            found_ids, score = score_decomposition(expected_ids, full_concept_ids)

            decomp_details.append(
                {
                    "topic": topic_name,
                    "decomp": decomp_name,
                    "expected_concept_ids": expected_ids,
                    "found_concept_ids": found_ids,
                    "matched": len(found_ids) == len(expected_ids),
                    "score_percent": score,
                },
            )

            decomp_scores.append(score)
            total_expected += len(expected_ids)
            total_found += len(found_ids)

    # Query accuracy is average of all minimum decomposition scores
    query_accuracy = sum(decomp_scores) / len(decomp_scores) if decomp_scores else 0.0

    return query_accuracy, decomp_details


def main(run_dir: Path, truth_set_path: Path, output_path: Optional[Path] = None):
    """
    Main function to calculate accuracy for an evaluation run.
    """
    print(f"Calculating accuracy for: {run_dir}\n")

    # Load truth set
    truth_set = load_truth_set(truth_set_path)
    queries = truth_set.get("queries", [])
    print(f"Loaded truth set with {len(queries)} queries")

    # Load evaluation runs
    eval_runs_path = run_dir / "evaluation_runs.json"
    if not eval_runs_path.exists():
        print(
            f"ERROR: evaluation_runs.json not found at {eval_runs_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    eval_runs = load_evaluation_runs(eval_runs_path)
    print("Loaded evaluation runs metadata\n")

    # Prepare reexecuted directory
    reexecuted_dir = run_dir / "reexecuted"
    if not reexecuted_dir.exists():
        print(
            f"ERROR: reexecuted directory not found at {reexecuted_dir}",
            file=sys.stderr,
        )
        print(
            "Run reexecute_cmr_queries.py first to generate full CMR results",
            file=sys.stderr,
        )
        sys.exit(1)

    # Process each query
    query_scores = []
    queries_processed = 0
    queries_skipped = 0

    for query in queries:
        query_number = query.get("query_number")
        query_text = query.get("query_text", "")

        print(f"Processing query {query_number}: {query_text[:60]}...")

        # Find corresponding evaluation result
        eval_result = find_evaluation_result(query_number, eval_runs)
        if not eval_result:
            print(f"  WARNING: No evaluation result found for query {query_number}")
            queries_skipped += 1
            continue

        search_id = eval_result.get("search_id")
        if not search_id:
            print("  WARNING: No search_id found in evaluation result")
            queries_skipped += 1
            continue

        # Load reexecuted concept IDs
        reexecuted_path = reexecuted_dir / f"{search_id}_concepts.json"
        full_concept_ids = load_reexecuted_concepts(reexecuted_path)

        if full_concept_ids is None:
            print(
                f"  WARNING: Could not load reexecuted results from {reexecuted_path}",
            )
            queries_skipped += 1
            continue

        # Calculate accuracy for this query
        query_accuracy, decomp_details = calculate_query_accuracy(
            query,
            full_concept_ids,
        )

        # Count total expected and found concepts
        total_expected = sum(len(d["expected_concept_ids"]) for d in decomp_details)
        total_found = sum(len(d["found_concept_ids"]) for d in decomp_details)

        query_scores.append(
            {
                "query_number": query_number,
                "query_text": query_text,
                "search_id": search_id,
                "total_expected_concepts": total_expected,
                "total_found_concepts": total_found,
                "accuracy_percent": round(query_accuracy, 2),
                "minimum_decomps_evaluated": len(decomp_details),
                "decomposition_details": decomp_details,
            },
        )

        print(
            f"  Accuracy: {query_accuracy:.1f}% ({total_found}/{total_expected} concepts found)",
        )
        queries_processed += 1

    # Calculate overall statistics
    overall_accuracy = (
        sum(q["accuracy_percent"] for q in query_scores) / len(query_scores)
        if query_scores
        else 0.0
    )

    queries_perfect = sum(1 for q in query_scores if q["accuracy_percent"] == 100.0)
    queries_partial = sum(1 for q in query_scores if 0 < q["accuracy_percent"] < 100.0)
    queries_zero = sum(1 for q in query_scores if q["accuracy_percent"] == 0.0)

    # Decomposition-level statistics
    all_decomps = [d for q in query_scores for d in q["decomposition_details"]]
    decomps_fully_matched = sum(1 for d in all_decomps if d["matched"])
    decomps_partially_matched = sum(
        1 for d in all_decomps if len(d["found_concept_ids"]) > 0 and not d["matched"]
    )
    decomps_not_matched = sum(
        1 for d in all_decomps if len(d["found_concept_ids"]) == 0
    )

    # Build output
    output_data = {
        "run_directory": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(queries),
        "queries_processed": queries_processed,
        "queries_skipped": queries_skipped,
        "overall_accuracy": round(overall_accuracy, 2),
        "query_scores": query_scores,
        "summary": {
            "queries_perfect_score": queries_perfect,
            "queries_partial_score": queries_partial,
            "queries_zero_score": queries_zero,
            "total_minimum_decomps": len(all_decomps),
            "decomps_fully_matched": decomps_fully_matched,
            "decomps_partially_matched": decomps_partially_matched,
            "decomps_not_matched": decomps_not_matched,
        },
    }

    # Determine output path
    if output_path is None:
        output_path = run_dir / "evaluation_accuracy.json"

    # Save results
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries processed: {queries_processed}/{len(queries)}")
    print(f"Queries skipped: {queries_skipped}")
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")
    print("\nQuery-level breakdown:")
    print(f"  Perfect (100%): {queries_perfect}")
    print(f"  Partial (>0% <100%): {queries_partial}")
    print(f"  Zero (0%): {queries_zero}")
    print("\nDecomposition-level breakdown:")
    print(f"  Fully matched: {decomps_fully_matched}/{len(all_decomps)}")
    print(f"  Partially matched: {decomps_partially_matched}/{len(all_decomps)}")
    print(f"  Not matched: {decomps_not_matched}/{len(all_decomps)}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate evaluation accuracy against reexecuted full CMR results",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to evaluation run directory (e.g., evaluations/run_20251027_213136)",
    )
    parser.add_argument(
        "--truth-set",
        type=Path,
        default=Path("evaluations/truth_set_20251027.json"),
        help="Path to truth set JSON file (default: evaluations/truth_set_20251027.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: <run_dir>/evaluation_accuracy.json)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.truth_set.exists():
        print(f"ERROR: Truth set not found: {args.truth_set}", file=sys.stderr)
        sys.exit(1)

    main(args.run_dir, args.truth_set, args.output)
