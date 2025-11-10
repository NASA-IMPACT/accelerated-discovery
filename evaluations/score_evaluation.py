#!/usr/bin/env python3
"""
Score agent evaluation outputs against SME truth set.

Compares top 5 collections from each agent decomposition against
expected CMR concept IDs from truth set.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save JSON file with pretty formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_agent_collection_pool(agent_output: Dict[str, Any]) -> Set[str]:
    """
    Extract top 5 concept IDs from each decomposition in agent output.

    Returns a set of all concept IDs found in top 5 of any decomposition.
    """
    pool = set()

    for topic in agent_output.get("topics", []):
        for decomp_result in topic.get("decomposition_results", []):
            # Get top 5 collections (implicit ranking by position)
            data_results = decomp_result.get("data_results", [])
            top_5 = data_results[:5]

            for collection in top_5:
                concept_id = collection.get("concept_id")
                if concept_id:
                    pool.add(concept_id)

    return pool


def score_query(
    truth_query: Dict[str, Any],
    agent_pool: Set[str],
) -> Dict[str, Any]:
    """
    Score a single query by comparing truth set against agent collection pool.

    Returns:
        - decomposition_scores: List of per-decomp results
        - score_percent: Average of minimum decomp scores
        - minimum_decomps_total: Count of minimum=true decomps
        - minimum_decomps_matched: Count of matched decomps
    """
    decomp_scores = []
    scores_for_average = []

    # Iterate through truth set topics and decomps
    for topic in truth_query.get("topics", []):
        topic_name = topic.get("topic", "")

        for decomp in topic.get("decomps", []):
            decomp_name = decomp.get("decomp", "")
            minimum = decomp.get("minimum", False)
            truth_ids = decomp.get("cmr_concept_ids", [])

            # Check if ANY truth ID exists in agent pool
            matched_ids = [tid for tid in truth_ids if tid in agent_pool]
            matched = len(matched_ids) > 0
            score = 100.0 if matched else 0.0

            decomp_score = {
                "topic": topic_name,
                "decomp": decomp_name,
                "minimum": minimum,
                "truth_concept_ids": truth_ids,
                "matched": matched,
                "matched_ids": matched_ids,
                "score": score,
            }

            decomp_scores.append(decomp_score)

            # Only count minimum=true decomps in average
            if minimum:
                scores_for_average.append(score)

    # Calculate query score (average of minimum decomps)
    if scores_for_average:
        query_score = sum(scores_for_average) / len(scores_for_average)
    else:
        query_score = 0.0

    minimum_decomps_matched = sum(1 for s in scores_for_average if s == 100.0)

    return {
        "decomposition_scores": decomp_scores,
        "score_percent": query_score,
        "minimum_decomps_total": len(scores_for_average),
        "minimum_decomps_matched": minimum_decomps_matched,
    }


def score_evaluation(
    runs_file: Path,
    truth_file: Path,
    captured_data_dir: Path,
) -> Dict[str, Any]:
    """
    Score all queries in an evaluation run.

    Args:
        runs_file: Path to evaluation_runs.json
        truth_file: Path to truth_set_20251027.json
        captured_data_dir: Directory containing agent output files

    Returns:
        Complete evaluation statistics dictionary
    """
    # Load input files
    print(f"Loading {runs_file}...")
    runs = load_json(runs_file)

    print(f"Loading {truth_file}...")
    truth_set = load_json(truth_file)

    # Get run_subdir from runs metadata if it exists
    run_subdir = runs.get("run_subdir")
    if run_subdir:
        print(f"Using run subdirectory: {run_subdir}")
        captured_data_dir = captured_data_dir / run_subdir

    # Create truth set lookup by query number
    truth_lookup = {q["query_number"]: q for q in truth_set.get("queries", [])}

    # Score each completed query
    query_scores = []

    for result in runs.get("results", []):
        if result.get("status") != "completed":
            print(
                f"Skipping query {result['query_number']} (status: {result.get('status')})",
            )
            continue

        query_num = result["query_number"]
        output_file = result.get("output_file")

        if not output_file:
            print(f"Skipping query {query_num} (no output file)")
            continue

        print(f"\nScoring query {query_num}...")

        # Load agent output - handle both old flat structure and new subfolder structure
        # Try direct path first (handles paths with captured_data/ prefix)
        output_path = Path(output_file)
        if not output_path.exists():
            # Try relative to captured_data_dir (for backwards compatibility)
            output_path = captured_data_dir / Path(output_file).name

        if not output_path.exists():
            print(f"  ERROR: Output file not found: {output_file}")
            continue

        agent_data = load_json(output_path)
        agent_output = agent_data.get("agent_output", {})

        # Get truth set for this query
        truth_query = truth_lookup.get(query_num)
        if not truth_query:
            print(f"  ERROR: No truth set for query {query_num}")
            continue

        # Extract agent collection pool
        agent_pool = extract_agent_collection_pool(agent_output)
        print(f"  Agent collection pool size: {len(agent_pool)}")

        # Score the query
        scoring_result = score_query(truth_query, agent_pool)

        # Build query score entry
        query_score_entry = {
            "query_number": query_num,
            "query_text": result.get("query_text", truth_query.get("query_text", "")),
            "score_percent": scoring_result["score_percent"],
            "agent_pool_size": len(agent_pool),
            "decomposition_scores": scoring_result["decomposition_scores"],
            "minimum_decomps_total": scoring_result["minimum_decomps_total"],
            "minimum_decomps_matched": scoring_result["minimum_decomps_matched"],
            "agent_metadata": {
                "duration_seconds": result.get("duration_seconds"),
                "topics_processed": result.get("topics_processed"),
                "total_cmr_results": result.get("total_cmr_results"),
                "total_filtered_results": result.get("total_filtered_results"),
            },
        }

        query_scores.append(query_score_entry)

        print(
            f"  Score: {scoring_result['score_percent']:.1f}% "
            f"({scoring_result['minimum_decomps_matched']}/{scoring_result['minimum_decomps_total']} decomps matched)",
        )

    # Calculate summary statistics
    if query_scores:
        total_queries = len(query_scores)
        avg_score = sum(q["score_percent"] for q in query_scores) / total_queries
        queries_100 = sum(1 for q in query_scores if q["score_percent"] == 100.0)
        queries_50_plus = sum(1 for q in query_scores if q["score_percent"] >= 50.0)
        queries_0 = sum(1 for q in query_scores if q["score_percent"] == 0.0)

        total_min_decomps = sum(q["minimum_decomps_total"] for q in query_scores)
        total_matched = sum(q["minimum_decomps_matched"] for q in query_scores)
        match_rate = (
            (total_matched / total_min_decomps * 100.0)
            if total_min_decomps > 0
            else 0.0
        )

        summary = {
            "average_score_percent": round(avg_score, 2),
            "queries_100_percent": queries_100,
            "queries_50_plus_percent": queries_50_plus,
            "queries_0_percent": queries_0,
            "total_minimum_decomps": total_min_decomps,
            "total_decomps_matched": total_matched,
            "match_rate_percent": round(match_rate, 2),
        }
    else:
        summary = {}

    # Build final output
    evaluation_stats = {
        "evaluation_metadata": {
            "scored_timestamp": datetime.now().isoformat(),
            "run_file": str(runs_file),
            "truth_set_file": str(truth_file),
            "model": runs.get("model", "unknown"),
            "queries_scored": len(query_scores),
            "queries_total": runs.get("queries_total", 0),
        },
        "query_scores": query_scores,
        "summary": summary,
    }

    return evaluation_stats


def find_latest_run_dir() -> Path | None:
    """Find the most recent run subdirectory in evaluations/"""
    evaluations_dir = Path("evaluations")
    run_dirs = [d for d in evaluations_dir.glob("run_*") if d.is_dir()]
    if not run_dirs:
        return None
    # Sort by directory name (which includes timestamp)
    return sorted(run_dirs)[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Score agent evaluation outputs against truth set",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Path to evaluation_runs.json (default: auto-detect latest run subdirectory)",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("evaluations/truth_set_20251027.json"),
        help="Path to truth set JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for evaluation statistics (default: same directory as runs file)",
    )

    args = parser.parse_args()

    # Auto-detect latest run if not specified
    if args.runs is None:
        latest_run_dir = find_latest_run_dir()
        if latest_run_dir is None:
            print("❌ No run subdirectories found in evaluations/")
            print("   Please specify --runs explicitly or run an evaluation first")
            return 1
        runs_file = (latest_run_dir / "evaluation_runs.json").resolve()
        print(f"📁 Auto-detected latest run: {latest_run_dir.name}")
    else:
        runs_file = args.runs.resolve()

    # Set output file to same directory as runs file if not specified
    if args.output is None:
        output_file = (runs_file.parent / "evaluation_stats.json").resolve()
    else:
        output_file = args.output.resolve()

    # Resolve paths
    truth_file = args.truth.resolve()
    captured_data_dir = runs_file.parent.parent / "captured_data"

    print("=" * 70)
    print("EVALUATION SCORING")
    print("=" * 70)

    # Score evaluation
    stats = score_evaluation(runs_file, truth_file, captured_data_dir)

    # Save results
    print(f"\nSaving results to {output_file}...")
    save_json(stats, output_file)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = stats.get("summary", {})
    print(f"Queries scored: {stats['evaluation_metadata']['queries_scored']}")
    print(f"Average score: {summary.get('average_score_percent', 0):.1f}%")
    print(f"Queries at 100%: {summary.get('queries_100_percent', 0)}")
    print(f"Queries at 50%+: {summary.get('queries_50_plus_percent', 0)}")
    print(f"Queries at 0%: {summary.get('queries_0_percent', 0)}")
    print(
        f"Decomposition match rate: {summary.get('match_rate_percent', 0):.1f}% "
        f"({summary.get('total_decomps_matched', 0)}/{summary.get('total_minimum_decomps', 0)})",
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
