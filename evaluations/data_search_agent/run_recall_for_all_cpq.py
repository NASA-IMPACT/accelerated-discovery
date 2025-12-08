#!/usr/bin/env python3
"""
Run recall calculation for all cpq_*_output directories and generate summary table.

This script:
1. Finds all cpq_*_output directories in a given parent directory
2. Calls calculate_batch_recall for each directory with k=all
3. Collects results and generates a summary table comparing metrics across different cpq values
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Import the calculate_batch_recall function
# sys.path.insert(0, str(Path(__file__).parent / "create_notebook"))
from create_notebook.calculate_recall import calculate_batch_recall


def find_cpq_directories(parent_dir: Path) -> List[tuple[str, Path]]:
    """
    Find all cpq_*_output directories in the parent directory.

    Args:
        parent_dir: Parent directory containing cpq_*_output subdirectories

    Returns:
        List of (cpq_value, directory_path) tuples, sorted by cpq_value
    """
    cpq_dirs = []

    for path in parent_dir.iterdir():
        if path.is_dir() and path.name.startswith("cpq_") and path.name.endswith("_output"):
            # Extract cpq value from directory name (e.g., "cpq_5_output" -> "5", "cpq_all_output" -> "all")
            try:
                cpq_value = path.name.split("_")[1]
                cpq_dirs.append((cpq_value, path))
            except (IndexError, ValueError):
                print(f"Warning: Skipping directory with unexpected name format: {path.name}")

    # Sort by cpq value: numeric values first, then "all" at the end
    def sort_key(item):
        cpq_val = item[0]
        try:
            return (0, int(cpq_val))  # Numeric values first, sorted numerically
        except ValueError:
            return (1, cpq_val)  # Non-numeric (like "all") at the end, sorted alphabetically

    cpq_dirs.sort(key=sort_key)

    return cpq_dirs


def extract_metrics_from_batch_results(batch_results: Dict) -> Dict:
    """
    Extract aggregate metrics from batch recall results.

    Args:
        batch_results: Results dictionary from calculate_batch_recall

    Returns:
        Dictionary with aggregated metrics for both minimum_true and all_decomps
    """
    if not batch_results:
        return None

    return {
        "minimum_true_recall": batch_results["minimum_true"]["average_collection_recall"],
        "minimum_true_coverage": batch_results["minimum_true"]["average_decomposition_coverage"],
        "all_decomps_recall": batch_results["all_decomps"]["average_collection_recall"],
        "all_decomps_coverage": batch_results["all_decomps"]["average_decomposition_coverage"],
        "matched_queries": batch_results["matched_queries"],
        "total_files": batch_results["total_files"],
    }


def create_summary_table(results: List[tuple[int, Dict]]) -> pd.DataFrame:
    """
    Create a summary table from all results.

    Args:
        results: List of (cpq_value, metrics) tuples

    Returns:
        Pandas DataFrame with summary
    """
    rows = []

    for cpq_value, metrics in results:
        if metrics:
            rows.append({
                "collections_per_query": cpq_value,
                "min_true_recall": f"{metrics['minimum_true_recall']:.2%}",
                "min_true_coverage": f"{metrics['minimum_true_coverage']:.2%}",
                "all_decomps_recall": f"{metrics['all_decomps_recall']:.2%}",
                "all_decomps_coverage": f"{metrics['all_decomps_coverage']:.2%}",
                "matched_queries": metrics['matched_queries'],
                "total_files": metrics['total_files'],
            })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run recall calculation for all cpq_*_output directories and generate summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 evaluations/run_recall_for_all_cpq.py \\
    --parent-dir evaluations/single_query_results/legacy_reranker_new_config \\
    --truth-file evaluations/create_notebook/truth_set_20251027_deduplicated.json \\
    --filtering-stage after_filtering
        """,
    )

    parser.add_argument(
        "--parent-dir",
        type=Path,
        required=True,
        help="Parent directory containing cpq_*_output subdirectories",
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        default = "evaluations/create_notebook/truth_set_20251027_deduplicated.json" , 
        help="Path to truth set JSON file",
    )
    parser.add_argument(
        "--filtering-stage",
        type=str,
        choices=["prior_filtering", "after_filtering", "after_ranking"],
        default="after_filtering",
        help="Which filtering stage to use (default: after_filtering)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save summary CSV (default: summary_cpq_comparison.csv in parent dir)",
    )

    args = parser.parse_args()

    # Validate parent directory
    if not args.parent_dir.exists():
        print(f"Error: Parent directory not found: {args.parent_dir}")
        return 1

    # Validate truth file
    if not args.truth_file.exists():
        print(f"Error: Truth file not found: {args.truth_file}")
        return 1

    # Find all cpq directories
    cpq_dirs = find_cpq_directories(args.parent_dir)

    if not cpq_dirs:
        print(f"Error: No cpq_*_output directories found in {args.parent_dir}")
        return 1

    print(f"Found {len(cpq_dirs)} cpq directories:")
    for cpq_value, path in cpq_dirs:
        print(f"  - cpq={cpq_value}: {path.name}")

    print(f"\nRunning recall calculation with:")
    print(f"  - top_k: all")
    print(f"  - filtering_stage: {args.filtering_stage}")
    print(f"  - truth_file: {args.truth_file.name}")
    print()

    # Run recall calculation for each directory
    results = []

    for cpq_value, cpq_dir in cpq_dirs:
        print("=" * 80)
        print(f"Processing cpq={cpq_value} ({cpq_dir.name})")
        print("=" * 80)

        # Call calculate_batch_recall with k=all (None)
        batch_results = calculate_batch_recall(
            agent_dir=cpq_dir,
            truth_file=args.truth_file,
            top_k=None,  # None means "all"
            filtering_stage=args.filtering_stage,
        )

        # Extract metrics
        metrics = extract_metrics_from_batch_results(batch_results) if batch_results else None

        if metrics:
            print(f"Results for cpq={cpq_value}:")
            print(f"  Minimum=True Recall: {metrics['minimum_true_recall']:.2%}")
            print(f"  Minimum=True Coverage: {metrics['minimum_true_coverage']:.2%}")
            print(f"  All Decomps Recall: {metrics['all_decomps_recall']:.2%}")
            print(f"  All Decomps Coverage: {metrics['all_decomps_coverage']:.2%}")
            print(f"  Matched Queries: {metrics['matched_queries']}/{metrics['total_files']}")
        else:
            print(f"Warning: No metrics extracted for cpq={cpq_value}")

        results.append((cpq_value, metrics))
        print()

    # Create summary table
    print("=" * 80)
    print("SUMMARY TABLE - Collections Per Query Comparison")
    print("=" * 80)

    # Print formatted table matching calculate_recall.py format
    print(f"\n{'CPQ Value':<12} {'Min=T ConceptIDs':<25} {'Min=T Decomps':<25} {'All ConceptIDs':<25} {'All Decomps':<25}")
    print(f"{'':12} {'Recall (%)':<25} {'Coverage (%)':<25} {'Recall (%)':<25} {'Coverage (%)':<25}")
    print(f"{'-'*112}")

    for cpq_value, metrics in results:
        if metrics:
            print(
                f"{cpq_value:<12} "
                f"{metrics['minimum_true_recall']:<25.2f} "
                f"{metrics['minimum_true_coverage']:<25.2f} "
                f"{metrics['all_decomps_recall']:<25.2f} "
                f"{metrics['all_decomps_coverage']:<25.2f}"
            )

    # Also create CSV for easy analysis
    summary_df = create_summary_table(results)
    output_file = args.output or (args.parent_dir / f"summary_cpq_comparison_{args.filtering_stage}.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Summary saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
