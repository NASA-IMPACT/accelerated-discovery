"""
Calculate collection recall and decomposition coverage metrics.

Compares agent output (data_results) against truth set (expected CMR concept IDs).

Metrics:
1. Collection Recall: % of expected concept IDs found in agent's data_results
2. Decomposition Coverage: % of decompositions where at least 1 expected concept ID was found
3. Query-level metrics: Aggregated across all decompositions per query
"""
import datetime
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_truth_set(truth_file: Path) -> Dict:
    """Load truth set JSON."""
    with open(truth_file) as f:
        return json.load(f)


def load_agent_output(agent_file: Path) -> Dict:
    """Load agent output JSON."""
    with open(agent_file) as f:
        data = json.load(f)
    return data.get("agent_output", data)


def extract_agent_concept_ids(
    agent_output: Dict,
    k: int | None = None,
    filtering_stage: str = "after_filtering",
) -> Set[str]:
    """
    Extract concept IDs from agent's output across all topics/decompositions.

    Args:
        agent_output: Agent output dictionary
        k: Number of top concept IDs to extract per decomposition. If None, extract all.
        filtering_stage: Which stage to extract from:
            - "prior_filtering": Extract from all_collections_from_cmr (before any filtering)
            - "after_filtering": Extract from input_to_reranker (after collections_per_query filtering)
            - "after_ranking": Extract from data_results (after reranking, default)

    Returns:
        Set of concept IDs found in the specified stage
    """
    concept_ids = set()

    # Determine which key to use based on filtering stage
    if filtering_stage == "prior_filtering":
        results_key = "all_collections_from_cmr"
    elif filtering_stage == "after_filtering":
        results_key = "input_to_reranker"
    elif filtering_stage == "after_ranking":
        # results_key = "data_results"
        results_key = "reranked_results"
    else:
        raise ValueError(
            f"Invalid filtering_stage: {filtering_stage}. "
            "Must be 'prior_filtering', 'after_filtering', or 'after_ranking'"
        )

    for topic in agent_output.get("topics", []):
        for decomp_result in topic.get("decomposition_results", []):
            set_for_decomp_result = set()
            results = decomp_result.get(results_key, [])

            # previously we just used :k of results 
            
            unique_concept_ids = list(dict.fromkeys(col.get("concept_id") for col in results if col.get("concept_id")))
            print(f"Unique concept IDs: {len(unique_concept_ids)}")

            for concept_id in unique_concept_ids:
                set_for_decomp_result.add(concept_id)

                if k and len(set_for_decomp_result) == k:
                    break
            
            concept_ids.update(set_for_decomp_result)

    print(f"Extracted {len(concept_ids)} unique concept IDs from agent output for top k={k}")
    return concept_ids


def match_query_by_text(
    agent_query: str, truth_queries: List[Dict]
) -> Dict | None:
    """
    Match agent query to truth set query by text similarity.

    Simple exact match after normalization. Can be enhanced with fuzzy matching.

    Args:
        agent_query: Query text from agent output
        truth_queries: List of truth set queries

    Returns:
        Matching truth query or None
    """
    # Normalize query text
    agent_norm = agent_query.lower().strip()

    for truth_query in truth_queries:
        truth_norm = truth_query["query_text"].lower().strip()
        if agent_norm == truth_norm:
            return truth_query

    return None


def calculate_recall_for_query(
    agent_concept_ids: Set[str],
    truth_query: Dict,
) -> Dict:
    """
    Calculate recall metrics for a single query.

    Calculates metrics separately for:
    - minimum=True: Only decompositions where minimum=True
    - minimum=False: ALL decompositions (both minimum=True and minimum=False)

    Args:
        agent_concept_ids: Set of concept IDs found by agent
        truth_query: Truth set query with expected concept IDs

    Returns:
        Dictionary with recall metrics for both minimum=True and minimum=False
    """
    # Metrics for minimum=True (only minimum=True decomps)
    min_true_metrics = {
        "total_expected": 0,
        "total_found": 0,
        "decomps_with_coverage": 0,
        "total_decomps": 0,
        "decomp_details": [],
    }

    # Metrics for minimum=False (ALL decomps, regardless of minimum value)
    min_false_metrics = {
        "total_expected": 0,
        "total_found": 0,
        "decomps_with_coverage": 0,
        "total_decomps": 0,
        "decomp_details": [],
    }

    for topic in truth_query.get("topics", []):
        for decomp in topic.get("decomps", []):
            is_minimum = decomp.get("minimum", False)
            expected_ids = set(decomp.get("cmr_concept_ids", []))

            # Remove empty strings
            expected_ids = {cid for cid in expected_ids if cid}

            if not expected_ids:
                # No expected IDs - skip this decomposition
                continue

            found_ids = agent_concept_ids & expected_ids

            # For minimum=True: only include decomps where minimum=True
            if is_minimum:
                min_true_metrics["total_decomps"] += 1
                min_true_metrics["total_expected"] += len(expected_ids)
                min_true_metrics["total_found"] += len(found_ids)

                if len(found_ids) > 0:
                    min_true_metrics["decomps_with_coverage"] += 1

                min_true_metrics["decomp_details"].append(
                    {
                        "topic": topic.get("topic"),
                        "decomp": decomp.get("decomp"),
                        "minimum": is_minimum,
                        "expected_count": len(expected_ids),
                        "found_count": len(found_ids),
                        "expected_ids": list(expected_ids),
                        "found_ids": list(found_ids),
                        "missing_ids": list(expected_ids - found_ids),
                        "has_coverage": len(found_ids) > 0,
                    }
                )

            # For minimum=False: include ALL decomps (both True and False)
            min_false_metrics["total_decomps"] += 1
            min_false_metrics["total_expected"] += len(expected_ids)
            min_false_metrics["total_found"] += len(found_ids)

            if len(found_ids) > 0:
                min_false_metrics["decomps_with_coverage"] += 1

            min_false_metrics["decomp_details"].append(
                {
                    "topic": topic.get("topic"),
                    "decomp": decomp.get("decomp"),
                    "minimum": is_minimum,
                    "expected_count": len(expected_ids),
                    "found_count": len(found_ids),
                    "expected_ids": list(expected_ids),
                    "found_ids": list(found_ids),
                    "missing_ids": list(expected_ids - found_ids),
                    "has_coverage": len(found_ids) > 0,
                }
            )

    # Calculate metrics for minimum=True (only minimum=True decomps)
    min_true_recall = (
        (min_true_metrics["total_found"] / min_true_metrics["total_expected"] * 100)
        if min_true_metrics["total_expected"] > 0
        else 0
    )
    min_true_coverage = (
        (min_true_metrics["decomps_with_coverage"] / min_true_metrics["total_decomps"] * 100)
        if min_true_metrics["total_decomps"] > 0
        else 0
    )

    # Calculate metrics for minimum=False (ALL decomps)
    min_false_recall = (
        (min_false_metrics["total_found"] / min_false_metrics["total_expected"] * 100)
        if min_false_metrics["total_expected"] > 0
        else 0
    )
    min_false_coverage = (
        (
            min_false_metrics["decomps_with_coverage"]
            / min_false_metrics["total_decomps"]
            * 100
        )
        if min_false_metrics["total_decomps"] > 0
        else 0
    )

    return {
        "query_number": truth_query.get("query_number"),
        "query_text": truth_query.get("query_text"),
        "minimum_true": {
            "total_expected_concepts": min_true_metrics["total_expected"],
            "total_found_concepts": min_true_metrics["total_found"],
            "total_decomps": min_true_metrics["total_decomps"],
            "decomps_with_coverage": min_true_metrics["decomps_with_coverage"],
            "collection_recall": min_true_recall,
            "decomposition_coverage": min_true_coverage,
            "decomp_details": min_true_metrics["decomp_details"],
        },
        "all_decomps": {
            "total_expected_concepts": min_false_metrics["total_expected"],
            "total_found_concepts": min_false_metrics["total_found"],
            "total_decomps": min_false_metrics["total_decomps"],
            "decomps_with_coverage": min_false_metrics["decomps_with_coverage"],
            "collection_recall": min_false_recall,
            "decomposition_coverage": min_false_coverage,
            "decomp_details": min_false_metrics["decomp_details"],
        },
    }


def calculate_overall_recall(
    agent_file: Path,
    truth_file: Path,
    verbose: bool = True,
    top_k: int | None = None,
    filtering_stage: str = "after_filtering",
) -> Dict:
    """
    Calculate overall recall metrics across all queries.

    Calculates metrics separately for minimum=True and minimum=False decompositions.

    Args:
        agent_file: Path to agent output JSON
        truth_file: Path to truth set JSON
        verbose: If True, print detailed results
        top_k: Number of top concept IDs to extract per decomposition. If None, extract all.
        filtering_stage: Which stage to extract from:
            - "prior_filtering": Extract from all_collections_from_cmr (before any filtering)
            - "after_filtering": Extract from input_to_reranker (after collections_per_query filtering)
            - "after_ranking": Extract from data_results (after reranking, default)

    Returns:
        Dictionary with overall and per-query metrics
    """
    # Load data
    agent_output = load_agent_output(agent_file)
    truth_set = load_truth_set(truth_file)

    # Extract all concept IDs from agent output
    agent_concept_ids = extract_agent_concept_ids(
        agent_output, k=top_k, filtering_stage=filtering_stage
    )

    if verbose:
        print(f"Filtering stage: {filtering_stage}")
        print(f"Agent found {len(agent_concept_ids)} unique concept IDs total")
        print(f"Truth set has {len(truth_set['queries'])} queries\n")

    # Match query
    agent_query = agent_output["search_metadata"]["original_query"]
    truth_query = match_query_by_text(agent_query, truth_set["queries"])

    if not truth_query:
        print(f"ERROR: Could not match query: {agent_query}")
        return None

    # Calculate recall for this query (both minimum=True and minimum=False)
    query_metrics = calculate_recall_for_query(agent_concept_ids, truth_query)

    if verbose:
        print(f"Query {query_metrics['query_number']}: {query_metrics['query_text']}\n")

        # Print minimum=True metrics (only minimum=True decomps)
        min_true = query_metrics["minimum_true"]
        print("MINIMUM=TRUE (only minimum=True decompositions):")
        print(f"  Collection Recall: {min_true['collection_recall']:.1f}%")
        print(
            f"    - Found {min_true['total_found_concepts']}/{min_true['total_expected_concepts']} expected concept IDs"
        )
        print(f"  Decomposition Coverage: {min_true['decomposition_coverage']:.1f}%")
        print(
            f"    - {min_true['decomps_with_coverage']}/{min_true['total_decomps']} decompositions have at least 1 expected concept ID"
        )

        # Print all_decomps metrics (ALL decomps)
        all_decomps = query_metrics["all_decomps"]
        print(f"\nALL DECOMPOSITIONS (both minimum=True and minimum=False):")
        print(f"  Collection Recall: {all_decomps['collection_recall']:.1f}%")
        print(
            f"    - Found {all_decomps['total_found_concepts']}/{all_decomps['total_expected_concepts']} expected concept IDs"
        )
        print(f"  Decomposition Coverage: {all_decomps['decomposition_coverage']:.1f}%")
        print(
            f"    - {all_decomps['decomps_with_coverage']}/{all_decomps['total_decomps']} decompositions have at least 1 expected concept ID"
        )

        print(f"\nPer-Decomposition Details:")
        print(f"{'='*80}")

        # Combine all decomp details for display
        all_details = min_true["decomp_details"] + all_decomps["decomp_details"]
        for detail in all_details:
            status = "✓" if detail["has_coverage"] else "✗"
            print(
                f"{status} [{detail['topic']}] {detail['decomp']} (minimum={detail['minimum']})"
            )
            print(
                f"   Found {detail['found_count']}/{detail['expected_count']} concept IDs"
            )
            if detail["found_ids"]:
                print(f"   Found: {', '.join(detail['found_ids'])}")
            if detail["missing_ids"]:
                print(f"   Missing: {', '.join(detail['missing_ids'])}")
            print()

    return {
        "agent_file": str(agent_file),
        "truth_file": str(truth_file),
        "filtering_stage": filtering_stage,
        "top_k": top_k,
        "agent_total_concept_ids": len(agent_concept_ids),
        "query_metrics": query_metrics,
    }


def calculate_batch_recall(
    agent_dir: Path,
    truth_file: Path,
    output_file: Path | None = None,
    top_k: int | None = None,
    filtering_stage: str = "after_filtering",
) -> Dict:
    """
    Calculate recall metrics for all agent output files in a directory.

    Calculates metrics separately for minimum=True and minimum=False decompositions.

    Args:
        agent_dir: Directory containing agent output JSON files
        truth_file: Path to truth set JSON
        output_file: Optional path to save results JSON
        top_k: Number of top concept IDs to extract per decomposition. If None, extract all.
        filtering_stage: Which stage to extract from:
            - "prior_filtering": Extract from all_collections_from_cmr (before any filtering)
            - "after_filtering": Extract from input_to_reranker (after collections_per_query filtering)
            - "after_ranking": Extract from data_results (after reranking, default)

    Returns:
        Dictionary with batch metrics
    """
    agent_files = list(Path(agent_dir).glob("*.json"))
    print(f"Found {len(agent_files)} agent output files")

    # remove the file that doesn't start with query_results_
    agent_files = [f for f in agent_files if f.name.startswith("query_results_")]

    all_results = []
    # Separate accumulators for minimum=True and minimum=False
    total_min_true_recall = 0
    total_min_true_coverage = 0
    total_min_false_recall = 0
    total_min_false_coverage = 0
    matched_queries = 0

    for agent_file in agent_files:
        # if not agent_file.name.startswith("query_results_Are_dry"):
        #     continue
        print(f"\n{'='*80}")
        print(f"Processing: {agent_file.name}")
        print(f"{'='*80}")

        try:
            result = calculate_overall_recall(
                agent_file,
                truth_file,
                verbose=True,
                top_k=top_k,
                filtering_stage=filtering_stage,
            )
            if result and result["query_metrics"]:
                all_results.append(result)
                # Accumulate metrics for both minimum=True and all_decomps
                min_true = result["query_metrics"]["minimum_true"]
                all_decomps = result["query_metrics"]["all_decomps"]

                total_min_true_recall += min_true["collection_recall"]
                total_min_true_coverage += min_true["decomposition_coverage"]
                total_min_false_recall += all_decomps["collection_recall"]
                total_min_false_coverage += all_decomps["decomposition_coverage"]

                matched_queries += 1
        except Exception as e:
            print(f"ERROR processing {agent_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate averages for both minimum=True and minimum=False
    if matched_queries > 0:
        avg_min_true_recall = total_min_true_recall / matched_queries
        avg_min_true_coverage = total_min_true_coverage / matched_queries
        avg_min_false_recall = total_min_false_recall / matched_queries
        avg_min_false_coverage = total_min_false_coverage / matched_queries
    else:
        avg_min_true_recall = 0
        avg_min_true_coverage = 0
        avg_min_false_recall = 0
        avg_min_false_coverage = 0

    batch_results = {
        "total_files": len(agent_files),
        "matched_queries": matched_queries,
        "filtering_stage": filtering_stage,
        "top_k": top_k,
        "minimum_true": {
            "average_collection_recall": avg_min_true_recall,
            "average_decomposition_coverage": avg_min_true_coverage,
        },
        "all_decomps": {
            "average_collection_recall": avg_min_false_recall,
            "average_decomposition_coverage": avg_min_false_coverage,
        },
        "per_query_results": all_results,
    }

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(agent_files)}")
    print(f"Successfully matched queries: {matched_queries}")
    print(f"\nMINIMUM=TRUE (only minimum=True decompositions):")
    print(f"  Average Collection Recall: {avg_min_true_recall:.1f}%")
    print(f"  Average Decomposition Coverage: {avg_min_true_coverage:.1f}%")
    print(f"\nALL DECOMPOSITIONS (both minimum=True and minimum=False):")
    print(f"  Average Collection Recall: {avg_min_false_recall:.1f}%")
    print(f"  Average Decomposition Coverage: {avg_min_false_coverage:.1f}%")

    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(batch_results, f, indent=2)
        print(f"\n✓ Saved batch results to {output_file}")

    return batch_results


# def main():
#     """Main entry point."""
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Calculate collection recall and decomposition coverage metrics"
#     )
#     parser.add_argument(
#         "agent",
#         type=Path,
#         help="Agent output file or directory",
#     )
#     parser.add_argument(
#         "truth_set",
#         type=Path,
#         help="Truth set JSON file",
#     )
#     parser.add_argument(
#         "--output",
#         "-o",
#         type=Path,
#         help="Output file for results JSON",
#     )
#     parser.add_argument(
#         "--include-non-minimum",
#         action="store_true",
#         help="Include decompositions with minimum=False in metrics",
#     )
#     parser.add_argument(
#         "--quiet",
#         action="store_true",
#         help="Suppress detailed output",
#     )

#     args = parser.parse_args()

#     only_minimum = not args.include_non_minimum

#     # Check if input is file or directory
#     if args.agent.is_file():
#         # Single file mode
#         result = calculate_overall_recall(
#             args.agent,
#             args.truth_set,
#             only_minimum,
#             verbose=not args.quiet,
#         )

#         if args.output and result:
#             args.output.parent.mkdir(parents=True, exist_ok=True)
#             with open(args.output, "w") as f:
#                 json.dump(result, f, indent=2)
#             print(f"\n✓ Saved results to {args.output}")

#     elif args.agent.is_dir():
#         # Batch mode
#         calculate_batch_recall(
#             args.agent,
#             args.truth_set,
#             only_minimum,
#             args.output,
#         )
#     else:
#         print(f"ERROR: {args.agent} is not a file or directory")
#         return


if __name__ == "__main__":
    
    """Calculate recall for a single query."""
    # Path to agent output (with data_results populated)
    # agent_dir = Path(
    #     "evaluations/evaluate_run_files/run_20251106_173040"
    # )

    # get top k from params 
    # code for getting top k from params in the command 

    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch CMR keywords, instruments, and science keywords",
    )

    parser.add_argument(
        "--top-k",
        type=str,
        default="5,10,20,all",
        help="Comma-separated list of k values to test (e.g., '5,10,20,all'). Use 'all' for no limit.",
    )


    # get agent_ dir and truth file from params
    
    parser.add_argument(
        "--agent-dir",
        type=str,
        default="evaluations/single_query_results/legacy_reranker_new_config",
        help="Path to directory with agent output files",
    )
    
    parser.add_argument(
        "--truth-file",
        type=str,
        default="evaluations/create_notebook/truth_set_20251027_deduplicated.json",
        help="Path to the truth set JSON file",
    )

    parser.add_argument(
        "--filtering-stage",
        type=str,
        default="after_filtering",
        choices=["prior_filtering", "after_filtering", "after_ranking"],
        help=(
            "Which filtering stage to use for concept extraction:\n"
            "  'prior_filtering': Use all_collections_from_cmr (before any filtering)\n"
            "  'after_filtering': Use input_to_reranker (after collections_per_query filtering, default)\n"
            "  'after_ranking': Use data_results (after reranking)"
        ),
    )

    args = parser.parse_args()

    k_values_str = args.top_k.split(",")
    k_values = []
    for k_str in k_values_str:
        k_str = k_str.strip()
        if k_str.lower() == "all":
            k_values.append(None)
        else:
            k_values.append(int(k_str))

    agent_dir = Path(args.agent_dir)

    # Path to truth set
    truth_file = Path(args.truth_file)

    print("Calculating recall metrics...")
    print(f"Agent output: {agent_dir}")
    print(f"Truth set: {truth_file}")
    print(f"K values: {k_values}")
    print(f"Filtering stage: {args.filtering_stage}\n")

    # Create output directory name based on agent_dir, truth_file, and filtering_stage
    agent_dir_name = agent_dir.name
    truth_file_stem = truth_file.stem  # filename without extension
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        f"evaluations/evaluate_run_files/recall_{agent_dir_name}_{truth_file_stem}_{args.filtering_stage}/{current_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Store all results
    all_k_results = {}

    for k in k_values:
        k_label = "all" if k is None else f"top{k}"
        print(f"\n{'='*80}")
        print(f"Computing recall for k={k_label}")
        print(f"{'='*80}\n")

        output_file = output_dir / f"recall_results_{k_label}.json"

        # Calculate metrics for both minimum=True and minimum=False
        result = calculate_batch_recall(
            agent_dir=agent_dir,
            truth_file=truth_file,
            output_file=output_file,
            top_k=k,
            filtering_stage=args.filtering_stage,
        )

        if result:
            all_k_results[k_label] = result

    # Save combined results to single file
    combined_output = output_dir / "recall_results_all_k_values.json"
    with open(combined_output, "w") as f:
        json.dump(all_k_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMBINED RESULTS")
    print(f"{'='*80}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Saved combined results to {combined_output}")

    # Print summary table with 4 columns
    print(f"\nSummary of all k values:")
    print(f"{'K Value':<12} {'Min=T ConceptIDs':<25} {'Min=T Decomps':<25} {'All ConceptIDs':<25} {'All Decomps':<25}")
    print(f"{'':12} {'Recall (%)':<25} {'Coverage (%)':<25} {'Recall (%)':<25} {'Coverage (%)':<25}")
    print(f"{'-'*112}")
    for k_label, result in all_k_results.items():
        min_true = result['minimum_true']
        all_decomps = result['all_decomps']
        print(
            f"{k_label:<12} "
            f"{min_true['average_collection_recall']:<25.2f} "
            f"{min_true['average_decomposition_coverage']:<25.2f} "
            f"{all_decomps['average_collection_recall']:<25.2f} "
            f"{all_decomps['average_decomposition_coverage']:<25.2f}"
        )


# if __name__ == "__main__":
#     main()

