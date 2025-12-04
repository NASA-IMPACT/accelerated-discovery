#!/usr/bin/env python3
"""
Deduplicate concept IDs within each query in the truth set.

This script creates a deduplicated version of the truth set where, within a single query,
each concept ID appears only once across all minimum=true decompositions.

This ensures consistent counting when measuring recall/coverage metrics.

Usage:
    uv run python evaluations/deduplicate_truth_set.py
"""

import json
from pathlib import Path
from typing import Dict


def deduplicate_query_concepts(query: Dict) -> Dict:
    """
    Deduplicate concept IDs WITHIN each decomposition only.

    Does NOT deduplicate across decompositions - if the same concept ID appears
    in multiple decompositions within a query, it is kept in all of them.

    Only removes duplicates within the same decomposition's concept ID list.

    Args:
        query: Query dictionary from truth set

    Returns:
        Modified query with deduplicated concept IDs (within each decomp)
    """
    query_copy = json.loads(json.dumps(query))  # Deep copy

    for topic in query_copy.get("topics", []):
        for decomp in topic.get("decomps", []):
            # Process all decomps, not just minimum=true
            # Get current concept IDs
            concept_ids = decomp.get("cmr_concept_ids", [])

            # Deduplicate within this decomposition only (preserve order)
            seen = set()
            unique_concepts = []
            duplicates_found = []

            for cid in concept_ids:
                if cid not in seen:
                    unique_concepts.append(cid)
                    seen.add(cid)
                else:
                    duplicates_found.append(cid)

            # Update decomposition with deduplicated IDs
            decomp["cmr_concept_ids"] = unique_concepts

            # Add note if we removed duplicates within this decomp
            if duplicates_found:
                dup_str = ", ".join(duplicates_found)
                if "notes" in decomp:
                    decomp["notes"] += f" (Removed within-decomp duplicates: {dup_str})"
                else:
                    decomp["notes"] = f"Removed within-decomp duplicates: {dup_str}"

    return query_copy


def deduplicate_truth_set(input_path: Path, output_path: Path):
    """
    Create deduplicated version of truth set.

    Args:
        input_path: Path to original truth set
        output_path: Path to write deduplicated truth set
    """
    print(f"Loading truth set from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        truth_set = json.load(f)

    # Statistics
    total_concepts_before = 0
    total_concepts_after = 0
    queries_affected = 0

    # Process each query
    deduplicated_queries = []
    for query in truth_set.get("queries", []):
        # Count before (including duplicates)
        concepts_before = []
        for topic in query.get("topics", []):
            for decomp in topic.get("decomps", []):
                if decomp.get("minimum", False):
                    concepts_before.extend(decomp.get("cmr_concept_ids", []))

        # Deduplicate
        deduped_query = deduplicate_query_concepts(query)

        # Count after (including any remaining duplicates)
        concepts_after = []
        for topic in deduped_query.get("topics", []):
            for decomp in topic.get("decomps", []):
                if decomp.get("minimum", False):
                    concepts_after.extend(decomp.get("cmr_concept_ids", []))

        # Track changes
        before_count = len(concepts_before)
        after_count = len(concepts_after)
        total_concepts_before += before_count
        total_concepts_after += after_count

        if before_count != after_count:
            queries_affected += 1
            print(f"  Query {query['query_number']}: {before_count} → {after_count} concepts")
            print(f"    {query['query_text'][:60]}...")

        deduplicated_queries.append(deduped_query)

    # Create output
    output_data = {
        "queries": deduplicated_queries,
        "_metadata": {
            "source": "Deduplicated version of truth_set_20251027.json",
            "deduplication_note": "Concept IDs are unique WITHIN each decomposition only. Does NOT deduplicate across decompositions.",
            "original_file": str(input_path),
            "total_queries": len(deduplicated_queries),
            "queries_affected": queries_affected,
            "total_minimum_concepts_before": total_concepts_before,
            "total_minimum_concepts_after": total_concepts_after,
            "duplicates_removed": total_concepts_before - total_concepts_after,
        },
    }

    # Write output
    print(f"\nWriting deduplicated truth set to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("DEDUPLICATION SUMMARY")
    print("=" * 70)
    print(f"Total queries: {len(deduplicated_queries)}")
    print(f"Queries affected: {queries_affected}")
    print(f"Minimum=true concepts before: {total_concepts_before}")
    print(f"Minimum=true concepts after: {total_concepts_after}")
    print(f"Duplicates removed: {total_concepts_before - total_concepts_after}")
    print("=" * 70)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent

    input_path = script_dir / "truth_set_20251027.json"
    output_path = script_dir / "truth_set_20251027_deduplicated.json"

    if not input_path.exists():
        print(f"ERROR: Input truth set not found: {input_path}")
        return 1

    deduplicate_truth_set(input_path, output_path)
    print("\n✓ Deduplication complete!")
    return 0


if __name__ == "__main__":
    exit(main())
