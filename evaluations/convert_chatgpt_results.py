#!/usr/bin/env python3
"""
Convert ChatGPT results CSV to JSON format.

ChatGPT was queried with the same evaluation questions using different prompt
engineering approaches. This script converts the CSV results (which came from an
Excel file with merged cells) into a structured JSON format for comparison against
our data search agent's results.

This is NOT a truth set - it represents ChatGPT's baseline performance for comparison.

CSV Structure:
- Column 0: SME name (merged across all rows for that SME's queries)
- Column 1: Prompt type (Basic/Angles) (merged across query groups)
- Column 2: MCP (Yes/No) (merged across query groups)
- Column 3: Query text (merged across all decomps for that query)
- Column 4: Inferred Decomp (unique per row)
- Column 5: Comma Separated Datasets (CMR concept IDs, unique per row)
- Column 6: Conversation Link (optional)
- Column 7: Additional notes (optional)
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List


def extract_concept_ids(dataset_str: str) -> List[str]:
    """
    Extract CMR concept IDs from a comma-separated string.

    Handles various formats:
    - Standard format: C1234567890-PROVIDER
    - Lowercase (converts to uppercase)
    - Extra whitespace
    - Trailing commas

    Args:
        dataset_str: Comma-separated string of concept IDs

    Returns:
        List of cleaned, deduplicated concept IDs in order
    """
    if not dataset_str or not dataset_str.strip():
        return []

    # Pattern matches: C + digits + dash + uppercase letters/underscores
    # We'll also match lowercase and convert them
    pattern = r"[Cc]\d+(?:-[A-Za-z_]+)?"

    matches = re.findall(pattern, dataset_str)

    # Clean up: uppercase, strip, deduplicate while preserving order
    seen = set()
    cleaned = []
    for match in matches:
        # Uppercase the entire match
        concept_id = match.upper().strip()
        if concept_id and concept_id not in seen:
            seen.add(concept_id)
            cleaned.append(concept_id)

    return cleaned


def parse_csv_with_merged_cells(csv_path: Path) -> List[Dict]:
    """
    Parse CSV file that originated from Excel with merged cells.

    When cells are merged in Excel and exported to CSV:
    - Only the first row contains the value
    - Subsequent rows have empty strings for that column
    - We need to "forward fill" from above (except for concept IDs)

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of row dictionaries with filled values
    """
    rows = []

    # Track the last non-empty value for each column that should be filled
    # Columns that forward-fill: 0 (SME), 1 (Prompt), 2 (MCP), 3 (Query)
    # Columns that DON'T forward-fill: 4 (Decomp), 5 (Datasets), 6 (Link), 7 (Notes)
    last_values = {
        "sme": "",
        "prompt_type": "",
        "mcp": "",
        "query": "",
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Skip header row
        next(reader)

        for row_num, row in enumerate(reader, start=2):
            # Ensure we have at least 8 columns (pad if needed)
            while len(row) < 8:
                row.append("")

            # Column 0: SME name (forward fill, normalize to title case)
            if row[0].strip():
                sme_raw = row[0].strip().title()
                # Fix known typos
                if sme_raw == "Emyl":
                    sme_raw = "Emily"
                last_values["sme"] = sme_raw
            # Use last value even if current row is empty

            # Column 1: Prompt type (forward fill)
            if row[1].strip():
                last_values["prompt_type"] = row[1].strip()

            # Column 2: MCP (forward fill)
            if row[2].strip():
                last_values["mcp"] = row[2].strip()

            # Column 3: Query text (forward fill)
            if row[3].strip():
                last_values["query"] = row[3].strip()

            # Column 4: Decomp (DO NOT forward fill - each row unique)
            decomp = row[4].strip()

            # Column 5: Datasets (DO NOT forward fill - each row unique)
            datasets_str = row[5].strip()
            concept_ids = extract_concept_ids(datasets_str)

            # Column 6: Conversation link
            link = row[6].strip() if len(row) > 6 else ""

            # Column 7: Notes
            notes = row[7].strip() if len(row) > 7 else ""

            # Create row dict with filled values
            row_dict = {
                "sme": last_values["sme"],
                "prompt_type": last_values["prompt_type"],
                "mcp": last_values["mcp"],
                "query": last_values["query"],
                "decomp": decomp,
                "concept_ids": concept_ids,
                "conversation_link": link,
                "notes": notes,
                "row_num": row_num,
            }

            rows.append(row_dict)

    return rows


def group_rows_into_queries(rows: List[Dict]) -> List[Dict]:
    """
    Group parsed rows into query structures.

    Each query is identified by a unique combination of:
    - SME name
    - Prompt type
    - MCP setting
    - Query text

    Within each query, decompositions are grouped.

    Args:
        rows: List of parsed row dictionaries

    Returns:
        List of query dictionaries with nested decompositions
    """
    queries = []
    current_query = None

    for row in rows:
        # Create query key
        query_key = (
            row["sme"],
            row["prompt_type"],
            row["mcp"],
            row["query"],
        )

        # Check if we're starting a new query
        if current_query is None or current_query["_key"] != query_key:
            # Save previous query if exists
            if current_query is not None:
                queries.append(current_query)

            # Start new query
            current_query = {
                "_key": query_key,  # Internal key for grouping
                "sme": row["sme"],
                "prompt_type": row["prompt_type"],
                "mcp": row["mcp"] == "Yes",
                "query_text": row["query"],
                "conversation_link": row["conversation_link"]
                if row["conversation_link"]
                else None,
                "decompositions": [],
            }

        # Add decomposition to current query
        if row["decomp"]:  # Only add if decomp is not empty
            decomp_dict = {
                "decomp": row["decomp"],
                "cmr_concept_ids": row["concept_ids"],
            }

            # Add optional fields if present
            if row["notes"]:
                decomp_dict["notes"] = row["notes"]

            current_query["decompositions"].append(decomp_dict)

    # Don't forget the last query
    if current_query is not None:
        queries.append(current_query)

    # Clean up internal keys
    for query in queries:
        del query["_key"]

    return queries


def add_query_numbers(queries: List[Dict]) -> List[Dict]:
    """
    Add sequential query numbers to each query.

    Args:
        queries: List of query dictionaries

    Returns:
        Same list with query_number field added
    """
    for i, query in enumerate(queries, start=1):
        query["query_number"] = i

    return queries


def convert_chatgpt_csv_to_json(csv_path: Path, output_path: Path) -> Dict:
    """
    Convert ChatGPT results CSV to JSON format.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSON file

    Returns:
        Dictionary with conversion statistics
    """
    print(f"Reading CSV from: {csv_path}")
    rows = parse_csv_with_merged_cells(csv_path)

    print(f"Parsed {len(rows)} rows")
    print("Grouping into queries...")

    queries = group_rows_into_queries(rows)
    queries = add_query_numbers(queries)

    # Calculate statistics
    total_decomps = sum(len(q["decompositions"]) for q in queries)
    total_concepts = sum(
        len(d["cmr_concept_ids"]) for q in queries for d in q["decompositions"]
    )

    # Create output structure
    output = {
        "metadata": {
            "source": "ChatGPT baseline results",
            "description": "ChatGPT results for comparison against our data search agent. NOT ground truth.",
            "total_queries": len(queries),
            "total_decompositions": total_decomps,
            "total_concept_ids": total_concepts,
            "smes": sorted(list(set(q["sme"] for q in queries))),
            "prompt_types": sorted(list(set(q["prompt_type"] for q in queries))),
        },
        "queries": queries,
    }

    # Write output
    print(f"Writing JSON to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print statistics
    stats = {
        "queries": len(queries),
        "decompositions": total_decomps,
        "concept_ids": total_concepts,
        "smes": output["metadata"]["smes"],
        "prompt_types": output["metadata"]["prompt_types"],
    }

    print("\n" + "=" * 60)
    print("Conversion Statistics:")
    print("=" * 60)
    print(f"Total Queries: {stats['queries']}")
    print(f"Total Decompositions: {stats['decompositions']}")
    print(f"Total Concept IDs: {stats['concept_ids']}")
    print(f"SMEs: {', '.join(stats['smes'])}")
    print(f"Prompt Types: {', '.join(stats['prompt_types'])}")
    print("=" * 60)

    return stats


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent

    # Input/output paths
    csv_path = script_dir / "DataAgentQueries - ChatGPT.csv"
    output_path = script_dir / "chatgpt_results.json"

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    try:
        convert_chatgpt_csv_to_json(csv_path, output_path)
        print(f"\n✓ Successfully converted to: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
