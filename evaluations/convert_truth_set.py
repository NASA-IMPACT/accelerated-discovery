#!/usr/bin/env python3
"""
Convert truth_set_20251027.csv to a clean JSON structure.

Structure: Query → Topics → Decomps → CMR Concept IDs
"""

import csv
import json
import re
from collections import defaultdict
from typing import Any, Dict, List


def extract_cmr_ids(text: str) -> List[str]:
    """Extract CMR concept IDs using regex pattern C[0-9]+-[A-Z_]+"""
    if not text:
        return []

    # Pattern for CMR IDs: C followed by digits, hyphen, uppercase letters/underscores
    pattern = r"C\d+-[A-Z_]+"
    matches = re.findall(pattern, text)

    # Return unique IDs while preserving order
    seen = set()
    result = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            result.append(match)

    return result


def parse_minimum(value: str) -> bool:
    """Parse the Minimum column to boolean"""
    if not value:
        return False
    return value.strip().lower() in ["yes", "true", "1"]


def read_csv_with_multiline(filepath: str) -> List[Dict[str, str]]:
    """Read CSV handling multi-line entries"""
    with open(filepath, "r", encoding="utf-8") as f:
        # Use csv.reader to handle quoted fields properly
        reader = csv.reader(f)
        headers = next(reader)

        rows = []
        current_row = None

        for row in reader:
            if len(row) < len(headers):
                # Pad short rows
                row.extend([""] * (len(headers) - len(row)))

            # Check if this is a new entry (has a query number)
            query_num = row[0].strip() if row[0] else ""

            if query_num and query_num.isdigit():
                # New entry
                if current_row:
                    rows.append(current_row)
                current_row = dict(zip(headers, row))
            elif current_row:
                # Continuation of previous entry - merge text fields
                for i, (header, value) in enumerate(zip(headers, row)):
                    if value.strip():
                        if current_row[header]:
                            current_row[header] += "\n" + value
                        else:
                            current_row[header] = value

        # Don't forget the last row
        if current_row:
            rows.append(current_row)

        return rows


def build_tree_structure(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build the hierarchical structure: Query → Topics → Decomps → CMR IDs"""

    queries_dict = defaultdict(
        lambda: {
            "query_number": None,
            "sme": None,
            "query_text": None,
            "topics": defaultdict(
                lambda: defaultdict(
                    lambda: {
                        "decomp": None,
                        "minimum": False,
                        "cmr_concept_ids": [],
                    },
                ),
            ),
        },
    )

    for row in rows:
        query_num = int(row["Query S. No."].strip())
        sme = row["SME"].strip()
        query_text = row["Query"].strip()
        topic = row["Topic"].strip()
        decomp = row["Decomps"].strip()
        minimum = parse_minimum(row["Minimum"])

        # Extract CMR IDs from CMR Concept Id column only (ignore Possible Matches)
        cmr_ids = extract_cmr_ids(row["CMR Concept Id"])

        # Build the tree
        query_entry = queries_dict[query_num]
        query_entry["query_number"] = query_num
        query_entry["sme"] = sme
        query_entry["query_text"] = query_text

        # Add to decomp (multiple rows might contribute to same decomp)
        decomp_entry = query_entry["topics"][topic][decomp]
        decomp_entry["decomp"] = decomp
        decomp_entry["minimum"] = (
            minimum or decomp_entry["minimum"]
        )  # Keep true if any says true

        # Merge CMR IDs
        existing_ids = set(decomp_entry["cmr_concept_ids"])
        for cid in cmr_ids:
            if cid not in existing_ids:
                decomp_entry["cmr_concept_ids"].append(cid)
                existing_ids.add(cid)

    # Convert nested dicts to clean list structure
    result = {"queries": []}

    for query_num in sorted(queries_dict.keys()):
        query_entry = queries_dict[query_num]

        topics_list = []
        for topic_name in query_entry["topics"]:
            decomps_list = []
            for decomp_name in query_entry["topics"][topic_name]:
                decomp_entry = query_entry["topics"][topic_name][decomp_name]
                decomps_list.append(
                    {
                        "decomp": decomp_entry["decomp"],
                        "minimum": decomp_entry["minimum"],
                        "cmr_concept_ids": decomp_entry["cmr_concept_ids"],
                    },
                )

            topics_list.append(
                {
                    "topic": topic_name,
                    "decomps": decomps_list,
                },
            )

        result["queries"].append(
            {
                "query_number": query_entry["query_number"],
                "sme": query_entry["sme"],
                "query_text": query_entry["query_text"],
                "topics": topics_list,
            },
        )

    return result


def main():
    input_file = "truth_set_20251027.csv"
    output_file = "truth_set_20251027.json"

    print(f"Reading {input_file}...")
    rows = read_csv_with_multiline(input_file)
    print(f"Found {len(rows)} rows")

    print("Building tree structure...")
    result = build_tree_structure(rows)
    print(f"Created {len(result['queries'])} queries")

    print(f"Writing to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Done!")

    # Print summary
    total_topics = sum(len(q["topics"]) for q in result["queries"])
    total_decomps = sum(
        len(t["decomps"]) for q in result["queries"] for t in q["topics"]
    )
    print("\nSummary:")
    print(f"  Queries: {len(result['queries'])}")
    print(f"  Topics: {total_topics}")
    print(f"  Decomps: {total_decomps}")


if __name__ == "__main__":
    main()
