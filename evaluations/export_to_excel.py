#!/usr/bin/env python3
"""
Export evaluation results to Excel format for manual review.

Converts captured_data JSON files to hierarchical Excel spreadsheet with merged cells.

Usage:
    uv run python evaluations/export_to_excel.py --runs evaluations/run_20251027_213136/evaluation_runs.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_datetime(iso_string: Optional[str]) -> str:
    """Format ISO datetime string to readable format."""
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return str(iso_string) if iso_string else ""


def get_bounding_box(decomposition_result: Dict[str, Any]) -> str:
    """Extract first non-null bounding box from searchable queries."""
    searchable_queries = decomposition_result.get("searchable_queries", [])
    for query in searchable_queries:
        bbox = query.get("bounding_box")
        if bbox:
            return bbox
    return ""


def construct_cmr_url(concept_id: str) -> str:
    """Construct CMR concept URL from concept ID."""
    if not concept_id:
        return ""
    return f"https://cmr.earthdata.nasa.gov/search/concepts/{concept_id}"


def extract_rows_from_json(json_file: Path) -> List[Dict[str, Any]]:
    """
    Extract flat row data from a single JSON output file.

    Only extracts the top 5 collections per decomposition (per SCORING.md evaluation criteria).

    Returns list of row dictionaries with fields:
    - query, topic, decomp, justification, cmr_title, start, end, bbox, link
    - merge_metadata: {query_count, topic_count, decomp_count}
    """
    data = load_json(json_file)
    agent_output = data.get("agent_output", {})

    # Get query text from search metadata
    search_metadata = agent_output.get("search_metadata", {})
    query_text = search_metadata.get("original_query", "")

    rows = []
    topics = agent_output.get("topics", [])

    for topic_data in topics:
        topic_info = topic_data.get("topic", {})
        topic_title = topic_info.get("title", "")

        decomposition_results = topic_data.get("decomposition_results", [])

        for decomp_result in decomposition_results:
            decomp_info = decomp_result.get("decomposition", {})
            decomp_title = decomp_info.get("title", "")
            decomp_justification = decomp_info.get("scientific_justification", "")

            # Get bounding box for this decomposition
            bbox = get_bounding_box(decomp_result)

            # Get data results (CMR collections) - top 5 only per SCORING.md
            data_results = decomp_result.get("data_results", [])
            top_5_results = data_results[:5]

            for dataset in top_5_results:
                row = {
                    "query": query_text,
                    "topic": topic_title,
                    "decomp": decomp_title,
                    "justification": decomp_justification,
                    "cmr_title": dataset.get("entry_title", ""),
                    "start": format_datetime(dataset.get("time_start")),
                    "end": format_datetime(dataset.get("time_end")),
                    "bbox": bbox,
                    "link": construct_cmr_url(dataset.get("concept_id", "")),
                    "bol": "",
                    "comment": "",
                    "missing_perfect": "",
                }
                rows.append(row)

    return rows


def calculate_merge_ranges(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate merge ranges for hierarchical cells.

    Adds merge_ranges to each row:
    - query_start, query_end (span all rows with same query)
    - topic_start, topic_end (span all rows with same topic within query)
    - decomp_start, decomp_end (span all rows with same decomp within topic)
    """
    if not rows:
        return []

    # Track current groupings
    current_query = None
    current_topic = None
    current_decomp = None

    query_start = 0
    topic_start = 0
    decomp_start = 0

    result_rows = []

    for i, row in enumerate(rows):
        # Check if query changed
        if row["query"] != current_query:
            # Finalize previous decomp merge ranges before changing query
            if current_decomp is not None:
                for j in range(decomp_start, i):
                    result_rows[j]["decomp_merge"] = (decomp_start + 2, i + 1)

            # Finalize previous topic merge ranges before changing query
            if current_topic is not None:
                for j in range(topic_start, i):
                    result_rows[j]["topic_merge"] = (topic_start + 2, i + 1)

            # Finalize previous query merge ranges
            if current_query is not None:
                for j in range(query_start, i):
                    result_rows[j]["query_merge"] = (
                        query_start + 2,
                        i + 1,
                    )  # +2 for header row

            current_query = row["query"]
            query_start = i
            current_topic = None  # Reset topic
            topic_start = i  # Reset topic start position
            current_decomp = None  # Reset decomp
            decomp_start = i  # Reset decomp start position

        # Check if topic changed
        if row["topic"] != current_topic:
            # Finalize previous decomp merge ranges before changing topic
            if current_decomp is not None:
                for j in range(decomp_start, i):
                    result_rows[j]["decomp_merge"] = (decomp_start + 2, i + 1)

            # Finalize previous topic merge ranges
            if current_topic is not None:
                for j in range(topic_start, i):
                    result_rows[j]["topic_merge"] = (topic_start + 2, i + 1)

            current_topic = row["topic"]
            topic_start = i
            current_decomp = None  # Reset decomp
            decomp_start = i  # Reset decomp start position

        # Check if decomp changed
        if row["decomp"] != current_decomp:
            # Finalize previous decomp merge ranges
            if current_decomp is not None:
                for j in range(decomp_start, i):
                    result_rows[j]["decomp_merge"] = (decomp_start + 2, i + 1)

            current_decomp = row["decomp"]
            decomp_start = i

        result_rows.append(row)

    # Finalize last groups
    total_rows = len(rows)
    for j in range(query_start, total_rows):
        result_rows[j]["query_merge"] = (query_start + 2, total_rows + 1)
    for j in range(topic_start, total_rows):
        result_rows[j]["topic_merge"] = (topic_start + 2, total_rows + 1)
    for j in range(decomp_start, total_rows):
        result_rows[j]["decomp_merge"] = (decomp_start + 2, total_rows + 1)

    return result_rows


def create_excel_file(
    rows: List[Dict[str, Any]],
    output_file: Path,
    run_date: str,
) -> None:
    """Create Excel file with hierarchical merged cells."""
    wb = Workbook()
    ws = wb.active

    # Set sheet name to "results - YYYY-MM-DD" using run date
    ws.title = f"results - {run_date}"

    # Define column headers
    headers = [
        "",  # A - empty/index
        "SME",  # B
        "Query",  # C
        "Topic",  # D
        "Decomp",  # E
        "",  # F - Scientific justification (no header label)
        "CMR Title",  # G
        "Start",  # H
        "End",  # I
        "Bbox",  # J
        "Link",  # K
        "Bol",  # L
        "Comment",  # M
        "Missing Perfect",  # N
    ]

    # Write header row
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Calculate merge ranges
    rows_with_merges = calculate_merge_ranges(rows)

    # Write data rows
    for row_idx, row_data in enumerate(rows_with_merges, start=2):
        # Column A - empty/index
        ws.cell(row=row_idx, column=1, value="")

        # Column B - SME
        cell = ws.cell(row=row_idx, column=2, value=row_data.get("sme", ""))
        cell.alignment = Alignment(horizontal="center", vertical="top")

        # Column C - Query
        cell = ws.cell(row=row_idx, column=3, value=row_data["query"])
        cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Column D - Topic
        cell = ws.cell(row=row_idx, column=4, value=row_data["topic"])
        cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Column E - Decomp
        cell = ws.cell(row=row_idx, column=5, value=row_data["decomp"])
        cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Column F - Justification
        cell = ws.cell(row=row_idx, column=6, value=row_data["justification"])
        cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Column G - CMR Title
        cell = ws.cell(row=row_idx, column=7, value=row_data["cmr_title"])
        cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Column H - Start
        ws.cell(row=row_idx, column=8, value=row_data["start"])

        # Column I - End
        ws.cell(row=row_idx, column=9, value=row_data["end"])

        # Column J - Bbox
        ws.cell(row=row_idx, column=10, value=row_data["bbox"])

        # Column K - Link
        cell = ws.cell(row=row_idx, column=11, value=row_data["link"])
        cell.style = "Hyperlink"

        # Columns L-N - Empty for manual annotation
        ws.cell(row=row_idx, column=12, value=row_data["bol"])
        ws.cell(row=row_idx, column=13, value=row_data["comment"])
        ws.cell(row=row_idx, column=14, value=row_data["missing_perfect"])

    # Apply merged cells
    # Track which ranges we've already merged to avoid duplicates
    merged_query_ranges = set()
    merged_topic_ranges = set()
    merged_decomp_ranges = set()

    for row_data in rows_with_merges:
        # Merge SME (Column B) and Query (Column C) - same range
        query_range = row_data.get("query_merge")
        if query_range and query_range not in merged_query_ranges:
            start_row, end_row = query_range
            if end_row > start_row:
                ws.merge_cells(f"B{start_row}:B{end_row}")  # SME
                ws.merge_cells(f"C{start_row}:C{end_row}")  # Query
            merged_query_ranges.add(query_range)

        # Merge Topic (Column D)
        topic_range = row_data.get("topic_merge")
        if topic_range and topic_range not in merged_topic_ranges:
            start_row, end_row = topic_range
            if end_row > start_row:
                ws.merge_cells(f"D{start_row}:D{end_row}")
            merged_topic_ranges.add(topic_range)

        # Merge Decomp (Column E) and Justification (Column F)
        decomp_range = row_data.get("decomp_merge")
        if decomp_range and decomp_range not in merged_decomp_ranges:
            start_row, end_row = decomp_range
            if end_row > start_row:
                ws.merge_cells(f"E{start_row}:E{end_row}")
                ws.merge_cells(f"F{start_row}:F{end_row}")
            merged_decomp_ranges.add(decomp_range)

    # Set column widths
    column_widths = {
        "A": 5,  # Index
        "B": 10,  # SME
        "C": 50,  # Query
        "D": 30,  # Topic
        "E": 30,  # Decomp
        "F": 50,  # Justification
        "G": 50,  # CMR Title
        "H": 12,  # Start
        "I": 12,  # End
        "J": 25,  # Bbox
        "K": 60,  # Link
        "L": 8,  # Bol
        "M": 30,  # Comment
        "N": 15,  # Missing Perfect
    }

    for col_letter, width in column_widths.items():
        ws.column_dimensions[col_letter].width = width

    # Save workbook
    wb.save(output_file)
    print(f"✅ Excel file created: {output_file}")


def export_evaluation_to_excel(
    runs_file: Path,
    output_file: Optional[Path] = None,
    truth_set_file: Optional[Path] = None,
) -> None:
    """
    Export evaluation results to Excel format.

    Args:
        runs_file: Path to evaluation_runs.json
        output_file: Path to output Excel file (default: same directory as runs_file)
        truth_set_file: Path to truth_set JSON file (default: evaluations/truth_set_20251027.json)
    """
    print(f"📖 Loading evaluation runs from {runs_file}")
    runs_data = load_json(runs_file)

    # Load truth set to get SME information
    if truth_set_file is None:
        truth_set_file = Path("evaluations/truth_set_20251027.json")

    if truth_set_file.exists():
        print(f"📖 Loading truth set from {truth_set_file}")
        truth_data = load_json(truth_set_file)
        # Create lookup: query_number -> SME
        sme_lookup = {
            q["query_number"]: q["sme"] for q in truth_data.get("queries", [])
        }
    else:
        print(f"⚠️  Truth set not found at {truth_set_file}, SME column will be empty")
        sme_lookup = {}

    # Set default output file if not specified
    if output_file is None:
        output_file = runs_file.parent / "results_export.xlsx"

    # Get run subdirectory info and date
    run_subdir = runs_data.get("run_subdir")
    run_timestamp = runs_data.get("run_timestamp", "")

    # Extract date (YYYY-MM-DD) from run_timestamp (ISO format)
    try:
        run_dt = datetime.fromisoformat(run_timestamp)
        run_date = run_dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        # Fallback to today if timestamp is invalid
        run_date = datetime.now().strftime("%Y-%m-%d")

    print(f"📁 Run subdirectory: {run_subdir}")
    print(f"📅 Run date: {run_date}")

    # Process all completed queries
    results = runs_data.get("results", [])
    completed = [r for r in results if r.get("status") == "completed"]

    print(f"Found {len(completed)} completed queries out of {len(results)} total")

    if not completed:
        print("❌ No completed queries to export")
        return

    # Extract rows from all JSON files
    all_rows = []
    for result in completed:
        query_num = result.get("query_number")
        output_file_path = result.get("output_file")

        if not output_file_path:
            print(f"⚠️  Skipping query {query_num}: no output file")
            continue

        json_path = Path(output_file_path)
        if not json_path.exists():
            print(f"⚠️  Skipping query {query_num}: file not found: {json_path}")
            continue

        print(f"Processing query {query_num}: {json_path.name}")

        # Get SME for this query
        sme = sme_lookup.get(query_num, "")

        try:
            rows = extract_rows_from_json(json_path)
            # Add SME to each row
            for row in rows:
                row["sme"] = sme
            print(f"  Extracted {len(rows)} rows (SME: {sme})")
            all_rows.extend(rows)
        except Exception as e:
            print(f"❌ Error processing query {query_num}: {e}")
            continue

    if not all_rows:
        print("❌ No rows extracted from any queries")
        return

    print(f"\n📊 Total rows to export: {len(all_rows)}")

    # Create Excel file
    print("📝 Creating Excel file...")
    create_excel_file(all_rows, output_file, run_date)

    print("\n✅ Export complete!")
    print(f"   Output: {output_file}")
    print(f"   Total rows: {len(all_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export evaluation results to Excel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Path to evaluation_runs.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output Excel file (default: same directory as runs file)",
    )

    parser.add_argument(
        "--truth-set",
        type=Path,
        default=None,
        help="Path to truth set JSON file (default: evaluations/truth_set_20251027.json)",
    )

    args = parser.parse_args()

    # Validate runs file exists
    if not args.runs.exists():
        print(f"❌ Error: Runs file not found: {args.runs}")
        return 1

    export_evaluation_to_excel(args.runs, args.output, args.truth_set)
    return 0


if __name__ == "__main__":
    exit(main())
