#!/usr/bin/env python3
"""
Run evaluation queries from truth set and collect results.

Usage:
    uv run python evaluations/run_evaluation.py --model gpt-5-mini
    uv run python evaluations/run_evaluation.py --model gpt-5-mini --resume
    uv run python evaluations/run_evaluation.py --query-number 1  # Run specific query
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Import slugify from the agent's metadata module
sys.path.insert(0, str(Path(__file__).parent.parent))
from akd.agents.data_search.utils.metadata import slugify


def load_truth_set(filepath: str) -> Dict:
    """Load the truth set JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_state(filepath: str) -> Optional[Dict]:
    """Load existing evaluation state if it exists"""
    if Path(filepath).exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_evaluation_state(filepath: str, state: Dict):
    """Save evaluation state to JSON"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"💾 State saved to {filepath}")


def find_output_file(
    search_id_partial: str,
    captured_data_dir: str = "captured_data",
    run_subdir: Optional[str] = None,
) -> Optional[str]:
    """Find the output file for a given search based on partial search_id match"""
    if run_subdir:
        captured_path = Path(captured_data_dir) / run_subdir
    else:
        captured_path = Path(captured_data_dir)

    if not captured_path.exists():
        return None

    # Look for files matching the pattern
    for json_file in captured_path.glob("*.json"):
        if search_id_partial.lower() in json_file.stem.lower():
            return str(json_file)

    return None


def extract_metadata_from_output(output_file: str) -> Optional[Dict]:
    """Extract search metadata from the output JSON file"""
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        agent_output = data.get("agent_output", {})

        # Get metadata from agent_output.search_metadata
        search_metadata = agent_output.get("search_metadata", {})

        return {
            "search_id": search_metadata.get("search_id"),
            "duration_seconds": search_metadata.get("duration_seconds"),
            "topics_processed": search_metadata.get("topics_processed"),
            "total_cmr_results": agent_output.get("total_cmr_results"),
            "total_filtered_results": agent_output.get("total_filtered_results"),
        }
    except Exception as e:
        print(f"⚠️  Error reading output file {output_file}: {e}")
        return None


def run_query(
    query_text: str,
    model: str = "gpt-5-mini",
    output_subdir: Optional[str] = None,
    no_retrieve_all: bool = False,
    skip_ranking: bool = False,
) -> subprocess.CompletedProcess:
    """Run a single query using the CLI"""
    cmd = [
        "uv",
        "run",
        "akd/agents/data_search/cli.py",
        query_text,
        "--model",
        model,
    ]

    if output_subdir:
        cmd.extend(["--output-subdir", output_subdir])

    if no_retrieve_all:
        cmd.append("--no-retrieve-all")

    if skip_ranking:
        cmd.append("--skip-ranking")

    print(f'\n🚀 Running: {" ".join(cmd[:4])} "{query_text[:60]}..." --model {model}')
    if output_subdir:
        print(f"📁 Output subdirectory: {output_subdir}")
    if no_retrieve_all:
        print("⚡ Fast mode: Only retrieving first page per query")
    if skip_ranking:
        print(
            "🎯 Baseline mode: Skipping ranking/filtering - returning all deduplicated collections",
        )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    return result


def create_query_slug(query_text: str, max_length: int = 50) -> str:
    """Create a URL-safe slug from query text for matching output files

    Uses the same slugify function as the agent to ensure consistency.
    """
    # Use the agent's slugify function to match file naming logic
    return slugify(query_text, max_length=max_length)


async def run_evaluation(
    truth_set_file: str = "evaluations/truth_set_20251027.json",
    output_file: str | None = None,
    model: str = "gpt-5-mini",
    resume: bool = False,
    query_number: Optional[int] = None,
    query_range: Optional[str] = None,
    no_retrieve_all: bool = False,
    skip_ranking: bool = False,
):
    """Run evaluation on all queries in truth set"""

    # Load truth set
    print(f"📖 Loading truth set from {truth_set_file}")
    truth_set = load_truth_set(truth_set_file)
    queries = truth_set["queries"]
    print(f"Found {len(queries)} queries")

    # Initialize or load state
    if resume:
        # For resume, output_file must be specified or we can't find the previous state
        if not output_file:
            print("⚠️  --resume requires --output to specify which run to resume")
            return
        state = load_evaluation_state(output_file)
        if state:
            print(
                f"📥 Resuming from previous run ({state['queries_completed']}/{state['queries_total']} completed)",
            )
        else:
            print("⚠️  No previous state found, starting fresh")
            state = None
    else:
        state = None

    if not state:
        # Generate run subdirectory name from timestamp
        run_timestamp = datetime.now()
        run_subdir = f"run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Create output directory for this run
        run_dir = Path("evaluations") / run_subdir
        run_dir.mkdir(parents=True, exist_ok=True)

        # Set output_file to run subdirectory if not specified
        if not output_file:
            output_file = str(run_dir / "evaluation_runs.json")

        state = {
            "run_timestamp": run_timestamp.isoformat(),
            "run_subdir": run_subdir,
            "model": model,
            "queries_total": len(queries),
            "queries_completed": 0,
            "results": [],
        }
        print(f"📁 Run subdirectory: evaluations/{run_subdir}/")
        print(f"📁 Data subdirectory: captured_data/{run_subdir}/")
    else:
        run_subdir = state.get("run_subdir")

    # Filter queries if specific query number or range requested
    if query_number is not None:
        queries = [q for q in queries if q["query_number"] == query_number]
        if not queries:
            print(f"❌ Query number {query_number} not found in truth set")
            return
        print(f"🎯 Running single query: #{query_number}")
    elif query_range is not None:
        # Parse range like "11-20"
        try:
            start, end = map(int, query_range.split("-"))
            queries = [q for q in queries if start <= q["query_number"] <= end]
            if not queries:
                print(f"❌ No queries found in range {query_range}")
                return
            print(f"🎯 Running queries {start}-{end} ({len(queries)} queries)")
        except ValueError:
            print(f"❌ Invalid range format: {query_range}. Use format like '11-20'")
            return

    # Process each query
    for query in queries:
        query_num = query["query_number"]
        query_text = query["query_text"]

        # Check if already completed (when resuming)
        if resume and any(
            r["query_number"] == query_num and r["status"] == "completed"
            for r in state["results"]
        ):
            print(f"⏭️  Skipping query #{query_num} (already completed)")
            continue

        print(f"\n{'=' * 80}")
        print(f"Query #{query_num}/{len(truth_set['queries'])}")
        print(f"Query: {query_text}")
        print(f"{'=' * 80}")

        # Run the query
        try:
            result = run_query(
                query_text,
                model,
                run_subdir,
                no_retrieve_all,
                skip_ranking,
            )

            # Check if successful
            if result.returncode == 0:
                print("✅ Query completed successfully")

                # Try to find the output file
                query_slug = create_query_slug(query_text)
                output_file_path = find_output_file(query_slug, run_subdir=run_subdir)

                if output_file_path:
                    print(f"📄 Found output file: {output_file_path}")
                    metadata = extract_metadata_from_output(output_file_path)

                    result_entry = {
                        "query_number": query_num,
                        "query_text": query_text,
                        "output_file": output_file_path,
                        "status": "completed",
                        **metadata,
                    }
                else:
                    print("⚠️  Output file not found")
                    result_entry = {
                        "query_number": query_num,
                        "query_text": query_text,
                        "output_file": None,
                        "status": "completed_no_output",
                        "search_id": None,
                    }

            else:
                print(f"❌ Query failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr[:500]}")
                result_entry = {
                    "query_number": query_num,
                    "query_text": query_text,
                    "output_file": None,
                    "status": "failed",
                    "error": result.stderr[:1000],
                }

            # Update state
            # Remove any existing entry for this query number
            state["results"] = [
                r for r in state["results"] if r["query_number"] != query_num
            ]
            state["results"].append(result_entry)
            state["queries_completed"] = len(
                [
                    r
                    for r in state["results"]
                    if r["status"] in ["completed", "completed_no_output"]
                ],
            )

            # Save state after each query
            save_evaluation_state(output_file, state)

        except Exception as e:
            print(f"❌ Exception running query: {e}")
            result_entry = {
                "query_number": query_num,
                "query_text": query_text,
                "output_file": None,
                "status": "error",
                "error": str(e),
            }
            state["results"].append(result_entry)
            save_evaluation_state(output_file, state)

    # Final summary
    print(f"\n{'=' * 80}")
    print("📊 EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total queries: {state['queries_total']}")
    print(f"Completed: {state['queries_completed']}")
    failed = len([r for r in state["results"] if r["status"] == "failed"])
    errors = len([r for r in state["results"] if r["status"] == "error"])
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"\n💾 Final results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation queries from truth set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use for queries (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed queries)",
    )

    parser.add_argument(
        "--query-number",
        type=int,
        help="Run a specific query number only (for testing)",
    )

    parser.add_argument(
        "--query-range",
        type=str,
        help="Run a range of queries (e.g., '11-20' or '5-10')",
    )

    parser.add_argument(
        "--truth-set",
        default="evaluations/truth_set_20251027.json",
        help="Path to truth set JSON file",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to output results file (default: auto-create in evaluations/run_TIMESTAMP/)",
    )

    parser.add_argument(
        "--no-retrieve-all",
        action="store_true",
        help="Disable full CMR pagination (faster but incomplete - only retrieves first page per query)",
    )

    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip ranking/filtering - return all CMR collections (baseline mode)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            truth_set_file=args.truth_set,
            output_file=args.output,
            model=args.model,
            resume=args.resume,
            query_number=args.query_number,
            query_range=args.query_range,
            no_retrieve_all=args.no_retrieve_all,
            skip_ranking=args.skip_ranking,
        ),
    )


if __name__ == "__main__":
    main()
