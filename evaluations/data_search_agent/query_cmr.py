#!/usr/bin/env python3
"""
CMR Paginated Collection Search with Sorting Options

This script queries the CMR MCP endpoint with pagination support and configurable sorting.
It fetches all pages of results based on the query parameters.

Usage:
    python cmr_paginated_search.py --config config.json
    python cmr_paginated_search.py --sort relevance
    python cmr_paginated_search.py --sort usage --max-pages 5
    python cmr_paginated_search.py --input-json results.json --output-json updated_results.json
    python cmr_paginated_search.py --input-json results.json --output-json updated_results.json --use-rrf
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import httpx

from akd.structures import SearchResultItem
from akd.tools.search.utils import reciprocal_rank_fusion


def weighted_rrf_fusion(
    primary_results: list[SearchResultItem],
    secondary_results: list[SearchResultItem],
    primary_weight: float = 2.0,
    secondary_weight: float = 1.0,
    k: int = 60,
    keys: list[str] | str = "concept_id",
    value_normalizers: dict | None = None,
    debug: bool = False,
) -> list[SearchResultItem]:
    """
    Weighted RRF fusion that prioritizes primary list but boosts items appearing in secondary list.

    Args:
        primary_results: Primary ranked list (e.g., relevance-sorted)
        secondary_results: Secondary ranked list (e.g., usage-sorted)
        primary_weight: Weight multiplier for primary list scores (default: 2.0)
        secondary_weight: Weight multiplier for secondary list scores (default: 1.0)
        k: RRF k parameter
        keys: Deduplication keys
        value_normalizers: Custom normalizers for matching
        debug: Enable debug logging

    Returns:
        Fused and sorted results with weighted RRF scores

    Example:
        # Give relevance 2x weight, usage 1x weight
        fused = weighted_rrf_fusion(
            primary_results=search_results_relevance,
            secondary_results=search_results_usage,
            primary_weight=2.0,
            secondary_weight=1.0,
            k=60,
        )
    """
    # Duplicate primary list according to weight (simple approach)
    # For weight=2.0, include primary list twice
    primary_copies = [primary_results] * int(primary_weight)
    secondary_copies = [secondary_results] * int(secondary_weight)

    all_lists = primary_copies + secondary_copies

    return reciprocal_rank_fusion(
        *all_lists,
        k=k,
        keys=keys,
        value_normalizers=value_normalizers,
        debug=debug,
    )


async def call_mcp_tool(
    client: httpx.AsyncClient,
    base_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call an MCP tool via HTTP.

    Args:
        client: HTTP client
        base_url: Base URL for the CMR MCP endpoint
        tool_name: Name of the MCP tool
        arguments: Tool arguments

    Returns:
        Tool result as dictionary
    """
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    response = await client.post(
        base_url,
        json=request,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )

    if response.status_code == 200:
        # Parse SSE response
        content = response.text.strip()
        for line in content.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line.split("data: ", 1)[1])

                if "error" in data:
                    return {"error": data["error"]}

                # Parse the tool result
                tool_result = json.loads(data["result"]["content"][0]["text"])
                return tool_result

    return {"error": f"HTTP {response.status_code}: {response.text}"}


async def search_all_pages(
    base_url: str,
    query_params: dict[str, Any],
    sort_by: Literal["relevance", "usage"] = "relevance",
    max_pages: int | None = None,
    verbose: bool = True,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Search collections across all pages.

    Args:
        base_url: Base URL for the CMR MCP endpoint
        query_params: Query parameters (temporal, bounding_box, keyword, etc.)
        sort_by: Sorting strategy - "relevance" or "usage"
        max_pages: Maximum number of pages to fetch (None for all)
        verbose: Print progress information
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing all results and metadata
    """
    # Add sort_key based on sort_by parameter
    if sort_by == "usage":
        query_params["sort_key"] = "-usage_score"  # Descending usage score
    elif sort_by == "relevance":
        query_params["sort_key"] = "score"  # Relevance score

    all_collections = []
    total_hits = 0
    page_num = 1
    query_time_total = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            # Update page number
            query_params["page_num"] = page_num

            if verbose:
                print(f"\n📄 Fetching page {page_num}...")
                print(f"   Sort by: {sort_by}")
                if "sort_key" in query_params:
                    print(f"   Sort key: {query_params['sort_key']}")

            # Query the page
            result = await call_mcp_tool(client, base_url, "search_collections", query_params)

            # Handle errors
            if "error" in result:
                print(f"❌ Error on page {page_num}: {result['error']}")
                break

            # Extract metadata
            total_hits = result.get("total_hits", 0)
            query_time = result.get("query_time_ms", 0)
            query_time_total += query_time
            page_size = result.get("page_size", 0)
            collections = result.get("collections", [])

            if verbose:
                print(f"   ✅ Found {len(collections)} collections on this page")
                print(f"   ⏱️  Query time: {query_time}ms")

            # Add to results
            all_collections.extend(collections)

            # Check if we should continue
            if not collections:
                if verbose:
                    print(f"   ℹ️  No more results")
                break

            # Check max_pages limit
            if max_pages and page_num >= max_pages:
                if verbose:
                    print(f"   ⚠️  Reached max_pages limit ({max_pages})")
                break

            # Check if we've fetched all results
            if len(all_collections) >= total_hits:
                if verbose:
                    print(f"   ✅ Fetched all {total_hits} results")
                break

            page_num += 1

    return {
        "total_hits": total_hits,
        "total_fetched": len(all_collections),
        "total_pages": page_num,
        "query_time_total_ms": query_time_total,
        "sort_by": sort_by,
        "sort_key": query_params.get("sort_key"),
        "query_params": {
            k: v for k, v in query_params.items() if k not in ["page_num", "sort_key"]
        },
        "collections": all_collections,
    }


def convert_collections_to_search_results(
    collections: list[dict[str, Any]], sort_key: str = "relevance"
) -> list[SearchResultItem]:
    """Convert CMR collection dicts to SearchResultItem objects.

    Args:
        collections: List of collection dictionaries from CMR
        sort_key: Key to use for identifying the source ("relevance" or "usage")

    Returns:
        List of SearchResultItem objects
    """
    search_results = []
    for collection in collections:
        # Use concept_id as unique identifier
        concept_id = collection.get("concept_id", "")
        title = collection.get("entry_title") or collection.get("short_name", "")

        search_results.append(
            SearchResultItem(
                url=f"https://cmr.earthdata.nasa.gov/search/concepts/{concept_id}",
                title=title,
                content=collection.get("summary", ""),
                query=sort_key,
                extra={
                    "concept_id": concept_id,
                    "short_name": collection.get("short_name"),
                    "version": collection.get("version"),
                    "provider": collection.get("provider"),
                    "original_collection": collection,
                },
            )
        )
    return search_results


def convert_search_results_to_collections(
    search_results: list[SearchResultItem],
) -> list[dict[str, Any]]:
    """Convert SearchResultItem objects back to CMR collection dicts.

    Args:
        search_results: List of SearchResultItem objects

    Returns:
        List of collection dictionaries
    """
    collections = []
    for item in search_results:
        # Retrieve original collection from extra field
        if item.extra and "original_collection" in item.extra:
            collections.append(item.extra["original_collection"])
    return collections


async def process_json_queries(
    input_json_path: Path,
    output_json_path: Path,
    base_url: str,
    sort_by: Literal["relevance", "usage", "rrf"] = "relevance",
    use_rrf: bool = False,
    rrf_k: int = 60,
    verbose: bool = True,
    timeout: float = 30.0,
):
    """Process all searchable queries from a JSON file and update it with CMR results.

    Args:
        input_json_path: Path to input JSON file with searchable_queries
        output_json_path: Path to save updated JSON file
        base_url: Base URL for the CMR MCP endpoint
        sort_by: Sorting strategy - "relevance", "usage", or "rrf"
        use_rrf: If True, fetch both relevance and usage sorted results and fuse with RRF
        rrf_k: RRF k parameter (default 60)
        verbose: Print progress information
        timeout: Request timeout in seconds
    """
    # Load the input JSON
    with open(input_json_path) as f:
        data = json.load(f)

    print(f"\n📂 Loaded JSON from {input_json_path}")

    total_collections_overall = 0

    # Process each topic
    for topic_idx, topic in enumerate(data.get("topics", [])):
        topic_title = topic.get("topic", {}).get("title", "Unknown")
        print(f"\n{'='*60}")
        print(f"📌 Topic {topic_idx + 1}: {topic_title}")
        print(f"{'='*60}")

        topic_total_collections = 0

        # Process each decomposition
        for decomp_idx, decomp in enumerate(topic.get("decomposition_results", [])):
            decomp_title = decomp.get("decomposition", {}).get("title", "Unknown")
            print(f"\n  🔬 Decomposition {decomp_idx + 1}: {decomp_title}")

            searchable_queries = decomp.get("searchable_queries", [])
            all_collections = []

            # Process each searchable query
            for query_idx, query in enumerate(searchable_queries):
                mcp_params = query.get("mcp_parameters_sent", {})

                if not mcp_params:
                    print(f"    ⚠️  Query {query_idx + 1}: No MCP parameters, skipping")
                    continue

                keyword = mcp_params.get("keyword", "N/A")
                print(f"\n    🔍 Query {query_idx + 1}/{len(searchable_queries)}: keyword='{keyword}'")

                # Remove page_num from params to fetch all pages
                query_params = {k: v for k, v in mcp_params.items() if k != "page_num"}

                if use_rrf or sort_by == "rrf":
                    # Fetch results sorted by both relevance and usage
                    print(f"    🔀 Fetching relevance-sorted results...")
                    results_relevance = await search_all_pages(
                        base_url=base_url,
                        query_params=query_params,
                        sort_by="relevance",
                        max_pages=None,
                        verbose=verbose,
                        timeout=timeout,
                    )

                    print(f"    🔀 Fetching usage-sorted results...")
                    results_usage = await search_all_pages(
                        base_url=base_url,
                        query_params=query_params,
                        sort_by="usage",
                        max_pages=None,
                        verbose=verbose,
                        timeout=timeout,
                    )

                    # Convert to SearchResultItem objects
                    search_results_relevance = convert_collections_to_search_results(
                        results_relevance["collections"], "relevance"
                    )
                    search_results_usage = convert_collections_to_search_results(
                        results_usage["collections"], "usage"
                    )

                    # Apply weighted RRF fusion (relevance weight=3, usage weight=1)
                    print(f"    ⚡ Applying weighted RRF fusion (k={rrf_k}, relevance_weight=3, usage_weight=1)...")

                    # Custom normalizer to extract concept_id from extra dict
                    concept_id_normalizer = {
                        "concept_id": lambda item: item.extra.get("concept_id") if item.extra else None
                    }

                    fused_results = weighted_rrf_fusion(
                        primary_results=search_results_relevance,
                        secondary_results=search_results_usage,
                        primary_weight=5.0,
                        secondary_weight=1.0,
                        k=rrf_k,
                        keys=["concept_id"],
                        value_normalizers=concept_id_normalizer,
                        debug=verbose,
                    )

                    # Convert back to collections
                    collections = convert_search_results_to_collections(fused_results)

                    # Update cmr_collections_returned (use max of both)
                    query["cmr_collections_returned"] = max(
                        results_relevance["total_hits"], results_usage["total_hits"]
                    )
                    print(
                        f"    ✅ RRF fused {len(collections)} collections "
                        f"(relevance: {results_relevance['total_hits']}, usage: {results_usage['total_hits']})"
                    )

                    # Add collections to flat list
                    all_collections.extend(collections)

                else:
                    # Single sort mode (original behavior)
                    results = await search_all_pages(
                        base_url=base_url,
                        query_params=query_params,
                        sort_by=sort_by,
                        max_pages=None,
                        verbose=verbose,
                        timeout=timeout,
                    )

                    # Update cmr_collections_returned
                    query["cmr_collections_returned"] = results["total_hits"]
                    print(f"    ✅ Updated cmr_collections_returned: {results['total_hits']}")

                    # Add collections to flat list
                    all_collections.extend(results["collections"])

                # Remove non-MCP related fields from query
                keys_to_remove = [
                    "collections_to_ranking_pipeline",
                    "llm_reranker",
                    "reranker_score",
                    "enum_corrections",
                ]
                for key in keys_to_remove:
                    query.pop(key, None)

            # Update all_collections_from_cmr for this decomposition
            decomp["all_collections_from_cmr"] = all_collections
            decomp["total_cmr_collections"] = len(all_collections)
            topic_total_collections += len(all_collections)
            print(f"\n  📦 Decomposition total collections: {len(all_collections)}")

            # Remove non-MCP related fields from decomposition
            keys_to_remove = ["query_approaches", "repository"]
            for key in keys_to_remove:
                decomp.pop(key, None)

        # Add total collections for this topic
        topic["total_cmr_collections"] = topic_total_collections
        total_collections_overall += topic_total_collections
        print(f"\n  📊 Topic total collections: {topic_total_collections}")

    # Add overall total collections
    data["total_cmr_collections"] = total_collections_overall

    # Save updated JSON
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"💾 Updated JSON saved to {output_json_path}")
    print(f"📊 Overall total CMR collections: {total_collections_overall}")
    if use_rrf or sort_by == "rrf":
        print(f"🔀 RRF fusion enabled with k={rrf_k}")
    print(f"{'='*60}\n")


async def process_directory(
    input_dir: Path,
    output_dir: Path,
    base_url: str,
    sort_by: Literal["relevance", "usage", "rrf"] = "relevance",
    use_rrf: bool = False,
    rrf_k: int = 60,
    verbose: bool = True,
    timeout: float = 30.0,
):
    """Process all JSON files in a directory concurrently.

    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save output JSON files
        base_url: Base URL for the CMR MCP endpoint
        sort_by: Sorting strategy - "relevance", "usage", or "rrf"
        use_rrf: If True, fetch both relevance and usage sorted results and fuse with RRF
        rrf_k: RRF k parameter (default 60)
        verbose: Print progress information
        timeout: Request timeout in seconds
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in input directory
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"⚠️  No JSON files found in {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"📁 Found {len(json_files)} JSON files to process")
    print(f"{'='*60}\n")

    # Create tasks for all files
    tasks = []
    for idx, json_file in enumerate(json_files, 1):
        output_file = output_dir / json_file.name
        task = process_json_queries(
            input_json_path=json_file,
            output_json_path=output_file,
            base_url=base_url,
            sort_by=sort_by,
            use_rrf=use_rrf,
            rrf_k=rrf_k,
            verbose=verbose,
            timeout=timeout,
        )
        tasks.append((json_file.name, task))

    # Process all files concurrently
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    # Report results
    success_count = 0
    error_count = 0
    for (filename, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"❌ Error processing {filename}: {result}")
            error_count += 1
        else:
            print(f"✅ Successfully processed {filename}")
            success_count += 1

    print(f"\n{'='*60}")
    print(f"🎉 Completed processing {len(json_files)} files")
    print(f"✅ Success: {success_count}, ❌ Errors: {error_count}")
    print(f"📂 Output directory: {output_dir}")
    print(f"{'='*60}\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CMR paginated collection search with sorting options"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file with query parameters",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Path to input JSON file with searchable_queries to process",
    )
    parser.add_argument(
        "--input-dir",
        default = "evaluations/single_query_results/llm_reranker_new_config", 
        type=Path,
        help="Path to directory containing JSON files to process",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to save updated JSON file (required with --input-json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory (required with --input-dir)",
    )
    parser.add_argument(
        "--sort",
        choices=["relevance", "usage", "rrf"],
        default="rrf",
        help="Sort by relevance, data usage, or RRF fusion (default: relevance)",
    )
    parser.add_argument(
        "--use-rrf",
        action="store_true",
        help="Enable RRF fusion of relevance and usage sorted results",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k parameter (default: 60)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to fetch (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/mcp/cmr/mcp/",
        help="CMR MCP endpoint base URL",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Check if processing directory mode
    if args.input_dir:
        # output diris inside the input dir 
        if not args.output_dir:
            args.output_dir = args.input_dir / "cmr_paginated_search_outputs" / f"sorted_by_{args.sort}"
            if not args.output_dir.exists():
                args.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"No output directory specified. Using: {args.output_dir}\n")

        print(f"🚀 Processing directory mode")
        await process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            base_url=args.base_url,
            sort_by=args.sort,
            use_rrf=args.use_rrf,
            rrf_k=args.rrf_k,
            verbose=not args.quiet,
        )
        return

    # Check if processing JSON file mode
    if args.input_json:
        if not args.output_json:
            parser.error("--output-json is required when using --input-json")

        print(f"🚀 Processing JSON file mode")
        await process_json_queries(
            input_json_path=args.input_json,
            output_json_path=args.output_json,
            base_url=args.base_url,
            sort_by=args.sort,
            use_rrf=args.use_rrf,
            rrf_k=args.rrf_k,
            verbose=not args.quiet,
        )
        return

    # Load query parameters from config or use defaults
    if args.config and args.config.exists():
        with open(args.config) as f:
            query_params = json.load(f)
        print(f"📁 Loaded config from {args.config}")
    else:
        # Default query parameters
        query_params = {
            "temporal": "2003-01-01T00:00:00Z,2024-12-31T23:59:59Z",
            "bounding_box": "112.0,-28.0,154.0,-8.0",
            "keyword": "soil moisture",
            "page_size": 20,
        }
        print("📋 Using default query parameters")

    print("\n🔍 Query Parameters:")
    for key, value in query_params.items():
        print(f"   {key}: {value}")
    print()

    # Execute search
    print(f"🚀 Starting paginated search (sort by: {args.sort})")
    print("=" * 60)

    results = await search_all_pages(
        base_url=args.base_url,
        query_params=query_params,
        sort_by=args.sort,
        max_pages=args.max_pages,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Search Summary")
    print("=" * 60)
    print(f"   Total hits in CMR: {results['total_hits']:,}")
    print(f"   Total fetched: {results['total_fetched']:,}")
    print(f"   Pages fetched: {results['total_pages']}")
    print(f"   Total query time: {results['query_time_total_ms']:,}ms")
    print(f"   Sort strategy: {results['sort_by']}")
    print(f"   Sort key used: {results['sort_key']}")
    print()

    # Show sample results
    if results["collections"]:
        print("📚 Sample Collections (first 5):")
        for i, collection in enumerate(results["collections"][:5], 1):
            print(f"\n   {i}. {collection.get('short_name')} v{collection.get('version', 'N/A')}")
            print(f"      Title: {collection.get('entry_title', 'N/A')}")
            print(f"      Provider: {collection.get('provider', 'N/A')}")
            print(f"      Concept ID: {collection.get('concept_id', 'N/A')}")

    # Save results if output path provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to {args.output}")
    else:
        print("\n Tip: Use --output to save results to a JSON file")


if __name__ == "__main__":
    print("=" * 60)
    print("🛰️  CMR Paginated Collection Search")
    print("=" * 60)
    print("\nMake sure the FastAPI server is running:")
    print("  uv run uvicorn app:app --host 0.0.0.0 --port 8080")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n  Search interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
