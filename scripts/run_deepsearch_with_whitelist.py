#!/usr/bin/env python3
"""
Run DeepLitSearchAgent and compare results with and without ISSN-based whitelist validation.

This script:
- Executes a focused deep search for a given query
- Converts resulting links to SearchResultItem objects
- Validates sources via CrossRef and the ISSN whitelist
- Generates a combined Markdown report including both views
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Continue if python-dotenv is not installed; env vars may already be set
    pass

from akd.agents.search.deep_search import DeepLitSearchAgent, DeepLitSearchAgentConfig
from akd.structures import SearchResultItem
from akd.tools.source_validator import (
    SourceValidator,
    SourceValidatorInputSchema,
    create_source_validator,
)


def build_agent(fast: bool) -> DeepLitSearchAgent:
    # Configure a lightweight run to keep runtime practical for reporting
    config = DeepLitSearchAgentConfig(
        max_research_iterations=1 if fast else 3,
        use_semantic_scholar=False if fast else True,
        enable_per_link_assessment=False,
        enable_full_content_scraping=False,
        min_relevancy_score=0.3,
        full_content_threshold=0.7,
        enable_streaming=False,
    )
    return DeepLitSearchAgent(config=config, debug=False)


async def run_deep_search(query: str, fast: bool) -> List[Dict[str, str]]:
    # Fallback early if API keys are unavailable
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            from examples.source_validation_test import create_sample_search_results

            sample_items = create_sample_search_results()
            return [
                {
                    "url": str(it.url),
                    "title": it.title,
                    "content": it.content,
                    "category": it.category or "science",
                }
                for it in sample_items
            ]
        except Exception:
            pass

    agent = build_agent(fast)
    try:
        output = await agent.arun(agent.input_schema(query=query, category="science"))
        results: List[Dict[str, str]] = []
        for item in output.results:
            url = item.get("url")
            # Skip synthetic report record and non-http(s) URLs
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            results.append(item)
        return results
    except Exception:
        # Fall back to sample items if search tools are not available
        try:
            from examples.source_validation_test import create_sample_search_results

            sample_items = create_sample_search_results()
            return [
                {
                    "url": str(it.url),
                    "title": it.title,
                    "content": it.content,
                    "category": it.category or "science",
                }
                for it in sample_items
            ]
        except Exception:
            return []


def to_search_items(
    results: List[Dict[str, str]], query: str
) -> List[SearchResultItem]:
    items: List[SearchResultItem] = []
    for r in results:
        try:
            item = SearchResultItem(
                url=r.get("url", ""),
                title=r.get("title", "Untitled"),
                query=query,
                content=r.get("content", ""),
                category=r.get("category", "science"),
            )
            items.append(item)
        except Exception:
            # Ignore entries that fail URL validation, etc.
            continue
    return items


async def validate_with_whitelist(
    validator: SourceValidator,
    items: List[SearchResultItem],
) -> Tuple[List[Dict], Dict[str, float]]:
    if not items:
        return [], {
            "total": 0,
            "whitelisted": 0,
            "errors": 0,
            "whitelisted_pct": 0.0,
        }

    result = await validator.arun(SourceValidatorInputSchema(search_results=items))
    whitelisted: List[Dict] = []
    for item, vr in zip(items, result.validated_results):
        if vr.is_whitelisted:
            whitelisted.append(
                {
                    "title": item.title,
                    "url": str(item.url),
                    "matched_issn": vr.matched_issn,
                    "category": vr.whitelist_category,
                    "confidence": vr.confidence_score,
                }
            )

    stats = {
        "total": result.summary.get("total_processed", 0),
        "whitelisted": result.summary.get("whitelisted_count", 0),
        "errors": result.summary.get("error_count", 0),
        "whitelisted_pct": result.summary.get("whitelisted_percentage", 0.0),
    }
    return whitelisted, stats


def render_report(
    query: str,
    raw_results: List[Dict[str, str]],
    whitelisted_results: List[Dict],
    stats: Dict[str, float],
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append("# Deep Search Report\n")
    lines.append(f"- **Query**: {query}")
    lines.append(f"- **Generated at (UTC)**: {ts}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total results (raw)**: {len(raw_results)}")
    lines.append(
        f"- **Whitelisted results**: {int(stats.get('whitelisted', 0))} ({stats.get('whitelisted_pct', 0.0):.1f}%)"
    )
    lines.append(f"- **Validation errors**: {int(stats.get('errors', 0))}")
    lines.append("")
    lines.append("## Without whitelist (top 20)")
    for i, r in enumerate(raw_results[:20], start=1):
        lines.append(f"{i}. [{r.get('title', 'Untitled')}]({r.get('url', '')})")
    lines.append("")
    lines.append("## With whitelist (all)")
    if not whitelisted_results:
        lines.append("No results passed the ISSN whitelist.")
    else:
        for i, r in enumerate(whitelisted_results, start=1):
            meta: List[str] = []
            if r.get("category"):
                meta.append(f"Category: {r['category']}")
            if r.get("matched_issn"):
                meta.append(f"ISSN: {r['matched_issn']}")
            meta.append(f"Confidence: {r.get('confidence', 0.0):.2f}")
            meta_str = " | ".join(meta)
            lines.append(f"{i}. [{r['title']}]({r['url']}) — {meta_str}")
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep search with ISSN whitelist comparison"
    )
    parser.add_argument("query", type=str, help="Research query to run")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster, lower-cost search configuration",
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default=None,
        help="Path to docs/issn_whitelist.json (optional)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/deepsearch_whitelist_report.md",
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    # Step 1: Run deep search
    raw_results = await run_deep_search(args.query, args.fast)

    # Step 2: Convert to SearchResultItem for validation
    items = to_search_items(raw_results, args.query)

    # Step 3: Validate with ISSN whitelist
    validator = create_source_validator(
        whitelist_file_path=args.whitelist,
        timeout_seconds=25,
        max_concurrent_requests=8,
        debug=False,
    )
    whitelisted_results, stats = await validate_with_whitelist(validator, items)

    # Step 4: Render combined report
    report_text = render_report(args.query, raw_results, whitelisted_results, stats)

    # Step 5: Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to: {out_path}")
    print(f"Raw results: {len(raw_results)} | Whitelisted: {len(whitelisted_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
