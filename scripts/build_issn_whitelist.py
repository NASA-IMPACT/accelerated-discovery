#!/usr/bin/env python3
"""
Build an ISSN-based whitelist from the existing journal-name whitelist using CrossRef.

Inputs:
- docs/pubs_whitelist.json (name-based categories)

Outputs:
- docs/issn_whitelist.json (categories → ISSN list)
- docs/issn_whitelist_map.json (category → {journal name → [ISSNs...]})

Notes:
- Uses CrossRef works search (query.container-title) to retrieve ISSNs.
- Normalizes ISSNs to NNNN-NNNN format and de-duplicates.
- Best effort; some journals may not resolve cleanly due to naming ambiguity.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBS_WHITELIST = PROJECT_ROOT / "docs" / "pubs_whitelist.json"
ISSN_WHITELIST = PROJECT_ROOT / "docs" / "issn_whitelist.json"
ISSN_MAP = PROJECT_ROOT / "docs" / "issn_whitelist_map.json"


def normalize_issn(issn: str) -> Optional[str]:
    if issn is None:
        return None
    candidate = re.sub(r"[^0-9xX]", "", str(issn)).upper()
    if len(candidate) != 8:
        return None
    return f"{candidate[:4]}-{candidate[4:]}"


async def fetch_issns_for_journal(
    session: aiohttp.ClientSession,
    journal_name: str,
    max_items: int = 5,
) -> List[str]:
    # CrossRef works search by container title
    base_url = "https://api.crossref.org/works"
    params = {
        "query.container-title": journal_name,
        "rows": str(max_items),
        "select": "ISSN,issn-type,container-title,title",
    }
    headers = {
        "User-Agent": "AKD-ISSN-Builder/1.0 (mailto:research@example.org)",
        "Accept": "application/json",
    }

    # Build query string manually
    from urllib.parse import urlencode

    url = f"{base_url}?{urlencode(params)}"
    try:
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            items = data.get("message", {}).get("items", [])
            seen: Set[str] = set()
            for item in items:
                # ISSN array
                issn_values = []
                if isinstance(item.get("ISSN"), list):
                    issn_values.extend([str(v) for v in item.get("ISSN", [])])
                elif item.get("ISSN"):
                    issn_values.append(str(item.get("ISSN")))
                # issn-type objects
                issn_type = item.get("issn-type", [])
                if isinstance(issn_type, list):
                    for it in issn_type:
                        value = it.get("value") if isinstance(it, dict) else None
                        if value:
                            issn_values.append(str(value))
                for raw in issn_values:
                    norm = normalize_issn(raw)
                    if norm is not None:
                        seen.add(norm)
            return sorted(seen)
    except Exception:
        return []


async def build_whitelist() -> Tuple[
    Dict[str, List[str]], Dict[str, Dict[str, List[str]]]
]:
    if not PUBS_WHITELIST.exists():
        raise FileNotFoundError(f"Missing whitelist file: {PUBS_WHITELIST}")

    with open(PUBS_WHITELIST, "r", encoding="utf-8") as f:
        whitelist_data = json.load(f)

    categories = whitelist_data.get("data", {})
    if not isinstance(categories, dict):
        raise ValueError(
            "Unexpected pubs_whitelist.json format: 'data' must be an object"
        )

    # Prepare tasks
    category_to_journals: Dict[str, List[str]] = {}
    for category, cat_data in categories.items():
        journals = cat_data.get("journals", []) if isinstance(cat_data, dict) else []
        names: List[str] = []
        for j in journals:
            if isinstance(j, dict):
                name = (j.get("Journal Name") or "").strip()
                if name:
                    names.append(name)
        category_to_journals[category] = names

    # Fetch concurrently
    connector = aiohttp.TCPConnector(limit=12)
    async with aiohttp.ClientSession(connector=connector) as session:
        category_to_issn: Dict[str, List[str]] = {}
        category_to_map: Dict[str, Dict[str, List[str]]] = {}

        for category, journal_names in category_to_journals.items():
            category_to_map[category] = {}
            tasks = [fetch_issns_for_journal(session, name) for name in journal_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            issn_set: Set[str] = set()
            for name, res in zip(journal_names, results):
                issns = res if isinstance(res, list) else []
                category_to_map[category][name] = issns
                for issn in issns:
                    issn_set.add(issn)
            category_to_issn[category] = sorted(issn_set)

    return category_to_issn, category_to_map


def main() -> int:
    print("Building ISSN whitelist from journal-name whitelist...")
    try:
        category_to_issn, category_to_map = asyncio.run(build_whitelist())

        metadata = {
            "source": str(PUBS_WHITELIST),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        issn_whitelist_obj = {"categories": category_to_issn, "metadata": metadata}
        issn_map_obj = {"categories": category_to_map, "metadata": metadata}

        with open(ISSN_WHITELIST, "w", encoding="utf-8") as f:
            json.dump(issn_whitelist_obj, f, indent=2, ensure_ascii=False)
        with open(ISSN_MAP, "w", encoding="utf-8") as f:
            json.dump(issn_map_obj, f, indent=2, ensure_ascii=False)

        total_categories = len(category_to_issn)
        total_issn = sum(len(v) for v in category_to_issn.values())
        print(f"Wrote {total_issn} unique ISSNs across {total_categories} categories")
        print(f"Whitelist: {ISSN_WHITELIST}")
        print(f"Journal→ISSN map: {ISSN_MAP}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
