#!/usr/bin/env python3
"""
Build ISSN whitelist from journal mapping data.

This script converts journal name/publisher data into ISSN-based whitelists
by looking up ISSNs using the CrossRef API or manual mapping files.

Usage:
    python scripts/build_issn_whitelist.py --input docs/pubs_whitelist.json --output docs/issn_whitelist.json
    python scripts/build_issn_whitelist.py --map docs/issn_whitelist_map.json --output docs/issn_whitelist.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

import aiohttp
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from akd.tools.source_validator import SourceValidator


class ISSNWhitelistBuilder:
    """Builder for ISSN whitelists from journal data."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.session: Optional[aiohttp.ClientSession] = None
        self.issn_cache: Dict[str, List[str]] = {}
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def lookup_journal_issn(self, journal_name: str, publisher: str = "") -> List[str]:
        """
        Look up ISSNs for a journal using CrossRef API.
        
        Args:
            journal_name: Name of the journal
            publisher: Publisher name (optional, for better matching)
            
        Returns:
            List of normalized ISSNs found
        """
        cache_key = f"{journal_name.lower()}:{publisher.lower()}"
        if cache_key in self.issn_cache:
            return self.issn_cache[cache_key]
        
        if not self.session:
            raise RuntimeError("Session not initialized - use as async context manager")
            
        # Build search query
        query_parts = [journal_name]
        if publisher:
            query_parts.append(publisher)
        query = " ".join(query_parts)
        
        url = "https://api.crossref.org/journals"
        params = {
            "query": query,
            "rows": 5,
            "select": "title,publisher,ISSN"
        }
        
        headers = {
            "User-Agent": "ISSNWhitelistBuilder/1.0 (mailto:research@example.org)",
            "Accept": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("message", {}).get("items", [])
                    
                    # Find best match by title similarity
                    best_match = self._find_best_journal_match(journal_name, items)
                    if best_match:
                        issns = best_match.get("ISSN", [])
                        normalized_issns = []
                        for issn in issns:
                            normalized = SourceValidator._normalize_issn(str(issn))
                            if normalized:
                                normalized_issns.append(normalized)
                        
                        self.issn_cache[cache_key] = normalized_issns
                        if self.debug:
                            logger.info(f"Found ISSNs for '{journal_name}': {normalized_issns}")
                        return normalized_issns
                    
        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to lookup ISSNs for '{journal_name}': {e}")
        
        # Cache empty result to avoid repeated lookups
        self.issn_cache[cache_key] = []
        return []
    
    def _find_best_journal_match(self, target_title: str, crossref_items: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching journal from CrossRef results.
        
        Args:
            target_title: Target journal title
            crossref_items: List of CrossRef journal items
            
        Returns:
            Best matching item or None
        """
        if not target_title or not crossref_items:
            return None
            
        target_lower = target_title.lower().strip()
        best_match = None
        best_score = 0
        
        for item in crossref_items:
            item_title = item.get("title", "").lower().strip()
            if not item_title:
                continue
                
            # Exact match
            if target_lower == item_title:
                return item
            
            # Calculate word overlap score
            target_words = set(target_lower.split())
            item_words = set(item_title.split())
            
            if len(target_words) == 0 or len(item_words) == 0:
                continue
                
            overlap = len(target_words & item_words)
            total_words = len(target_words | item_words)
            
            if total_words > 0:
                score = overlap / total_words
                if score > best_score and score > 0.5:  # Minimum 50% overlap
                    best_score = score
                    best_match = item
        
        return best_match if best_score > 0.5 else None
    
    async def build_from_journal_data(self, journal_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build ISSN whitelist from journal name/publisher data.
        
        Args:
            journal_data: Journal data with categories and journal lists
            
        Returns:
            Dictionary mapping categories to ISSN lists
        """
        issn_whitelist = {}
        
        if "data" in journal_data:
            for category_name, category_info in journal_data["data"].items():
                if self.debug:
                    logger.info(f"Processing category: {category_name}")
                    
                category_issns = []
                journals = category_info.get("journals", [])
                
                for journal in journals:
                    if not isinstance(journal, dict):
                        continue
                        
                    journal_name = journal.get("Journal Name", "").strip()
                    publisher = journal.get("Publisher / Society", "").strip()
                    
                    if not journal_name:
                        continue
                    
                    if self.debug:
                        logger.info(f"Looking up ISSNs for: {journal_name}")
                    
                    issns = await self.lookup_journal_issn(journal_name, publisher)
                    category_issns.extend(issns)
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                
                # Remove duplicates while preserving order
                unique_issns = []
                seen = set()
                for issn in category_issns:
                    if issn not in seen:
                        unique_issns.append(issn)
                        seen.add(issn)
                
                issn_whitelist[category_name] = unique_issns
                if self.debug:
                    logger.info(f"Category '{category_name}': {len(unique_issns)} unique ISSNs")
        
        return issn_whitelist
    
    def build_from_issn_map(self, issn_map_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build ISSN whitelist from ISSN mapping data.
        
        Args:
            issn_map_data: ISSN mapping data with categories
            
        Returns:
            Dictionary mapping categories to ISSN lists
        """
        issn_whitelist = {}
        
        if "issn_map" in issn_map_data:
            # Group ISSNs by category
            category_groups = {}
            for issn, info in issn_map_data["issn_map"].items():
                category = info.get("category", "Unknown")
                if category not in category_groups:
                    category_groups[category] = []
                
                # Normalize ISSN
                normalized = SourceValidator._normalize_issn(issn)
                if normalized:
                    category_groups[category].append(normalized)
            
            # Remove duplicates in each category
            for category, issns in category_groups.items():
                unique_issns = list(dict.fromkeys(issns))  # Preserves order
                issn_whitelist[category] = unique_issns
                
                if self.debug:
                    logger.info(f"Category '{category}': {len(unique_issns)} ISSNs")
        
        return issn_whitelist


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build ISSN whitelist from journal data")
    parser.add_argument("--input", help="Input journal data file (JSON)")
    parser.add_argument("--map", help="Input ISSN mapping file (JSON)")
    parser.add_argument("--output", required=True, help="Output ISSN whitelist file (JSON)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if not args.input and not args.map:
        parser.error("Either --input or --map must be specified")
    
    if args.input and args.map:
        parser.error("Only one of --input or --map can be specified")
    
    # Configure logging
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    try:
        async with ISSNWhitelistBuilder(debug=args.debug) as builder:
            if args.input:
                # Build from journal data
                logger.info(f"Loading journal data from: {args.input}")
                with open(args.input, 'r', encoding='utf-8') as f:
                    journal_data = json.load(f)
                
                logger.info("Building ISSN whitelist from journal data...")
                issn_whitelist = await builder.build_from_journal_data(journal_data)
                
            else:
                # Build from ISSN mapping
                logger.info(f"Loading ISSN mapping from: {args.map}")
                with open(args.map, 'r', encoding='utf-8') as f:
                    issn_map_data = json.load(f)
                
                logger.info("Building ISSN whitelist from ISSN mapping...")
                issn_whitelist = builder.build_from_issn_map(issn_map_data)
            
            # Create output structure
            output_data = {
                "metadata": {
                    "description": "ISSN-based journal whitelist for source validation",
                    "source_file": args.input or args.map,
                    "build_method": "journal_lookup" if args.input else "issn_mapping",
                    "total_categories": len(issn_whitelist),
                    "total_issns": sum(len(issns) for issns in issn_whitelist.values())
                },
                "categories": issn_whitelist
            }
            
            # Write output
            logger.info(f"Writing ISSN whitelist to: {args.output}")
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Print summary
            total_issns = output_data["metadata"]["total_issns"]
            total_categories = output_data["metadata"]["total_categories"]
            
            logger.info(f"✅ Built ISSN whitelist: {total_issns} ISSNs across {total_categories} categories")
            
            for category, issns in issn_whitelist.items():
                logger.info(f"  {category}: {len(issns)} ISSNs")
            
    except Exception as e:
        logger.error(f"Failed to build ISSN whitelist: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)