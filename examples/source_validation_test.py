#!/usr/bin/env python3
"""
Example script demonstrating the journal validation pipeline.

This example shows how to:
1. Search for research papers using SearxNG and Semantic Scholar
2. Extract DOIs from the search results
3. Validate journals against a controlled whitelist using CrossRef API
4. Generate validation reports

Usage:
    python examples/journal_validation_pipeline.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from akd.structures import SearchResultItem
from akd.tools.factory import create_default_search_tool
from akd.tools.source_validator import create_source_validator

# Import SemanticScholarSearchTool separately since it's not in factory
try:
    from akd.tools.search import SemanticScholarSearchTool
except ImportError:
    SemanticScholarSearchTool = None

# Sample search queries for Earth Sciences
EARTH_SCIENCE_QUERIES = [
    "climate change modeling earth system",
    "atmospheric physics greenhouse gases",
    "carbon cycle biogeochemistry",
    "remote sensing earth observation",
    "geophysics seismic analysis",
]

# Sample search queries for Astronomy
ASTRONOMY_QUERIES = [
    "exoplanet detection transit photometry",
    "stellar evolution supernova",
    "galactic formation dark matter",
    "black hole accretion disk",
    "cosmic ray particle physics",
]


class JournalValidationPipeline:
    """
    Complete pipeline for searching, extracting, and validating research sources.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the validation pipeline.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

        # Initialize search tools
        self.searxng_tool = create_default_search_tool()

        # Initialize Semantic Scholar tool (requires API key in environment)
        try:
            if SemanticScholarSearchTool:
                self.semantic_scholar_tool = SemanticScholarSearchTool()
            else:
                self.semantic_scholar_tool = None
        except Exception as e:
            print(f"Warning: Could not initialize Semantic Scholar tool: {e}")
            self.semantic_scholar_tool = None

        # Initialize source validator
        self.source_validator = create_source_validator(debug=debug)

    async def search_multiple_sources(
        self, queries: List[str], max_results_per_query: int = 5
    ) -> List[SearchResultItem]:
        """
        Search multiple sources and combine results.

        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query per source

        Returns:
            Combined list of search results
        """
        all_results = []

        # Search using SearxNG
        try:
            searxng_input = {
                "queries": queries,
                "category": "science",
                "max_results": max_results_per_query * len(queries),
            }
            searxng_results = await self.searxng_tool.arun(searxng_input)
            all_results.extend(searxng_results.results)

            if self.debug:
                print(f"SearxNG found {len(searxng_results.results)} results")

        except Exception as e:
            print(f"Error with SearxNG search: {e}")

        # Search using Semantic Scholar (if available)
        if self.semantic_scholar_tool:
            try:
                semantic_input = {
                    "queries": queries,
                    "max_results": max_results_per_query * len(queries),
                }
                semantic_results = await self.semantic_scholar_tool.arun(semantic_input)
                all_results.extend(semantic_results.results)

                if self.debug:
                    print(
                        f"Semantic Scholar found {len(semantic_results.results)} results"
                    )

            except Exception as e:
                print(f"Error with Semantic Scholar search: {e}")

        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()

        for result in all_results:
            url_str = str(result.url)
            if url_str not in seen_urls:
                seen_urls.add(url_str)
                unique_results.append(result)

        if self.debug:
            print(f"Total unique results after deduplication: {len(unique_results)}")

        return unique_results

    async def validate_search_results(
        self, search_results: List[SearchResultItem]
    ) -> dict:
        """
        Validate search results against journal whitelist.

        Args:
            search_results: List of search results to validate

        Returns:
            Validation results with summary
        """
        validation_input = {"search_results": search_results}

        validation_results = await self.source_validator.arun(validation_input)

        return {
            "results": validation_results.validated_results,
            "summary": validation_results.summary,
        }

    def print_validation_report(self, validation_data: dict):
        """
        Print a comprehensive validation report.

        Args:
            validation_data: Validation results and summary
        """
        results = validation_data["results"]
        summary = validation_data["summary"]

        print("\n" + "=" * 80)
        print("JOURNAL VALIDATION REPORT")
        print("=" * 80)

        print("\nSUMMARY STATISTICS:")
        print(f"  Total papers processed: {summary['total_processed']}")
        print(f"  Whitelisted papers: {summary['whitelisted_count']}")
        print(f"  Whitelist success rate: {summary['whitelisted_percentage']:.1f}%")
        print(f"  Papers with errors: {summary['error_count']}")
        print(f"  Average confidence: {summary['avg_confidence']:.2f}")

        if summary.get("category_breakdown"):
            print("\nCATEGORY BREAKDOWN:")
            for category, count in summary["category_breakdown"].items():
                print(f"  {category}: {count} papers")

        print("\nDETAILED RESULTS:")
        print("-" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. VALIDATION RESULT")

            if result.validation_errors:
                print("   Status: ❌ FAILED")
                print(f"   Errors: {', '.join(result.validation_errors)}")
            elif result.is_whitelisted:
                print("   Status: ✅ WHITELISTED")
                print(f"   Category: {result.whitelist_category}")
                print(f"   Confidence: {result.confidence_score:.2f}")
            else:
                print("   Status: ⚠️ NOT WHITELISTED")
                print(f"   Confidence: {result.confidence_score:.2f}")

            if result.source_info:
                ji = result.source_info
                print(f"   Venue: {ji.title}")
                print(f"   Publisher: {ji.publisher or 'Unknown'}")
                print(f"   DOI: {ji.doi}")
                print(f"   Open Access: {ji.is_open_access or 'Unknown'}")
                print(f"   Original URL: {ji.url}")
                if ji.issn:
                    print(f"   ISSN: {', '.join(ji.issn)}")
            if result.matched_issn:
                print(f"   Matched ISSN: {result.matched_issn}")

        print("\n" + "=" * 80)

    async def run_full_pipeline(
        self, queries: List[str], max_results_per_query: int = 3
    ):
        """
        Run the complete validation pipeline.

        Args:
            queries: Search queries to process
            max_results_per_query: Maximum results per query per source
        """
        print(f"Starting validation pipeline with {len(queries)} queries...")
        print(f"Search queries: {queries}")

        # Step 1: Search multiple sources
        print("\n1. Searching multiple sources...")
        search_results = await self.search_multiple_sources(
            queries, max_results_per_query
        )

        if not search_results:
            print("No search results found. Pipeline stopped.")
            return

        print(f"Found {len(search_results)} unique search results")

        # Step 2: Validate against whitelist
        print("\n2. Validating against journal whitelist...")
        validation_data = await self.validate_search_results(search_results)

        # Step 3: Generate report
        print("\n3. Generating validation report...")
        self.print_validation_report(validation_data)


class EnhancedJournalValidationPipeline(JournalValidationPipeline):
    """
    Enhanced pipeline with CrossRef-based DOI resolution for indirect links.
    """

    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.crossref_base_url = "https://api.crossref.org/works"

    async def _extract_metadata_from_url(self, url: str) -> dict:
        """
        Extract bibliographic metadata from webpage using web scraper.

        Args:
            url: URL to extract metadata from

        Returns:
            Dictionary with title, author, publication_date, journal, etc.
        """
        try:
            # Import web scraper tools
            from akd.tools.scrapers.web_scrapers import (
                SimpleWebScraper,
                WebpageScraperToolInputSchema,
            )

            scraper = SimpleWebScraper()
            input_data = WebpageScraperToolInputSchema(url=url)
            result = await scraper.arun(input_data)

            metadata = {
                "title": result.metadata.title,
                "authors": getattr(result.metadata, "author", None),
                "journal": getattr(result.metadata, "publisher", None),
                "publication_date": result.metadata.published_date,
                "doi": result.metadata.doi,  # Might be extracted from meta tags
            }

            if self.debug:
                print(f"Extracted metadata from {url}: {metadata}")

            return metadata

        except Exception as e:
            if self.debug:
                print(f"Failed to extract metadata from {url}: {e}")
            return {}

    async def _resolve_doi_via_crossref_search(self, metadata: dict) -> str:
        """
        Resolve DOI using CrossRef search API with bibliographic metadata.

        Args:
            metadata: Dictionary with bibliographic information

        Returns:
            DOI string if found, None otherwise
        """
        if not metadata:
            return None

        # Build search query
        query_parts = []

        if metadata.get("title"):
            query_parts.append(metadata["title"])
        if metadata.get("authors"):
            if isinstance(metadata["authors"], str):
                query_parts.append(metadata["authors"])
            elif isinstance(metadata["authors"], list):
                query_parts.extend(metadata["authors"][:2])  # First 2 authors

        if not query_parts:
            return None

        search_query = " ".join(query_parts)

        # Try different CrossRef search strategies
        search_strategies = [
            # Strategy 1: Bibliographic search
            {
                "url": f"{self.crossref_base_url}",
                "params": {
                    "query.bibliographic": search_query,
                    "rows": 5,
                    "select": "DOI,title,container-title,author,published-print,published-online",
                },
            },
            # Strategy 2: Field-specific search
            {
                "url": f"{self.crossref_base_url}",
                "params": {
                    "query.title": metadata.get("title", ""),
                    "query.author": metadata.get("authors", ""),
                    "rows": 5,
                    "select": "DOI,title,container-title,author",
                },
            },
            # Strategy 3: Container title + author search
            {
                "url": f"{self.crossref_base_url}",
                "params": {
                    "query.container-title": metadata.get("journal", ""),
                    "query.author": metadata.get("authors", ""),
                    "rows": 5,
                    "select": "DOI,title,container-title,author",
                },
            },
        ]

        import urllib.parse

        import aiohttp

        async with aiohttp.ClientSession() as session:
            for strategy in search_strategies:
                try:
                    # Build URL with parameters
                    params = {k: v for k, v in strategy["params"].items() if v}
                    if not params:
                        continue

                    param_string = urllib.parse.urlencode(params)
                    full_url = f"{strategy['url']}?{param_string}"

                    if self.debug:
                        print(f"Searching CrossRef: {full_url}")

                    headers = {
                        "User-Agent": "JournalValidationPipeline/1.0 (mailto:research@example.org)",
                        "Accept": "application/json",
                    }

                    async with session.get(full_url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get("message", {}).get("items", [])

                            if items:
                                # Find best match based on title similarity
                                best_match = self._find_best_title_match(
                                    metadata.get("title", ""), items
                                )
                                if best_match:
                                    doi = best_match.get("DOI")
                                    if self.debug:
                                        print(f"Found DOI via CrossRef search: {doi}")
                                    return doi

                except Exception as e:
                    if self.debug:
                        print(f"CrossRef search strategy failed: {e}")
                    continue

        return None

    def _find_best_title_match(self, target_title: str, crossref_items: list) -> dict:
        """
        Find the best matching item based on title similarity.

        Args:
            target_title: Title to match against
            crossref_items: List of items from CrossRef API

        Returns:
            Best matching item or None
        """
        if not target_title or not crossref_items:
            return None

        target_title_lower = target_title.lower().strip()
        best_match = None
        best_score = 0

        for item in crossref_items:
            item_titles = item.get("title", [])
            if not item_titles:
                continue

            item_title = item_titles[0].lower().strip()

            # Simple similarity scoring
            if target_title_lower == item_title:
                return item  # Exact match

            # Partial match scoring
            overlap = len(set(target_title_lower.split()) & set(item_title.split()))
            total_words = len(set(target_title_lower.split()) | set(item_title.split()))

            if total_words > 0:
                score = overlap / total_words
                if score > best_score and score > 0.5:  # Minimum 50% overlap
                    best_score = score
                    best_match = item

        return best_match if best_score > 0.5 else None

    async def _enhanced_doi_extraction(self, search_result) -> str:
        """
        Enhanced DOI extraction with fallback to CrossRef search.

        Args:
            search_result: Search result item

        Returns:
            DOI string if found, None otherwise
        """
        # Try original regex-based extraction first
        if hasattr(search_result, "doi") and search_result.doi:
            return search_result.doi

        if hasattr(search_result, "url"):
            doi = self.source_validator._extract_doi_from_url(str(search_result.url))
            if doi:
                return doi

        if hasattr(search_result, "pdf_url") and search_result.pdf_url:
            doi = self.source_validator._extract_doi_from_url(
                str(search_result.pdf_url)
            )
            if doi:
                return doi

        # If regex extraction fails, try metadata extraction + CrossRef search
        if hasattr(search_result, "url"):
            metadata = await self._extract_metadata_from_url(str(search_result.url))

            # Check if DOI was found in metadata
            if metadata.get("doi"):
                return metadata["doi"]

            # Try CrossRef search with extracted metadata
            return await self._resolve_doi_via_crossref_search(metadata)

        return None

    async def enhanced_validate_search_results(
        self, search_results: List[SearchResultItem]
    ) -> dict:
        """
        Enhanced validation with improved DOI resolution.

        Args:
            search_results: List of search results to validate

        Returns:
            Validation results with summary
        """
        enhanced_results = []

        for result in search_results:
            # Try enhanced DOI extraction
            doi = await self._enhanced_doi_extraction(result)

            if doi:
                # Update the result with the found DOI
                if hasattr(result, "doi"):
                    result.doi = doi
                else:
                    # Create a new result with DOI
                    setattr(result, "doi", doi)

            enhanced_results.append(result)

        # Use standard validation with enhanced results
        validation_input = {"search_results": enhanced_results}
        validation_results = await self.source_validator.arun(validation_input)

        return {
            "results": validation_results.validated_results,
            "summary": validation_results.summary,
        }


async def main():
    """Main function to run the journal validation pipeline example."""
    print("Journal Validation Pipeline Example")
    print("===================================\n")

    # Create pipeline instance
    pipeline = JournalValidationPipeline(debug=True)

    # Example 1: Earth Sciences queries
    print("Running Earth Sciences validation...")
    await pipeline.run_full_pipeline(EARTH_SCIENCE_QUERIES[:2], max_results_per_query=3)

    print("\n\n" + "=" * 100 + "\n")

    # Example 2: Astronomy queries
    print("Running Astronomy validation...")
    await pipeline.run_full_pipeline(ASTRONOMY_QUERIES[:2], max_results_per_query=3)


def create_sample_search_results() -> List[SearchResultItem]:
    """
    Create sample search results for testing when search engines are not available.
    """
    return [
        SearchResultItem(
            url="https://doi.org/10.1016/j.actaastro.2023.05.014",
            title="Advanced spacecraft propulsion systems for deep space exploration",
            content="This paper discusses novel propulsion technologies...",
            query="spacecraft propulsion systems",
            category="science",
            doi="10.1016/j.actaastro.2023.05.014",
            published_date="2023-05-15",
            engine="test",
        ),
        SearchResultItem(
            url="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL104567",
            title="Climate sensitivity to atmospheric CO2 concentrations",
            content="Analysis of climate models shows...",
            query="climate change modeling",
            category="science",
            doi="10.1029/2023GL104567",
            published_date="2023-06-20",
            engine="test",
        ),
        SearchResultItem(
            url="https://example.com/invalid-journal-paper",
            title="Some research paper in unknown journal",
            content="This paper is from a journal not in our whitelist...",
            query="research paper",
            category="science",
            published_date="2023-07-01",
            engine="test",
        ),
    ]


async def test_with_sample_data():
    """Test the pipeline with sample data when search engines are not available."""
    print("Testing Journal Validation Pipeline with Sample Data")
    print("===================================================\n")

    # Create pipeline instance
    pipeline = JournalValidationPipeline(debug=True)

    # Create sample search results
    sample_results = create_sample_search_results()

    print(f"Created {len(sample_results)} sample search results")

    # Validate the sample results
    print("\nValidating sample results against journal whitelist...")
    validation_data = await pipeline.validate_search_results(sample_results)

    # Generate report
    print("\nGenerating validation report...")
    pipeline.print_validation_report(validation_data)


async def demo_enhanced_doi_resolution():
    """
    Demo of enhanced DOI resolution capabilities.
    """
    print("Enhanced DOI Resolution Demo")
    print("============================\n")

    # Create enhanced pipeline
    pipeline = EnhancedJournalValidationPipeline(debug=True)

    # Test cases with challenging URLs
    test_cases = [
        SearchResultItem(
            url="https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023GL104567",
            title="Climate sensitivity to atmospheric CO2 concentrations in AGU journal",
            content="Analysis of climate models shows enhanced sensitivity...",
            query="climate change modeling",
            category="science",
            published_date="2023-06-20",
            engine="test",
        ),
        SearchResultItem(
            url="https://www.nature.com/articles/s41586-023-12345-6",
            title="Advanced machine learning techniques for climate prediction",
            content="Novel approaches to predicting climate patterns...",
            query="machine learning climate",
            category="science",
            published_date="2023-08-15",
            engine="test",
        ),
        SearchResultItem(
            url="https://example-university.edu/research/paper-without-doi.html",
            title="Some research paper without clear DOI",
            content="Research content from institutional repository...",
            query="research paper",
            category="science",
            published_date="2023-07-01",
            engine="test",
        ),
    ]

    print(
        f"Testing enhanced DOI resolution with {len(test_cases)} challenging cases...\n"
    )

    # Test enhanced validation
    validation_data = await pipeline.enhanced_validate_search_results(test_cases)

    # Generate report
    pipeline.print_validation_report(validation_data)

    # Show DOI resolution statistics
    total_cases = len(test_cases)
    resolved_cases = sum(
        1 for result in validation_data["results"] if result.source_info
    )

    print("\nDOI RESOLUTION STATISTICS:")
    print(f"Total test cases: {total_cases}")
    print(f"Successfully resolved DOIs: {resolved_cases}")
    print(f"Resolution success rate: {(resolved_cases / total_cases) * 100:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--enhanced":
        asyncio.run(demo_enhanced_doi_resolution())
    elif len(sys.argv) > 1 and sys.argv[1] == "--sample":
        asyncio.run(test_with_sample_data())
    else:
        asyncio.run(main())
