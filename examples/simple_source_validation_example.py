#!/usr/bin/env python3
"""
Simple source validation example.

This example demonstrates the core source validation functionality
without complex search dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from akd.structures import SearchResultItem
from akd.tools.source_validator import create_source_validator


def create_test_search_results():
    """Create test search results with real DOIs for validation."""
    return [
        # Real AGU source paper - should be whitelisted
        SearchResultItem(
            url="https://doi.org/10.1029/2019GL084947",
            title="Arctic Sea Ice Decline and Recovery in Observations and the NASA GISS Climate Model",
            content="Analysis of Arctic sea ice trends using observations and climate model...",
            query="arctic sea ice climate",
            category="science",
            doi="10.1029/2019GL084947",
            published_date="2019-09-15",
            engine="test",
        ),
        # Real Nature Geoscience paper - should be whitelisted
        SearchResultItem(
            url="https://doi.org/10.1038/s41561-020-0566-6",
            title="Global carbon dioxide emissions from inland waters",
            content="Assessment of CO2 emissions from rivers, lakes, and reservoirs...",
            query="carbon emissions water",
            category="science",
            published_date="2020-04-20",
            engine="test",
        ),
        # Real Astronomy & Astrophysics paper - should be whitelisted
        SearchResultItem(
            url="https://doi.org/10.1051/0004-6361/202038711",
            title="Gaia early data release 3: The celestial reference frame",
            content="Precise astrometric measurements from Gaia mission...",
            query="gaia astrometry",
            category="science",
            published_date="2021-01-10",
            engine="test",
        ),
        # Paper without DOI - should fail
        SearchResultItem(
            url="https://example.com/no-doi-paper",
            title="Paper without DOI identifier",
            content="This paper has no DOI and should fail validation...",
            query="random research",
            category="science",
            published_date="2023-08-01",
            engine="test",
        ),
    ]


async def main():
    """Main function demonstrating source validation."""
    print("Simple Source Validation Example")
    print("=" * 50)

    # Create validator instance
    print("Creating source validator...")
    validator = create_source_validator(debug=True)

    # Create test search results
    print("\nCreating test search results...")
    test_results = create_test_search_results()

    print(f"Created {len(test_results)} test search results:")
    for i, result in enumerate(test_results, 1):
        print(f"  {i}. {result.title}")
        print(f"     DOI: {result.doi or 'None'}")
        print(f"     URL: {result.url}")

    # Run validation
    print("\nRunning validation against source whitelist...")
    validation_input = {"search_results": test_results}

    try:
        validation_results = await validator.arun(validation_input)

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)

        # Print summary
        summary = validation_results.summary
        print("\nSUMMARY:")
        print(f"  Total papers processed: {summary['total_processed']}")
        print(f"  Whitelisted papers: {summary['whitelisted_count']}")
        print(f"  Success rate: {summary['whitelisted_percentage']:.1f}%")
        print(f"  Papers with errors: {summary['error_count']}")
        print(f"  Average confidence: {summary['avg_confidence']:.2f}")

        if summary.get("category_breakdown"):
            print("\n  Categories found:")
            for category, count in summary["category_breakdown"].items():
                print(f"    {category}: {count} papers")

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 80)

        for i, result in enumerate(validation_results.validated_results, 1):
            print(f"\n{i}. {test_results[i - 1].title}")

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
                si = result.source_info
                print(f"   Source: {si.title}")
                print(f"   Publisher: {si.publisher or 'Unknown'}")
                print(f"   DOI: {si.doi}")
                print(f"   Open Access: {si.is_open_access or 'Unknown'}")
                if si.issn:
                    print(f"   ISSN: {', '.join(si.issn)}")

        print("\n" + "=" * 80)

        # Determine success
        if summary["whitelisted_count"] > 0:
            print("✅ Validation completed successfully - found whitelisted sources!")
        elif summary["error_count"] < summary["total_processed"]:
            print(
                "⚠️ Validation completed - no whitelisted sources found but CrossRef lookups worked"
            )
        else:
            print(
                "❌ All validations failed - check network connectivity or DOI formats"
            )

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
