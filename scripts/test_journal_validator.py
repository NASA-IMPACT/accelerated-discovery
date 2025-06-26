#!/usr/bin/env python3
"""
Simple test script for the journal validator tool.

This script tests the core functionality of the journal validation pipeline
with predefined test cases.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from akd.structures import SearchResultItem
from akd.tools.source_validator import create_journal_validator


def create_test_search_results():
    """Create test search results with known DOIs for validation."""
    return [
        # Should be whitelisted - AGU journal
        SearchResultItem(
            url="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL104567",
            title="Climate sensitivity to atmospheric CO2 concentrations",
            content="Analysis of climate models shows significant sensitivity to CO2...",
            query="climate change modeling",
            category="science",
            doi="10.1029/2023GL104567",
            published_date="2023-06-20",
            engine="test",
        ),
        # Should be whitelisted - Nature journal
        SearchResultItem(
            url="https://doi.org/10.1038/s41561-023-01234-5",
            title="Evidence for past water activity on Mars",
            content="Geological evidence suggests extensive water activity...",
            query="mars water geology",
            category="science",
            published_date="2023-07-15",
            engine="test",
        ),
        # URL with DOI that should extract correctly
        SearchResultItem(
            url="https://www.sciencedirect.com/science/article/pii/S0012821X23001234?via%3Dihub",
            title="Mantle convection dynamics and plate tectonics",
            content="New insights into mantle convection patterns...",
            query="mantle convection",
            category="science",
            published_date="2023-05-10",
            engine="test",
        ),
        # No DOI - should fail extraction
        SearchResultItem(
            url="https://example.com/some-random-article",
            title="Random article without DOI",
            content="This article has no DOI identifier...",
            query="random research",
            category="science",
            published_date="2023-08-01",
            engine="test",
        ),
    ]


async def test_journal_validator():
    """Test the journal validator with sample data."""
    print("Testing Journal Validator Tool")
    print("=" * 50)

    # Create validator instance
    validator = create_journal_validator(debug=True)

    # Create test search results
    test_results = create_test_search_results()

    print(f"\nCreated {len(test_results)} test search results:")
    for i, result in enumerate(test_results, 1):
        print(f"  {i}. {result.title}")
        print(f"     URL: {result.url}")
        print(f"     DOI: {result.doi or 'None'}")

    # Run validation
    print("\nRunning validation...")
    validation_input = {"search_results": test_results}

    try:
        validation_results = await validator.arun(validation_input)

        print("\nValidation completed successfully!")
        print(f"Summary: {validation_results.summary}")

        print("\nDetailed Results:")
        print("-" * 50)

        for i, result in enumerate(validation_results.validated_results, 1):
            print(f"\n{i}. Result:")
            print(f"   Whitelisted: {'Yes' if result.is_whitelisted else 'No'}")
            print(f"   Category: {result.whitelist_category or 'None'}")
            print(f"   Confidence: {result.confidence_score:.2f}")

            if result.validation_errors:
                print(f"   Errors: {', '.join(result.validation_errors)}")

            if result.journal_info:
                ji = result.journal_info
                print(f"   Journal: {ji.title}")
                print(f"   Publisher: {ji.publisher or 'Unknown'}")
                print(f"   DOI: {ji.doi}")
                print(f"   Open Access: {ji.is_open_access}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Total processed: {validation_results.summary['total_processed']}")
        print(f"  Whitelisted: {validation_results.summary['whitelisted_count']}")
        print(
            f"  Success rate: {validation_results.summary['whitelisted_percentage']:.1f}%"
        )
        print(f"  Errors: {validation_results.summary['error_count']}")

        if validation_results.summary.get("category_breakdown"):
            print(
                f"  Categories found: {list(validation_results.summary['category_breakdown'].keys())}"
            )

        return True

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_doi_extraction():
    """Test DOI extraction functionality."""
    print("\nTesting DOI Extraction")
    print("=" * 50)

    # Create validator to access DOI extraction
    validator = create_journal_validator()

    # Test URLs with different DOI formats
    test_urls = [
        "https://doi.org/10.1016/j.actaastro.2023.05.014",
        "https://dx.doi.org/10.1038/s41561-023-01234-5",
        "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL104567",
        "https://www.sciencedirect.com/science/article/pii/S0012821X23001234",
        "https://arxiv.org/abs/2023.12345",  # No DOI expected
        "https://example.com/paper?doi=10.1234/example.doi.2023",
        "https://some-journal.com/articles/doi:10.5678/journal.2023.001",
    ]

    print("Testing DOI extraction from various URL formats:")

    for url in test_urls:
        extracted_doi = validator._extract_doi_from_url(url)
        print(f"  URL: {url}")
        print(f"  DOI: {extracted_doi or 'None found'}")
        print()


async def main():
    """Main test function."""
    print("Journal Validator Tool Test Suite")
    print("=" * 60)

    # Test DOI extraction
    await test_doi_extraction()

    # Test full validation pipeline
    success = await test_journal_validator()

    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
