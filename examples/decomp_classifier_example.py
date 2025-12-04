"""
Example script demonstrating the decomposition classifier for aspect search.

This example shows how to:
1. Enable query classification in the aspect search agent
2. View classifications for decomposed queries
3. Filter queries by classification (e.g., skip TANGENTIAL queries)
4. Access classification results for analysis
"""

import asyncio
import os

from akd.agents.search.aspect_search import AspectSearchAgent, AspectSearchConfig
from akd.structures import DecompositionClassification
from akd.tools.decomp_classifier import DecompClassifierConfig


async def example_basic_classification():
    """
    Example 1: Basic classification without filtering.

    This runs the aspect search with classification enabled, allowing you to
    see how each decomposed query is categorized, but all queries are still executed.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Classification (No Filtering)")
    print("=" * 80 + "\n")

    config = AspectSearchConfig(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_query_classification=True,
        classifier_config=DecompClassifierConfig(
            model_name="gpt-5-mini",
            temperature=0.0,
        ),
        max_turns=2,  # Limit turns for faster example
        num_editors=2,  # Fewer editors for faster example
    )

    agent = AspectSearchAgent(config=config, debug=True)

    # Run aspect search on a sample topic
    result = await agent.arun(agent.input_schema(topic="What is fire risk in forests?"))

    print("\n" + "-" * 80)
    print("CLASSIFICATION RESULTS")
    print("-" * 80 + "\n")

    # Display classifications from interviews
    for idx, interview in enumerate(result.interview_results, 1):
        print(f"Interview {idx}:")
        if "messages" in interview:
            # Look for messages that contain classified queries
            for message in interview["messages"]:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # This is where queries were generated
                    print(f"  Editor: {interview.get('editor', {}).get('name', 'Unknown')}")

        # Check if classifications are available in the interview state
        # Note: Classifications may be stored differently depending on implementation
        print()

    print(f"Total search results: {len(result.search_results)}")
    print(f"Total references: {len(result.references)}")


async def example_filtered_classification():
    """
    Example 2: Classification with filtering to skip TANGENTIAL queries.

    This configuration will classify queries and only execute those that are
    EXACT, CALCULATOR, or PROXY, skipping any TANGENTIAL queries that are
    only weakly related to the topic.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Classification with Filtering (Skip TANGENTIAL)")
    print("=" * 80 + "\n")

    config = AspectSearchConfig(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_query_classification=True,
        classifier_config=DecompClassifierConfig(
            model_name="gpt-5-mini",
            temperature=0.0,
        ),
        # Only execute queries that are directly relevant
        filter_classifications=[
            DecompositionClassification.EXACT,
            DecompositionClassification.CALCULATOR,
            DecompositionClassification.PROXY,
            # TANGENTIAL queries will be skipped
        ],
        max_turns=2,
        num_editors=2,
    )

    agent = AspectSearchAgent(config=config, debug=True)

    # Run aspect search
    result = await agent.arun(agent.input_schema(topic="Ocean temperature trends and climate change"))

    print("\n" + "-" * 80)
    print("FILTERING RESULTS")
    print("-" * 80 + "\n")

    print("Configuration filters out TANGENTIAL queries.")
    print("Only EXACT, CALCULATOR, and PROXY queries are executed.")
    print(f"\nTotal search results: {len(result.search_results)}")
    print(f"Total references: {len(result.references)}")


async def example_standalone_classifier():
    """
    Example 3: Using the classifier tool standalone.

    This shows how to use the DecompClassifierTool directly without the
    aspect search agent, which can be useful for testing or analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Standalone Classifier Tool")
    print("=" * 80 + "\n")

    from akd.tools.decomp_classifier import DecompClassifierTool

    # Create classifier
    config = DecompClassifierConfig(
        model_name="gpt-5-mini",
        temperature=0.0,
    )
    classifier = DecompClassifierTool(config=config, debug=True)

    # Test queries
    original_topic = "What is fire risk in California forests?"
    test_queries = [
        "California wildfire risk index",
        "soil moisture content in forests",
        "wind speed and direction patterns",
        "chlorophyll content as indicator of forest health",
        "historical rainfall patterns in California",
        "general climate change overview",
    ]

    print(f"Original Topic: {original_topic}\n")
    print("Decomposed Queries:")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")
    print()

    # Classify
    result = await classifier.arun(
        classifier.input_schema(original_topic=original_topic, queries=test_queries)
    )

    print("\n" + "-" * 80)
    print("CLASSIFICATIONS")
    print("-" * 80 + "\n")

    # Display results
    for cq in result.classified_queries:
        print(f"Query: {cq.query}")
        print(f"Classification: {cq.classification.value.upper()}")
        print(f"Reasoning: {cq.reasoning}")
        print()

    # Summary by category
    print("-" * 80)
    print("SUMMARY BY CATEGORY")
    print("-" * 80 + "\n")

    from collections import Counter

    category_counts = Counter(cq.classification for cq in result.classified_queries)

    for category, count in category_counts.items():
        print(f"{category.value.upper()}: {count} queries")


async def example_domain_specific():
    """
    Example 4: Domain-specific classification for Earth science research.

    This demonstrates how the classifier handles domain-specific queries
    related to Earth observation data and CMR (Common Metadata Repository).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Domain-Specific Classification (Earth Science)")
    print("=" * 80 + "\n")

    from akd.tools.decomp_classifier import DecompClassifierTool

    config = DecompClassifierConfig(
        model_name="gpt-5-mini",
        temperature=0.0,
    )
    classifier = DecompClassifierTool(config=config)

    # Earth science topic
    original_topic = "What is the impact of soil moisture on flood risk?"
    earth_science_queries = [
        "SMAP soil moisture L3 product",
        "soil moisture anomaly calculation",
        "precipitation data from GPM",
        "topography and slope from DEM",
        "NDVI as proxy for vegetation water stress",
        "historical flood events database",
        "general hydrology textbook information",
    ]

    print(f"Original Topic: {original_topic}\n")
    print("Earth Science Queries:")
    for i, q in enumerate(earth_science_queries, 1):
        print(f"  {i}. {q}")
    print()

    result = await classifier.arun(
        classifier.input_schema(original_topic=original_topic, queries=earth_science_queries)
    )

    print("\n" + "-" * 80)
    print("EARTH SCIENCE CLASSIFICATIONS")
    print("-" * 80 + "\n")

    # Organize by category
    by_category = {
        DecompositionClassification.EXACT: [],
        DecompositionClassification.CALCULATOR: [],
        DecompositionClassification.PROXY: [],
        DecompositionClassification.TANGENTIAL: [],
    }

    for cq in result.classified_queries:
        by_category[cq.classification].append(cq)

    for category, queries in by_category.items():
        print(f"\n{category.value.upper()} ({len(queries)} queries):")
        for cq in queries:
            print(f"  • {cq.query}")
            print(f"    Reasoning: {cq.reasoning}")


async def main():
    """Run all examples."""
    print("=" * 80)
    print("DECOMPOSITION CLASSIFICATION EXAMPLES")
    print("=" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running this example.")
        return

    # Run examples
    try:
        # Example 3: Standalone classifier (fastest, no search)
        await example_standalone_classifier()

        # Example 4: Domain-specific (fast, no search)
        await example_domain_specific()

        # Example 1: Basic classification (slower, includes search)
        # await example_basic_classification()

        # Example 2: Filtered classification (slower, includes search)
        # await example_filtered_classification()

        print("\n" + "=" * 80)
        print("EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nNote: Examples 1 and 2 are commented out by default as they")
        print("perform full aspect search which takes longer. Uncomment them")
        print("in main() to run the full examples.")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
