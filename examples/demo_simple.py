import argparse
import asyncio
import os

from dotenv import load_dotenv

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search import DataSearchAgent, DataSearchAgentConfig
from akd.agents.data_search.components import (
    KnownParametersComponent,
    RepositoryRouterComponent,
    ScientificDecomposition,
    ScientificDecompositionComponent,
    SearchableParametersComponent,
    Topic,
    TopicSplittingComponent,
)
from akd.agents.data_search.handlers import CMRHandlerConfig
from akd.configs.data_search_config import get_config

# loads .env file into os.environ
load_dotenv()

# Load environment variables after imports
load_dotenv()

print("✅ Imports successful")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🔑 OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

# Model configuration for each pipeline component
MODEL_CONFIG = {
    "topic_splitting": "gpt-5-mini",
    "scientific_decomposition": "gpt-5-mini",
    "repository_routing": "gpt-5-mini",
    "cmr_query": "gpt-5-mini",
}

print("🎛️ Model Configuration:")
for component, model in MODEL_CONFIG.items():
    print(f"   • {component.replace('_', ' ').title()}: {model}")

# Load base configuration
config = get_config()

# Configure CMR handler
cmr_handler_config = CMRHandlerConfig(
    mcp_endpoint=config.mcp.endpoint,
    collection_search_page_size=20,
    granule_search_page_size=10,
    collections_per_query=5,
    max_collections_per_approach=5,
    final_collection_count=25,
    min_collection_relevance_score=0.3,
    collection_search_timeout=30.0,
    granule_search_timeout=45.0,
    enable_parallel_search=True,
    known_parameters_model=MODEL_CONFIG["cmr_query"],
    searchable_parameters_model=MODEL_CONFIG["cmr_query"],
    approach_filtering_model=MODEL_CONFIG["cmr_query"],
    final_ranking_model=MODEL_CONFIG["cmr_query"],
)

# Configure agent with model-specific settings
agent_config = DataSearchAgentConfig(
    debug=True,
    enable_parallel_search=True,
    # Universal component models
    topic_splitting_model=MODEL_CONFIG["topic_splitting"],
    scientific_decomposition_model=MODEL_CONFIG["scientific_decomposition"],
    repository_routing_model=MODEL_CONFIG["repository_routing"],
    # Handler-specific configurations
    cmr=cmr_handler_config,
)

# Initialize the agent with our configured models
agent = DataSearchAgent(config=agent_config, debug=True)

print("\n🤖 Agent initialized with component-specific models:")
print(f"   • Topic Splitting: {agent_config.topic_splitting_model}")
print(f"   • Scientific Decomposition: {agent_config.scientific_decomposition_model}")
print(f"   • Repository Routing: {agent_config.repository_routing_model}")
print(f"   • CMR Known Parameters: {agent_config.cmr.known_parameters_model}")
print(f"   • CMR Searchable Parameters: {agent_config.cmr.searchable_parameters_model}")
print(f"   • CMR Approach Filtering: {agent_config.cmr.approach_filtering_model}")
print(f"   • CMR Final Ranking: {agent_config.cmr.final_ranking_model}")


# =============================================================================
# INDIVIDUAL COMPONENT TESTING FUNCTIONS
# =============================================================================


async def test_topic_splitting_only(query: str = None):
    """Test only the topic splitting component."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING TOPIC SPLITTING ONLY")
    print(f"Query: '{demo_query}'")

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["topic_splitting"])
    component = TopicSplittingComponent(config=config, debug=True)

    # Test topic splitting
    print("\n1️⃣ Topic Splitting:")
    topics_output = await component.process(demo_query)

    print(f"   ✅ Identified {len(topics_output.topics)} topics:")
    for i, topic in enumerate(topics_output.topics, 1):
        print(f"   {i}. **{topic.title}**")
        print(f"      Context: {topic.functional_context}")

    return topics_output.topics


async def test_repository_routing_only(
    query: str = None,
    topic: Topic = None,
    decomposition: ScientificDecomposition = None,
):
    """Test only the repository routing component (requires decomposition)."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING REPOSITORY ROUTING ONLY")
    print(f"Query: '{demo_query}'")

    # Get prerequisites if not provided
    if not topic:
        print("   🔄 Getting topics from topic splitting first...")
        topics = await test_topic_splitting_only(demo_query)
        topic = topics[0]

    if not decomposition:
        print("   🔄 Getting decompositions first...")
        decompositions = await test_scientific_decomposition_only(demo_query, topic)
        decomposition = decompositions[0]

    print(f"   📌 Topic: {topic.title}")
    print(f"   🔬 Decomposition: {decomposition.title}")

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["repository_routing"])
    component = RepositoryRouterComponent(config=config, debug=True)

    # Test repository routing
    print("\n2️⃣ Repository Routing:")
    routing_output = await component.process(demo_query, topic, decomposition)

    print("   ✅ Routing Result:")
    print(f"      Repository: {routing_output.route.repository}")
    print(f"      Is External: {routing_output.route.is_external}")
    print(f"      Rationale: {routing_output.route.rationale}")

    return routing_output


async def test_scientific_decomposition_only(query: str = None, topic: Topic = None):
    """Test only the scientific decomposition component."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING SCIENTIFIC DECOMPOSITION ONLY")
    print(f"Query: '{demo_query}'")

    # Use provided topic or get first topic from topic splitting
    if not topic:
        print("   🔄 Getting topics from topic splitting first...")
        topics = await test_topic_splitting_only(demo_query)
        topic = topics[0]  # Use first topic

    print(f"   📌 Processing topic: {topic.title}")

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["scientific_decomposition"])
    component = ScientificDecompositionComponent(config=config, debug=True)

    # Test scientific decomposition
    print("\n3️⃣ Scientific Decomposition:")
    decomp_output = await component.process(demo_query, topic)

    print(f"   ✅ Generated {len(decomp_output.decompositions)} decompositions:")
    for i, decomp in enumerate(decomp_output.decompositions, 1):
        print(f"\n   {i}. **{decomp.title}**")
        print(f"      Scientific Justification: {decomp.scientific_justification}")

    return decomp_output.decompositions


async def test_known_parameters_only(
    query: str = None,
    topic: Topic = None,
    decomposition: ScientificDecomposition = None,
):
    """Test only the known parameters component."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING KNOWN PARAMETERS ONLY")
    print(f"Query: '{demo_query}'")

    # Get prerequisites if not provided
    if not topic:
        print("   🔄 Getting topics from topic splitting first...")
        topics = await test_topic_splitting_only(demo_query)
        topic = topics[0]

    if not decomposition:
        print("   🔄 Getting decompositions first...")
        decompositions = await test_scientific_decomposition_only(demo_query, topic)
        decomposition = decompositions[0]

    print(f"   📌 Topic: {topic.title}")
    print(f"   🔬 Decomposition: {decomposition.title}")

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["cmr_query"])
    component = KnownParametersComponent(config=config, debug=True)

    # Test known parameters
    print("\n4️⃣ Known Parameters:")
    known_params = await component.process(demo_query, topic, decomposition)

    print(f"   ✅ Generated {len(known_params.query_approaches)} query approaches:")
    print(f"   📋 Reasoning: {known_params.reasoning}")

    for i, approach in enumerate(known_params.query_approaches, 1):
        print(f"\n   Approach {i}:")
        approach_dict = approach.model_dump()
        for param, value in approach_dict.items():
            if value is not None:
                print(f"     {param}: {value}")

    return known_params.query_approaches


async def test_searchable_parameters_only(
    query: str = None,
    topic: Topic = None,
    decomposition: ScientificDecomposition = None,
    query_approaches: list = None,
):
    """Test only the searchable parameters component."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING SEARCHABLE PARAMETERS ONLY")
    print(f"Query: '{demo_query}'")

    # Get prerequisites if not provided
    if not topic:
        print("   🔄 Getting topics from topic splitting first...")
        topics = await test_topic_splitting_only(demo_query)
        topic = topics[0]

    if not decomposition:
        print("   🔄 Getting decompositions first...")
        decompositions = await test_scientific_decomposition_only(demo_query, topic)
        decomposition = decompositions[0]

    if not query_approaches:
        print("   🔄 Getting known parameters first...")
        query_approaches = await test_known_parameters_only(
            demo_query,
            topic,
            decomposition,
        )

    print(f"   📌 Topic: {topic.title}")
    print(f"   🔬 Decomposition: {decomposition.title}")
    print(f"   📊 Query Approaches: {len(query_approaches)}")

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["cmr_query"])
    component = SearchableParametersComponent(config=config, debug=True)

    # Test searchable parameters
    print("\n5️⃣ Searchable Parameters:")
    searchable_output = await component.process(
        demo_query,
        topic,
        decomposition,
        query_approaches,
    )

    print(
        f"   ✅ Generated {len(searchable_output.searchable_queries)} searchable queries:",
    )
    print(f"   📋 Strategy: {searchable_output.keyword_strategy}")

    for i, query in enumerate(searchable_output.searchable_queries, 1):
        print(f"\n   Query {i}:")
        print(f"     Keywords: {', '.join(query.primary_keywords)}")
        print(f"     Combined: {query.combined_keyword_string}")
        if query.instrument:
            print(f"     Instrument: {query.instrument}")
        if query.temporal:
            print(f"     Temporal: {query.temporal}")

    return searchable_output.searchable_queries


# =============================================================================
# ORIGINAL TESTING FUNCTIONS (PRESERVED)
# =============================================================================


async def test_individual_components():
    """
    Test universal components (topic splitting, routing, decomposition).

    Note: CMR-specific components (known parameters, searchable parameters, collection search)
    are now encapsulated within the CMR handler and tested via standalone component tests.
    """
    demo_query = (
        "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 UNIVERSAL COMPONENT TESTING")
    print(f"Query: '{demo_query}'")

    # 1. Topic Splitting
    print("\n1️⃣ Topic Splitting:")
    topics_output = await agent.topic_splitting_component.process(demo_query)
    print(f"   Found {len(topics_output.topics)} topics. Using first topic:")
    first_topic = topics_output.topics[0]
    print(f"   • {first_topic.title}")
    print(f"     Context: {first_topic.functional_context}")

    # 2. Repository Routing (first topic only)
    print(f"\n2️⃣ Repository Routing for '{first_topic.title}':")
    # Note: Routing is now per-decomposition, not per-topic
    print("   ℹ️  Repository routing now happens per decomposition, not per topic.")
    print(
        "   ℹ️  Use test_topic_splitting_only() and test_repository_routing_only() for isolated testing.",
    )

    # 3. Scientific Decomposition (first topic only)
    print(f"\n3️⃣ Scientific Decomposition for '{first_topic.title}':")
    decomp_output = await agent.scientific_decomposition_component.process(
        demo_query,
        first_topic,
    )
    print(f"   Found {len(decomp_output.decompositions)} decompositions. Using first:")
    first_decomp = decomp_output.decompositions[0]
    print(f"   • {first_decomp.title}")
    print(f"     Justification: {first_decomp.scientific_justification}")

    print(
        "\n   ℹ️  CMR-specific components (known parameters, searchable parameters, collection search)",
    )
    print("   ℹ️  are now encapsulated within the CMR handler.")
    print(
        "   ℹ️  Use standalone component tests (test_known_parameters_only, etc.) for detailed testing.",
    )


async def test_new_workflow(query: str = None):
    """Test the complete multi-repository data search workflow."""
    from akd.agents.data_search._base import DataSearchAgentInputSchema

    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🆕 TESTING MULTI-REPOSITORY DATA SEARCH WORKFLOW")
    print(f"🔍 Query: '{demo_query}'")
    print("\n   Running full agent pipeline...")

    # Run the full agent workflow
    input_params = DataSearchAgentInputSchema(query=demo_query)
    result = await agent.arun(input_params)

    # Display results
    print("\n📊 WORKFLOW RESULTS:")
    print(f"   Topics Processed: {len(result.topics)}")
    print(f"   Total Results Found: {result.total_results}")

    for i, topic_result in enumerate(result.topics, 1):
        print(f"\n   Topic {i}: {topic_result.topic.get('title', 'N/A')}")
        print(f"   Data Source: {topic_result.data_source}")
        print(f"   Decompositions: {len(topic_result.decomposition_results)}")

        for j, decomp_result in enumerate(topic_result.decomposition_results, 1):
            print(
                f"\n      Decomposition {j}: {decomp_result.decomposition.get('title', 'N/A')}",
            )
            print(f"      Repository: {decomp_result.repository}")
            print(f"      Results Found: {decomp_result.total_results_found}")
            print(f"      Data Results Returned: {len(decomp_result.data_results)}")

            if decomp_result.note:
                print(f"      Note: {decomp_result.note}")

            if decomp_result.data_results:
                print("      Sample results (first 3):")
                for k, data in enumerate(decomp_result.data_results[:3], 1):
                    title = data.get("title", data.get("short_name", "N/A"))
                    print(f"         {k}. {title}")

    print("\n   Search Metadata:")
    for key, value in result.search_metadata.items():
        print(f"      {key}: {value}")


# =============================================================================
# MAIN FUNCTION WITH COMMAND-LINE SUPPORT
# =============================================================================


async def main():
    """Main demo function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Data Search Agent Demo - Test individual components or full workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                          # Run full workflow (default)
  python demo.py --component topic        # Test topic splitting only
  python demo.py --component routing      # Test repository routing only
  python demo.py --component decomp       # Test scientific decomposition only
  python demo.py --component known        # Test known parameters only
  python demo.py --component searchable   # Test searchable parameters only
  python demo.py --component ranking      # Test collection ranking only
  python demo.py --component all          # Test all individual components
  python demo.py --query "MODIS temperature data"  # Custom query
        """,
    )

    parser.add_argument(
        "--component",
        "-c",
        choices=["topic", "routing", "decomp", "known", "searchable", "ranking", "all"],
        help="Test only a specific component",
    )

    parser.add_argument(
        "--query",
        "-q",
        default="Help me gather data to study the flood risk of the lower Mississippi basin",
        help="Custom research query to test with",
    )

    args = parser.parse_args()

    print("🧪 TESTING DATA SEARCH AGENT - NEW TOPIC-BASED WORKFLOW")
    print(f"Query: '{args.query}'")

    if args.component == "topic":
        await test_topic_splitting_only(args.query)
    elif args.component == "routing":
        await test_repository_routing_only(args.query)
    elif args.component == "decomp":
        await test_scientific_decomposition_only(args.query)
    elif args.component == "known":
        await test_known_parameters_only(args.query)
    elif args.component == "searchable":
        await test_searchable_parameters_only(args.query)
    elif args.component == "all":
        print("\n" + "=" * 80)
        print("TESTING ALL INDIVIDUAL COMPONENTS")
        print("=" * 80)

        # Test each component sequentially
        await test_topic_splitting_only(args.query)
        await test_repository_routing_only(args.query)
        await test_scientific_decomposition_only(args.query)
        await test_known_parameters_only(args.query)
        await test_searchable_parameters_only(args.query)

        print("\n" + "=" * 80)
        print("ALL INDIVIDUAL COMPONENT TESTS COMPLETED!")
        print("=" * 80)
    else:
        # Default: run full workflow once
        print("\n🔄 Running full workflow (use --component to test individual parts)")

        # Test complete new workflow
        await test_new_workflow(args.query)

    print("\n✅ Testing completed successfully!")
    if not args.component:
        print("🎯 The new topic-based workflow is operational!")
    else:
        print(f"🎯 Component '{args.component}' testing completed!")
        print("💡 Tip: Use 'python demo.py --component all' to test all components")
        print("💡 Tip: Use 'python demo.py' to run the full workflow")


# Run the demo
if __name__ == "__main__":
    asyncio.run(main())
