import argparse
import asyncio
import os

from dotenv import load_dotenv

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search import CMRDataSearchAgent, CMRDataSearchAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema
from akd.agents.data_search.components import (
    CollectionRankingComponent,
    KnownParametersComponent,
    RepositoryRouterComponent,
    ScientificDecomposition,
    ScientificDecompositionComponent,
    SearchableParametersComponent,
    Topic,
    TopicSplittingComponent,
)
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
    "collection_ranking": "gpt-5-mini",
    "cmr_query": "gpt-5-mini",
}

print("🎛️ Model Configuration:")
for component, model in MODEL_CONFIG.items():
    print(f"   • {component.replace('_', ' ').title()}: {model}")

# Load base configuration
config = get_config()

# Configure agent with model-specific settings
agent_config = CMRDataSearchAgentConfig(
    debug=True,
    mcp_endpoint=config.mcp.endpoint,
    max_collections_to_search=5,
    collection_search_page_size=20,
    granule_search_page_size=10,
    enable_parallel_search=True,
    collection_search_timeout=30.0,
    granule_search_timeout=45.0,
    min_collection_relevance_score=0.3,
    # Model configurations for components
    topic_splitting_model=MODEL_CONFIG["topic_splitting"],
    scientific_decomposition_model=MODEL_CONFIG["scientific_decomposition"],
    repository_routing_model=MODEL_CONFIG["repository_routing"],
    collection_ranking_model=MODEL_CONFIG["collection_ranking"],
    cmr_query_model=MODEL_CONFIG["cmr_query"],
    # Legacy compatibility
    angle_generation_model=MODEL_CONFIG["topic_splitting"],
)

# Initialize the agent with our configured models
agent = CMRDataSearchAgent(config=agent_config, debug=True)

print("\n🤖 Agent initialized with component-specific models:")
print(f"   • Topic Splitting: {agent_config.topic_splitting_model}")
print(f"   • Scientific Decomposition: {agent_config.scientific_decomposition_model}")
print(f"   • Repository Routing: {agent_config.repository_routing_model}")
print(f"   • Collection Ranking: {agent_config.collection_ranking_model}")
print(f"   • CMR Query Generation: {agent_config.cmr_query_model}")


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


async def test_repository_routing_only(query: str = None, topics: list = None):
    """Test only the repository routing component."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 TESTING REPOSITORY ROUTING ONLY")
    print(f"Query: '{demo_query}'")

    # Use provided topics or get them from topic splitting
    if not topics:
        print("   🔄 Getting topics from topic splitting first...")
        topics = await test_topic_splitting_only(demo_query)

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["repository_routing"])
    component = RepositoryRouterComponent(config=config, debug=True)

    # Test repository routing
    print("\n2️⃣ Repository Routing:")
    routing_output = await component.process(demo_query, topics)

    print("   ✅ Routing Results (per topic):")
    for i, (topic, route) in enumerate(zip(topics, routing_output.routes), start=1):
        print(f"\n   Topic {i}: {topic.title}")
        print(f"     Repositories: {route.repositories}")
        print(f"     Rationales:   {route.rationales}")

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


async def test_collection_ranking_only():
    """Test only the collection ranking component with mock data."""
    print("\n🧪 TESTING COLLECTION RANKING ONLY")

    # Mock collections data for testing
    mock_collections = [
        {
            "concept_id": "C1234567890-LAADS",
            "title": "MODIS Terra Land Surface Temperature",
            "abstract": "Daily land surface temperature data from MODIS Terra satellite",
            "dataset_id": "MOD11A1",
        },
        {
            "concept_id": "C9876543210-LAADS",
            "title": "MODIS Aqua Land Surface Temperature",
            "abstract": "Daily land surface temperature data from MODIS Aqua satellite",
            "dataset_id": "MYD11A1",
        },
        {
            "concept_id": "C5555555555-LPCLOUD",
            "title": "Landsat 8 Surface Temperature",
            "abstract": "Surface temperature from Landsat 8 thermal infrared sensor",
            "dataset_id": "LANDSAT_8_C1",
        },
    ]

    mock_scientific_angle = {
        "title": "Land Surface Temperature",
        "scientific_justification": "Direct measurement of surface heating for climate studies",
    }

    # Initialize component
    config = BaseAgentConfig(model_name=MODEL_CONFIG["collection_ranking"])
    component = CollectionRankingComponent(config=config, debug=True)

    # Test collection ranking
    print("\n6️⃣ Collection Ranking:")
    from akd.agents.data_search.components.collection_ranking import (
        CollectionRankingInputSchema,
    )

    ranking_input = CollectionRankingInputSchema(
        original_query="Find land surface temperature data for climate studies",
        scientific_angle=mock_scientific_angle,
        collections=mock_collections,
        max_collections=2,
    )

    ranking_result = await component.arun(ranking_input)

    print(f"   ✅ Ranked {len(ranking_result.ranked_collections)} collections:")
    print(f"   📋 Summary: {ranking_result.ranking_summary}")

    for i, ranked_col in enumerate(ranking_result.ranked_collections, 1):
        original_collection = mock_collections[ranked_col.collection_index]
        print(
            f"\n   {i}. {original_collection['dataset_id']} - {original_collection['title']}",
        )
        print(f"      Relevance Score: {ranked_col.relevance_score:.2f}")
        print(f"      Reasoning: {ranked_col.ranking_reasoning}")


# =============================================================================
# ORIGINAL TESTING FUNCTIONS (PRESERVED)
# =============================================================================


async def test_individual_components():
    """Test each component individually."""
    demo_query = (
        "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 COMPONENT TESTING")
    print(f"Query: '{demo_query}'")

    # 1. Topic Splitting
    print("\n1️⃣ Topic Splitting:")
    topics_output = await agent.topic_splitting_component.process(demo_query)
    for topic in topics_output.topics:
        print(f"   • {topic.title}")
        print(f"     Context: {topic.functional_context}")

    # 2. Repository Routing
    print("\n2️⃣ Repository Routing:")
    routes = []
    for topic in topics_output.topics:
        routing = await agent.repository_router_component.process(demo_query, topic)
        routes.append(routing.route)

    print("   • Per-topic routing:")
    for i, (topic, route) in enumerate(
        zip(topics_output.topics, routes),
        start=1,
    ):
        print(f"     - {i}. {topic.title}: {route.repositories}")

    # 3. Scientific Decomposition (first CMR topic)
    # Pick first topic that includes CMR
    first_topic = None
    for topic, route in zip(topics_output.topics, routes):
        if any(r.upper() == "CMR" or r == "CMR" for r in route.repositories):
            first_topic = topic
            break
    if first_topic is not None:
        print(f"\n3️⃣ Scientific Decomposition for '{first_topic.title}':")
        decomp_output = await agent.scientific_decomposition_component.process(
            demo_query,
            first_topic,
        )
        for decomp in decomp_output.decompositions:
            print(f"   • {decomp.title}")
            print(f"     Justification: {decomp.scientific_justification}")

        # 4. Known Parameters (first decomposition)
        if decomp_output.decompositions:
            first_decomp = decomp_output.decompositions[0]
            print(f"\n4️⃣ Known Parameters for '{first_decomp.title}':")
            known_params = await agent.known_parameters_component.process(
                demo_query,
                first_topic,
                first_decomp,
            )
            for i, approach in enumerate(known_params.query_approaches, 1):
                print(f"   Approach {i}:")
                approach_dict = approach.model_dump()
                for param, value in approach_dict.items():
                    if value is not None:
                        print(f"     {param}: {value}")

            # 5. Searchable Parameters
            if known_params.query_approaches:
                print("\n5️⃣ Searchable Parameters:")
                searchable_params = await agent.searchable_parameters_component.process(
                    demo_query,
                    first_topic,
                    first_decomp,
                    known_params.query_approaches,
                )
                for i, query in enumerate(searchable_params.searchable_queries, 1):
                    print(f"   Query {i}:")
                    print(f"     Keywords: {', '.join(query.primary_keywords)}")
                    print(f"     Combined: {query.combined_keyword_string}")


async def test_new_workflow():
    """Test the complete new workflow end-to-end."""
    demo_query = (
        "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🆕 TESTING NEW TOPIC-BASED WORKFLOW")
    print(f"🔍 Query: '{demo_query}'")

    # Test the full new workflow
    input_params = DataSearchAgentInputSchema(query=demo_query)
    result = await agent.arun(input_params)

    print("\n📊 SEARCH RESULTS:")
    print(f"   Topics Found: {len(result.topics)}")
    print(f"   Total Data Files: {result.total_results}")
    print(
        f"   Search Duration: {result.search_metadata.get('duration_seconds', 0):.1f}s",
    )
    print(
        f"   Workflow Version: {result.search_metadata.get('workflow_version', 'unknown')}",
    )

    # Display structured results
    for i, topic_result in enumerate(result.topics, 1):
        print(f"\n📌 TOPIC {i}: {topic_result.topic['title']}")
        print(f"   Data Source: {topic_result.data_source}")

        if topic_result.note:
            print(f"   → {topic_result.note}")
        else:
            print(f"   Decompositions: {len(topic_result.decomposition_results)}")

            for j, decomp_result in enumerate(topic_result.decomposition_results, 1):
                print(
                    f"\n   🔬 DECOMPOSITION {j}: {decomp_result.decomposition['title']}",
                )
                print(
                    f"      Scientific Justification: {decomp_result.decomposition['scientific_justification']}",
                )
                print(f"      Query Approaches: {len(decomp_result.query_approaches)}")
                print(f"      Collections Found: {len(decomp_result.collections)}")
                print(f"      Data Files Found: {len(decomp_result.granules)}")

                # Show top collections
                for k, collection in enumerate(decomp_result.collections[:3], 1):
                    title = collection.get("title", "No title")
                    short_name = collection.get("short_name", "N/A")
                    print(f"         {k}. [{short_name}] {title}")

                if len(decomp_result.collections) > 3:
                    print(
                        f"         ... and {len(decomp_result.collections) - 3} more collections",
                    )


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
    elif args.component == "ranking":
        await test_collection_ranking_only()
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
        await test_collection_ranking_only()

        print("\n" + "=" * 80)
        print("ALL INDIVIDUAL COMPONENT TESTS COMPLETED!")
        print("=" * 80)
    else:
        # Default: run full workflow (preserving original behavior)
        print("\n🔄 Running full workflow (use --component to test individual parts)")

        # Test individual components first
        await test_individual_components()

        # Test complete new workflow
        await test_new_workflow()

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
