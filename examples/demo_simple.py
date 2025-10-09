import argparse
import asyncio
import os

from dotenv import load_dotenv

from akd.agents._base import BaseAgentConfig
from akd.agents.data_search import CMRDataSearchAgent, CMRDataSearchAgentConfig
from akd.agents.data_search.components import (
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
    cmr_query_model=MODEL_CONFIG["cmr_query"],
)

# Initialize the agent with our configured models
agent = CMRDataSearchAgent(config=agent_config, debug=True)

print("\n🤖 Agent initialized with component-specific models:")
print(f"   • Topic Splitting: {agent_config.topic_splitting_model}")
print(f"   • Scientific Decomposition: {agent_config.scientific_decomposition_model}")
print(f"   • Repository Routing: {agent_config.repository_routing_model}")
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


# =============================================================================
# ORIGINAL TESTING FUNCTIONS (PRESERVED)
# =============================================================================


async def test_individual_components():
    """Test each component individually - single pathway for speed."""
    demo_query = (
        "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🧪 COMPONENT TESTING - SINGLE PATHWAY")
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
    routing = await agent.repository_router_component.process(demo_query, first_topic)
    print(f"   • Repositories: {routing.route.repositories}")
    print(f"   • Rationales: {routing.route.rationales}")

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

    # 4. Known Parameters (first decomposition)
    print(f"\n4️⃣ Known Parameters for '{first_decomp.title}':")
    known_params = await agent.known_parameters_component.process(
        demo_query,
        first_topic,
        first_decomp,
    )
    print(f"   Generated {len(known_params.query_approaches)} approaches:")
    for i, approach in enumerate(known_params.query_approaches, 1):
        print(f"   Approach {i}:")
        approach_dict = approach.model_dump()
        for param, value in approach_dict.items():
            if value is not None:
                print(f"     {param}: {value}")

    # 5. Searchable Parameters (all approaches from first decomposition)
    print(
        f"\n5️⃣ Searchable Parameters for all {len(known_params.query_approaches)} approaches:",
    )
    searchable_params = await agent.searchable_parameters_component.process(
        demo_query,
        first_topic,
        first_decomp,
        known_params.query_approaches,
    )
    print(
        f"   Generated {len(searchable_params.searchable_queries)} searchable queries:",
    )
    for i, query in enumerate(searchable_params.searchable_queries, 1):
        print(f"   Query {i}:")
        print(f"     Keywords: {', '.join(query.primary_keywords)}")
        print(f"     Combined: {query.combined_keyword_string}")


async def test_new_workflow(query: str = None):
    """Test the complete new workflow step-by-step with single path processing."""
    demo_query = (
        query
        or "Help me gather data to study the flood risk of the lower Mississippi basin"
    )

    print("\n🆕 TESTING NEW TOPIC-BASED WORKFLOW - STEP BY STEP, SINGLE PATH")
    print(f"🔍 Query: '{demo_query}'")

    # Step 1: Topic Splitting
    print("\n1️⃣ Topic Splitting:")
    topics_output = await agent.topic_splitting_component.process(demo_query)
    print(f"   ✅ Identified {len(topics_output.topics)} topics:")
    for i, topic in enumerate(topics_output.topics, 1):
        print(f"   {i}. **{topic.title}**")
        print(f"      Context: {topic.functional_context}")

    # Select first topic for single-path processing
    if not topics_output.topics:
        print("   ❌ No topics found, ending workflow")
        return

    first_topic = topics_output.topics[0]
    print(f"\n   📌 Selected for processing: {first_topic.title}")

    # Step 2: Repository Routing (for selected topic only)
    print("\n2️⃣ Repository Routing:")
    routing_output = await agent.repository_router_component.process(
        demo_query,
        first_topic,
    )
    print(f"   ✅ Repository routing for '{first_topic.title}':")
    print(f"      Repositories: {routing_output.route.repositories}")
    print(f"      Rationales: {routing_output.route.rationales}")

    # Check if CMR is selected
    from akd.agents.data_search.components.repository_router import NASARepositoryEnum

    has_cmr = NASARepositoryEnum.CMR in routing_output.route.repositories
    if not has_cmr:
        print(
            f"   ⚠️  CMR not selected, ending workflow (routed to: {routing_output.route.repositories})",
        )
        return

    # Step 3: Scientific Decomposition (for selected topic only)
    print("\n3️⃣ Scientific Decomposition:")
    decomp_output = await agent.scientific_decomposition_component.process(
        demo_query,
        first_topic,
    )
    print(f"   ✅ Generated {len(decomp_output.decompositions)} decompositions:")
    for i, decomp in enumerate(decomp_output.decompositions, 1):
        print(f"   {i}. **{decomp.title}**")
        print(f"      Scientific Justification: {decomp.scientific_justification}")

    # Select first decomposition for single-path processing
    if not decomp_output.decompositions:
        print("   ❌ No decompositions found, ending workflow")
        return

    first_decomp = decomp_output.decompositions[0]
    print(f"\n   📌 Selected for processing: {first_decomp.title}")

    # Step 4: Known Parameters (for selected topic + decomposition)
    print("\n4️⃣ Known Parameters:")
    known_params_output = await agent.known_parameters_component.process(
        demo_query,
        first_topic,
        first_decomp,
    )
    print(
        f"   ✅ Generated {len(known_params_output.query_approaches)} query approaches:",
    )
    print(f"   📋 Reasoning: {known_params_output.reasoning}")

    for i, approach in enumerate(known_params_output.query_approaches, 1):
        print(f"\n   Approach {i}:")
        approach_dict = approach.model_dump()
        for param, value in approach_dict.items():
            if value is not None:
                print(f"     {param}: {value}")

    # Step 5: Searchable Parameters
    print("\n5️⃣ Searchable Parameters:")
    searchable_output = await agent.searchable_parameters_component.process(
        demo_query,
        first_topic,
        first_decomp,
        known_params_output.query_approaches,
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

    # Step 6: Collection Search
    print("\n6️⃣ Collection Search:")
    collections = []
    for i, query in enumerate(searchable_output.searchable_queries, 1):
        print(f"   Executing query {i}: {query.combined_keyword_string}")
        try:
            search_params = query.get_mcp_parameters()
            search_params["page_size"] = agent.config.collection_search_page_size

            tool_input = agent.collection_search_tool.input_schema(**search_params)
            result = await agent.collection_search_tool.arun(tool_input)

            if hasattr(result, "collections") and result.collections:
                collections.extend(result.collections)
                print(f"     → Found {len(result.collections)} collections")
            else:
                print("     → No collections found")
        except Exception as e:
            print(f"     → Query failed: {e}")

    print(f"   ✅ Total collections found: {len(collections)}")

    # Step 7: Collection Ranking (if too many collections)
    print("\n7️⃣ Collection Ranking:")
    # Limit collections (no LLM-based ranking in demo)
    if len(collections) > agent.config.max_collections_to_search:
        print(
            f"   Too many collections ({len(collections)}), truncating to top {agent.config.max_collections_to_search}",
        )
        collections = collections[: agent.config.max_collections_to_search]
    else:
        print(f"   Using all {len(collections)} collections (within limit)")

    # Show selected collections
    for i, collection in enumerate(collections[:3], 1):
        title = collection.get("title", "No title")
        short_name = collection.get("short_name", "N/A")
        print(f"   {i}. [{short_name}] {title}")

    if len(collections) > 3:
        print(f"   ... and {len(collections) - 3} more collections")

    # Step 8: Granule Search
    print("\n8️⃣ Granule Search:")
    all_granules = []

    for i, collection in enumerate(collections, 1):
        concept_id = collection.get("concept_id")
        if not concept_id:
            continue

        print(
            f"   Searching granules for collection {i}: {collection.get('short_name', 'N/A')}",
        )

        try:
            granule_params = {
                "collection_concept_id": concept_id,
                "page_size": agent.config.granule_search_page_size,
            }

            granule_search_params = agent.granule_search_tool.input_schema(
                **granule_params,
            )
            result = await agent.granule_search_tool.arun(granule_search_params)

            if hasattr(result, "results") and result.results.get("granules"):
                granules = result.results["granules"]
                all_granules.extend(granules)
                print(f"     → Found {len(granules)} granules")
            else:
                print("     → No granules found")
        except Exception as e:
            print(f"     → Granule search failed: {e}")

    print(f"   ✅ Total granules found: {len(all_granules)}")

    # Final Summary
    print("\n📊 SINGLE-PATH WORKFLOW RESULTS:")
    print(f"   Selected Topic: {first_topic.title}")
    print(f"   Selected Decomposition: {first_decomp.title}")
    print(f"   Query Approaches: {len(known_params_output.query_approaches)}")
    print(f"   Searchable Queries: {len(searchable_output.searchable_queries)}")
    print(f"   Collections Found: {len(collections)}")
    print(f"   Data Files Found: {len(all_granules)}")

    if all_granules:
        print("\n   📁 Sample granules:")
        for i, granule in enumerate(all_granules[:3], 1):
            title = granule.get("title", "No title")
            print(f"   {i}. {title}")

        if len(all_granules) > 3:
            print(f"   ... and {len(all_granules) - 3} more granules")


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
