"""
Loader Demo - Test individual components using captured workflow data.

This demo loads JSON files created by demo_capture.py and provides easy access
to test any component with saved inputs/outputs. Perfect for debugging specific
components without re-running the entire workflow.

Usage:
    python demo_loader.py captured_data/my_file.json
    python demo_loader.py captured_data/my_file.json --component known_parameters --topic 0 --decomp 1
    python demo_loader.py captured_data/my_file.json --list  # Show available data
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from akd.agents.data_search import DataSearchAgent, DataSearchAgentConfig
from akd.agents.data_search.components import ScientificDecomposition, Topic
from akd.agents.data_search.handlers import CMRHandlerConfig
from akd.agents.data_search.handlers.cmr import CMRQueryApproach, CMRSearchableQuery

# Load environment variables
load_dotenv()

print("✅ Imports successful")
print(f"🔑 OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")


class WorkflowLoader:
    """
    Load and test components from captured workflow data.

    This class provides easy access to all component inputs and outputs
    saved by demo_capture.py, allowing you to test individual components
    without re-running the entire pipeline.
    """

    def __init__(self, json_file: str):
        """
        Initialize loader with captured workflow data.

        Args:
            json_file: Path to JSON file created by demo_capture.py
        """
        self.json_file = json_file

        if not Path(json_file).exists():
            raise FileNotFoundError(f"Captured data file not found: {json_file}")

        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.query = self.data["query"]
        self.has_timing_data = "timing_data" in self.data

        print(f"📄 Loaded captured workflow: {Path(json_file).name}")
        print(f"🔍 Query: {self.query}")
        print(f"📊 Topics: {len(self.data['topics'])}")
        if self.has_timing_data:
            print("⏱️  Timing data: Available")
        else:
            print("⏱️  Timing data: Not available (captured with older version)")

        # Initialize agent with same config as capture
        self._setup_agent()

    def _setup_agent(self):
        """Setup agent with configuration matching the captured data."""

        # Extract model config from captured data or use defaults
        model_config = self.data.get("metadata", {}).get(
            "model_config",
            {
                "topic_splitting": "gpt-5-mini",
                "scientific_decomposition": "gpt-5-mini",
                "repository_routing": "gpt-5-mini",
                "collection_ranking": "gpt-5-mini",
                "cmr_query": "gpt-5-mini",
            },
        )

        cmr_handler_config = CMRHandlerConfig(
            collection_search_page_size=20,
            granule_search_page_size=10,
            final_collection_count=5,
            collection_search_timeout=30.0,
            granule_search_timeout=45.0,
            known_parameters_model=model_config.get("cmr_query", "gpt-5-mini"),
            searchable_parameters_model=model_config.get("cmr_query", "gpt-5-mini"),
            approach_filtering_model=model_config.get(
                "collection_ranking",
                "gpt-5-mini",
            ),
            final_ranking_model=model_config.get("collection_ranking", "gpt-5-mini"),
        )

        agent_config = DataSearchAgentConfig(
            debug=True,
            topic_splitting_model=model_config["topic_splitting"],
            scientific_decomposition_model=model_config["scientific_decomposition"],
            repository_routing_model=model_config["repository_routing"],
            cmr=cmr_handler_config,
        )

        self.agent = DataSearchAgent(config=agent_config, debug=True)

    def list_available_data(self):
        """Print summary of all available data for testing."""
        print(f"\n📋 AVAILABLE DATA IN {Path(self.json_file).name}")
        print(f"🔍 Query: {self.query}")
        print(f"📅 Captured: {self.data.get('timestamp', 'Unknown')}")

        metadata = self.data.get("metadata", {})
        print("\n📊 Summary:")
        print(f"   Topics: {metadata.get('total_topics', 0)}")
        print(f"   Decompositions: {metadata.get('total_decompositions', 0)}")
        print(f"   Total Paths: {metadata.get('total_paths', 0)}")
        print(f"   Collections: {metadata.get('total_collections', 0)}")
        print(f"   Granules: {metadata.get('total_granules', 0)}")

        print("\n📝 Topics Available:")
        for topic_data in self.data["topics"]:
            topic_idx = topic_data["index"]
            topic = topic_data["topic"]
            decomp_count = len(topic_data["decompositions"])

            print(f"   Topic {topic_idx}: {topic['title']}")
            print(f"     Decompositions: {decomp_count}")

            if topic_data.get("note"):
                print(f"     Note: {topic_data['note']}")
            else:
                for decomp_data in topic_data["decompositions"]:
                    decomp_idx = decomp_data["index"]
                    decomp = decomp_data["decomposition"]
                    print(f"       Decomp {decomp_idx}: {decomp['title']}")

    def get_topic(self, topic_idx: int) -> Topic:
        """
        Get reconstructed Topic object.

        Args:
            topic_idx: Index of topic to retrieve

        Returns:
            Topic object reconstructed from saved data
        """
        if topic_idx >= len(self.data["topics"]):
            raise IndexError(
                f"Topic index {topic_idx} not found (max: {len(self.data['topics']) - 1})",
            )

        topic_data = self.data["topics"][topic_idx]["topic"]
        return Topic(**topic_data)

    def get_decomposition(
        self,
        topic_idx: int,
        decomp_idx: int,
    ) -> ScientificDecomposition:
        """
        Get reconstructed ScientificDecomposition object.

        Args:
            topic_idx: Index of topic
            decomp_idx: Index of decomposition within topic

        Returns:
            ScientificDecomposition object reconstructed from saved data
        """
        topic_data = self.data["topics"][topic_idx]

        if decomp_idx >= len(topic_data["decompositions"]):
            raise IndexError(
                f"Decomposition index {decomp_idx} not found in topic {topic_idx}",
            )

        decomp_data = topic_data["decompositions"][decomp_idx]["decomposition"]
        return ScientificDecomposition(**decomp_data)

    def get_component_inputs(
        self,
        component_name: str,
        topic_idx: int = 0,
        decomp_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Get all inputs needed for a specific component.

        Args:
            component_name: Name of component to get inputs for
            topic_idx: Index of topic (if applicable)
            decomp_idx: Index of decomposition (if applicable)

        Returns:
            Dictionary with all required inputs for the component
        """
        inputs = {"query": self.query}

        if component_name == "topic_splitting":
            # Topic splitting only needs query
            return inputs

        elif component_name == "repository_routing":
            inputs["topic"] = self.get_topic(topic_idx)
            return inputs

        elif component_name == "scientific_decomposition":
            inputs["topic"] = self.get_topic(topic_idx)
            return inputs

        elif component_name == "known_parameters":
            inputs["topic"] = self.get_topic(topic_idx)
            inputs["decomposition"] = self.get_decomposition(topic_idx, decomp_idx)
            return inputs

        elif component_name == "searchable_parameters":
            inputs["topic"] = self.get_topic(topic_idx)
            inputs["decomposition"] = self.get_decomposition(topic_idx, decomp_idx)
            inputs["query_approaches"] = self.get_component_outputs(
                "known_parameters",
                topic_idx,
                decomp_idx,
            )
            return inputs

        else:
            raise ValueError(f"Unknown component: {component_name}")

    def get_component_outputs(
        self,
        component_name: str,
        topic_idx: int = 0,
        decomp_idx: int = 0,
    ) -> Any:
        """
        Get saved outputs from a specific component.

        Args:
            component_name: Name of component to get outputs for
            topic_idx: Index of topic (if applicable)
            decomp_idx: Index of decomposition (if applicable)

        Returns:
            Component outputs (format depends on component)
        """
        if component_name == "topic_splitting":
            return [Topic(**topic["topic"]) for topic in self.data["topics"]]

        elif component_name == "repository_routing":
            topic_data = self.data["topics"][topic_idx]
            return topic_data["routing"]

        elif component_name == "scientific_decomposition":
            topic_data = self.data["topics"][topic_idx]
            return [
                ScientificDecomposition(**decomp["decomposition"])
                for decomp in topic_data["decompositions"]
            ]

        elif component_name == "known_parameters":
            decomp_data = self.data["topics"][topic_idx]["decompositions"][decomp_idx]
            if "known_params" not in decomp_data:
                raise ValueError(
                    f"No known_params data found for topic {topic_idx}, decomposition {decomp_idx}",
                )

            # Reconstruct query approaches
            query_approaches = []
            for qa_data in decomp_data["known_params"]["query_approaches"]:
                query_approaches.append(CMRQueryApproach(**qa_data))
            return query_approaches

        elif component_name == "searchable_parameters":
            decomp_data = self.data["topics"][topic_idx]["decompositions"][decomp_idx]
            if "searchable_params" not in decomp_data:
                raise ValueError(
                    f"No searchable_params data found for topic {topic_idx}, decomposition {decomp_idx}",
                )

            # Reconstruct searchable queries
            searchable_queries = []
            for sq_data in decomp_data["searchable_params"]["searchable_queries"]:
                searchable_queries.append(CMRSearchableQuery(**sq_data))
            return searchable_queries

        elif component_name == "collection_search":
            decomp_data = self.data["topics"][topic_idx]["decompositions"][decomp_idx]
            return decomp_data.get("collections_raw", [])

        elif component_name == "granule_search":
            decomp_data = self.data["topics"][topic_idx]["decompositions"][decomp_idx]
            return decomp_data.get("granules", [])

        else:
            raise ValueError(f"Unknown component: {component_name}")

    async def test_component(
        self,
        component_name: str,
        topic_idx: int = 0,
        decomp_idx: int = 0,
    ) -> Any:
        """
        Test a specific component with appropriate saved inputs.

        Args:
            component_name: Name of component to test
            topic_idx: Index of topic (if applicable)
            decomp_idx: Index of decomposition (if applicable)

        Returns:
            Fresh component output from running with saved inputs
        """
        print(f"\n🧪 TESTING COMPONENT: {component_name.upper()}")
        print(f"📍 Using: Topic {topic_idx}, Decomposition {decomp_idx}")

        if component_name == "topic_splitting":
            print(f"🔍 Input: {self.query}")
            result = await self.agent.topic_splitting_component.process(self.query)
            print(f"✅ Found {len(result.topics)} topics")
            for i, topic in enumerate(result.topics):
                print(f"   {i}. {topic.title}")
            return result

        elif component_name == "repository_routing":
            topic = self.get_topic(topic_idx)
            print(f"🔍 Input: {topic.title}")
            result = await self.agent.repository_router_component.process(
                self.query,
                topic,
            )
            print(f"✅ Repositories: {result.route.repositories}")
            return result

        elif component_name == "scientific_decomposition":
            topic = self.get_topic(topic_idx)
            print(f"🔍 Input: {topic.title}")
            result = await self.agent.scientific_decomposition_component.process(
                self.query,
                topic,
            )
            print(f"✅ Found {len(result.decompositions)} decompositions")
            for i, decomp in enumerate(result.decompositions):
                print(f"   {i}. {decomp.title}")
            return result

        elif component_name == "known_parameters":
            topic = self.get_topic(topic_idx)
            decomposition = self.get_decomposition(topic_idx, decomp_idx)
            print(f"🔍 Input: {topic.title} → {decomposition.title}")
            result = await self.agent.known_parameters_component.process(
                self.query,
                topic,
                decomposition,
            )
            print(f"✅ Generated {len(result.query_approaches)} query approaches")
            return result

        elif component_name == "searchable_parameters":
            topic = self.get_topic(topic_idx)
            decomposition = self.get_decomposition(topic_idx, decomp_idx)
            query_approaches = self.get_component_outputs(
                "known_parameters",
                topic_idx,
                decomp_idx,
            )
            print(
                f"🔍 Input: {topic.title} → {decomposition.title} + {len(query_approaches)} approaches",
            )
            result = await self.agent.searchable_parameters_component.process(
                self.query,
                topic,
                decomposition,
                query_approaches,
            )
            print(f"✅ Generated {len(result.searchable_queries)} searchable queries")
            return result

        else:
            raise ValueError(f"Unknown component: {component_name}")

    async def compare_with_saved(
        self,
        component_name: str,
        topic_idx: int = 0,
        decomp_idx: int = 0,
    ):
        """
        Test component and compare output with saved results.

        Args:
            component_name: Name of component to test
            topic_idx: Index of topic (if applicable)
            decomp_idx: Index of decomposition (if applicable)
        """
        print(f"\n🔄 COMPARING FRESH vs SAVED: {component_name.upper()}")

        # Get fresh result
        fresh_result = await self.test_component(component_name, topic_idx, decomp_idx)

        # Get saved result
        try:
            saved_result = self.get_component_outputs(
                component_name,
                topic_idx,
                decomp_idx,
            )
            print("\n📊 COMPARISON:")

            if component_name == "topic_splitting":
                print(f"   Fresh topics: {len(fresh_result.topics)}")
                print(f"   Saved topics: {len(saved_result)}")

            elif component_name == "known_parameters":
                print(f"   Fresh approaches: {len(fresh_result.query_approaches)}")
                print(f"   Saved approaches: {len(saved_result)}")

            elif component_name == "searchable_parameters":
                print(f"   Fresh queries: {len(fresh_result.searchable_queries)}")
                print(f"   Saved queries: {len(saved_result)}")

            # Add more comparisons as needed

        except Exception as e:
            print(f"⚠️ Could not load saved results: {e}")

    def show_timing_summary(self):
        """Display timing summary from captured data."""
        if not self.has_timing_data:
            print("⚠️ No timing data available in this capture file")
            return

        timing_data = self.data["timing_data"]
        summary = timing_data.get("summary", {})

        print(f"\n⏱️  TIMING SUMMARY - {Path(self.json_file).name}")
        print(f"🕐 Total Duration: {timing_data.get('total_duration', 0):.2f}s")
        print(f"🧩 Components: {summary.get('total_components', 0)}")

        # Component breakdown
        component_breakdown = summary.get("component_breakdown", {})
        if component_breakdown:
            print("\n📊 COMPONENT BREAKDOWN:")
            for comp_name, stats in component_breakdown.items():
                print(
                    f"   {comp_name:25} {stats['total_time']:8.2f}s ({stats['percentage']:5.1f}%) [{stats['call_count']} calls]",
                )

        # Path breakdown
        path_breakdown = summary.get("path_breakdown", {})
        if path_breakdown:
            print("\n🛤️  PATH BREAKDOWN:")
            for path_name, stats in path_breakdown.items():
                print(
                    f"   {path_name:15} {stats['total_time']:8.2f}s ({stats['percentage']:5.1f}%) [{stats['decomposition_count']} decomps]",
                )

        # Slowest components
        slowest = summary.get("slowest_components", [])
        if slowest:
            print("\n🐌 SLOWEST COMPONENTS:")
            for i, comp in enumerate(slowest[:3], 1):
                print(
                    f"   {i}. {comp['component']:20} {comp['total_time']:8.2f}s ({comp['percentage']:5.1f}%)",
                )

    def analyze_component_timing(self, component_name: str):
        """Analyze detailed timing for a specific component."""
        if not self.has_timing_data:
            print("⚠️ No timing data available in this capture file")
            return

        timing_data = self.data["timing_data"]
        components = timing_data.get("components", {})

        if component_name not in components:
            print(f"❌ No timing data found for component: {component_name}")
            available = list(components.keys())
            print(f"💡 Available components: {', '.join(available)}")
            return

        comp_data = components[component_name]
        metadata_entries = comp_data.get("metadata", [])

        print(f"\n🔍 DETAILED TIMING ANALYSIS: {component_name.upper()}")
        print(f"📊 Total Time: {comp_data['total_duration']:.2f}s")
        print(f"📈 Call Count: {comp_data['call_count']}")
        print(
            f"⏱️  Average Time: {comp_data['total_duration'] / comp_data['call_count']:.2f}s per call",
        )

        if metadata_entries:
            print("\n📝 INDIVIDUAL CALLS:")
            for i, entry in enumerate(metadata_entries, 1):
                duration = entry.get("duration", 0)
                topic_idx = entry.get("topic_idx")
                decomp_idx = entry.get("decomp_idx")

                path_info = ""
                if topic_idx is not None:
                    path_info = f" (Topic {topic_idx}"
                    if decomp_idx is not None:
                        path_info += f", Decomp {decomp_idx}"
                    path_info += ")"

                print(f"   {i:2d}. {duration:6.2f}s{path_info}")

                # Show relevant metadata
                for key, value in entry.items():
                    if key not in ["duration", "start_time", "topic_idx", "decomp_idx"]:
                        print(f"       {key}: {value}")

    def compare_timing_across_paths(self):
        """Compare timing across different topic/decomposition paths."""
        if not self.has_timing_data:
            print("⚠️ No timing data available in this capture file")
            return

        timing_data = self.data["timing_data"]
        paths = timing_data.get("paths", {})

        if not paths:
            print("❌ No path-level timing data available")
            return

        print("\n🛤️  PATH TIMING COMPARISON")

        # Sort paths by total duration
        sorted_paths = sorted(
            paths.items(),
            key=lambda x: x[1]["total_duration"],
            reverse=True,
        )

        for path_name, path_data in sorted_paths:
            duration = path_data["total_duration"]
            decomp_count = len(path_data.get("decompositions", {}))
            print(f"\n📍 {path_name.upper()}:")
            print(f"   Total Time: {duration:.2f}s")
            print(f"   Decompositions: {decomp_count}")
            if decomp_count > 0:
                print(f"   Avg per Decomp: {duration / decomp_count:.2f}s")

            # Show component breakdown for this path
            components = path_data.get("components", {})
            if components:
                print("   Components:")
                for comp_name, comp_stats in components.items():
                    comp_time = comp_stats["total_duration"]
                    comp_calls = comp_stats["call_count"]
                    print(f"     {comp_name:20} {comp_time:6.2f}s ({comp_calls} calls)")


async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Test individual components using captured workflow data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python demo_loader.py captured_data/my_file.json --list
  python demo_loader.py captured_data/my_file.json --component topic_splitting
  python demo_loader.py captured_data/my_file.json --component known_parameters --topic 0 --decomp 1
  python demo_loader.py captured_data/my_file.json --component searchable_parameters --topic 1 --decomp 0 --compare

  # Timing analysis
  python demo_loader.py captured_data/my_file.json --timing
  python demo_loader.py captured_data/my_file.json --analyze-timing known_parameters
  python demo_loader.py captured_data/my_file.json --timing-paths
        """,
    )

    parser.add_argument(
        "json_file",
        help="Path to JSON file created by demo_capture.py",
    )

    parser.add_argument(
        "--component",
        "-c",
        choices=[
            "topic_splitting",
            "repository_routing",
            "scientific_decomposition",
            "known_parameters",
            "searchable_parameters",
            "collection_ranking",
        ],
        help="Component to test",
    )

    parser.add_argument(
        "--topic",
        "-t",
        type=int,
        default=0,
        help="Topic index to use (default: 0)",
    )

    parser.add_argument(
        "--decomp",
        "-d",
        type=int,
        default=0,
        help="Decomposition index to use (default: 0)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available data",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fresh result with saved result",
    )

    parser.add_argument(
        "--timing",
        "-t",
        action="store_true",
        help="Show timing summary from captured data",
    )

    parser.add_argument(
        "--analyze-timing",
        "-at",
        metavar="COMPONENT",
        help="Analyze detailed timing for a specific component",
    )

    parser.add_argument(
        "--timing-paths",
        "-tp",
        action="store_true",
        help="Compare timing across different topic/decomposition paths",
    )

    args = parser.parse_args()

    try:
        # Load the workflow data
        loader = WorkflowLoader(args.json_file)

        if args.list:
            loader.list_available_data()

        elif args.timing:
            loader.show_timing_summary()

        elif args.analyze_timing:
            loader.analyze_component_timing(args.analyze_timing)

        elif args.timing_paths:
            loader.compare_timing_across_paths()

        elif args.component:
            if args.compare:
                await loader.compare_with_saved(args.component, args.topic, args.decomp)
            else:
                await loader.test_component(args.component, args.topic, args.decomp)

        else:
            # Interactive mode - show available data
            loader.list_available_data()
            print("\n💡 Use --component to test a specific component")
            print("💡 Use --timing to see timing summary")
            print("💡 Use --list to see this summary again")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
