#!/usr/bin/env python3
"""
CLI interface for the testing framework.

This script provides command-line access to all testing functionality,
including running test suites, evaluating results, and generating reports.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from evaluators.decomposition_evaluator import DecompositionEvaluator
from evaluators.topic_evaluator import TopicEvaluator
from loguru import logger
from test_data_manager import TestDataManager
from test_models import TestCase, TestCategory
from test_runner import TestRunner

# Load environment
load_dotenv()


class TestCLI:
    """
    Command-line interface for the testing framework.
    """

    def __init__(self):
        """Initialize the CLI."""
        # Set up default paths relative to the testing directory
        testing_dir = Path(__file__).parent
        self.default_csv_path = testing_dir / "test_data" / "queries.csv"
        self.default_output_dir = testing_dir / "test_runs"
        self.default_capture_dir = testing_dir / "captured_data"

        # Initialize components
        self.data_manager = None
        self.test_runner = None

    def _setup_components(self, csv_path: str, output_dir: str, capture_dir: str):
        """Set up the testing components."""
        if not self.data_manager:
            self.data_manager = TestDataManager(csv_path)

        if not self.test_runner:
            self.test_runner = TestRunner(
                output_dir=output_dir,
                capture_data_dir=capture_dir,
                debug=True,
            )

    async def run_full_suite(
        self,
        csv_path: str,
        output_dir: str,
        capture_dir: str,
        max_tests: Optional[int] = None,
        category: Optional[str] = None,
    ):
        """Run the full test suite."""
        print("🚀 Starting full test suite...")

        # Setup components
        self._setup_components(csv_path, output_dir, capture_dir)

        # Load test cases
        print("📋 Loading test cases...")
        test_cases = self.data_manager.load_test_cases()

        # Filter by category if specified
        if category:
            try:
                cat_enum = TestCategory(category.lower())
                test_cases = self.data_manager.get_test_cases_by_category(cat_enum)
                print(
                    f"🏷️  Filtered to {len(test_cases)} test cases in category '{category}'",
                )
            except ValueError:
                print(f"❌ Invalid category: {category}")
                print(f"Valid categories: {[c.value for c in TestCategory]}")
                return

        # Limit test cases if requested
        if max_tests and max_tests < len(test_cases):
            test_cases = test_cases[:max_tests]
            print(f"🔢 Limited to first {max_tests} test cases")

        print(f"🧪 Running {len(test_cases)} test cases...")

        # Run the test suite
        results = await self.test_runner.run_test_suite(test_cases)

        # Print summary
        print("\n📊 TEST SUITE RESULTS")
        print(f"🔍 Run ID: {results.summary.run_id}")
        print(f"📈 Total Tests: {results.summary.total_tests}")
        print(f"✅ Passed: {results.summary.passed_tests}")
        print(f"❌ Failed: {results.summary.failed_tests}")
        print(f"🚨 Errors: {results.summary.error_tests}")
        print(f"📊 Success Rate: {results.summary.success_rate:.1f}%")
        print(f"⏱️  Duration: {results.summary.total_duration:.1f}s")

        return results

    async def run_single_test(
        self,
        query: str,
        csv_path: str,
        output_dir: str,
        capture_dir: str,
    ):
        """Run a single test by query."""
        print("🧪 Running single test...")
        print(f"❓ Query: {query}")

        # Setup components
        self._setup_components(csv_path, output_dir, capture_dir)

        # Find the test case
        test_case = self.data_manager.get_test_case_by_query(query)
        if not test_case:
            print(f"❌ No test case found for query: {query}")
            print("💡 Available queries:")
            test_cases = self.data_manager.load_test_cases()
            for i, tc in enumerate(test_cases[:5], 1):
                print(f"   {i}. {tc.query[:80]}...")
            return

        # Run the test
        results = await self.test_runner.run_single_test(test_case)

        # Print results
        print("\n📊 SINGLE TEST RESULTS")
        print(f"🔍 Test ID: {test_case.test_id}")
        print(
            f"✅ Execution: {'SUCCESS' if results.test_results[0].execution_success else 'FAILED'}",
        )
        if results.test_results[0].execution_error:
            print(f"🚨 Error: {results.test_results[0].execution_error}")
        print(f"⏱️  Duration: {results.test_results[0].execution_time:.1f}s")

        return results

    async def run_fast_smoke_test(
        self,
        query: Optional[str],
        csv_path: str,
        output_dir: str,
        capture_dir: str,
    ):
        """Run a fast smoke test using single-path execution."""
        print("⚡ Running fast smoke test...")
        print("📊 Mode: Single-path, gpt-5-nano, [0] selection at all branches")

        # Setup components
        self._setup_components(csv_path, output_dir, capture_dir)

        # If no query provided, use first test case
        if not query:
            print("💡 No query provided, using first test case...")
            test_cases = self.data_manager.load_test_cases()
            if not test_cases:
                print("❌ No test cases found")
                return
            test_case = test_cases[0]
            print(f"❓ Using query: {test_case.query[:80]}...")
        else:
            # Find the test case
            test_case = self.data_manager.get_test_case_by_query(query)
            if not test_case:
                print(f"❌ No test case found for query: {query}")
                print("💡 Using query without expected results...")
                # Create a minimal test case
                test_case = TestCase(
                    test_id=f"adhoc_{hash(query) % 10000}",
                    query=query,
                    category=TestCategory.EARTH_SCIENCE,
                    expected_topics=[],
                    expected_decompositions=[],
                )

        # Run the fast smoke test
        results = await self.test_runner.run_fast_smoke_test(test_case)

        # Print results
        print("\n⚡ FAST SMOKE TEST RESULTS")
        print(f"🔍 Test ID: {test_case.test_id}")
        print(
            f"✅ Execution: {'SUCCESS' if results.test_results[0].execution_success else 'FAILED'}",
        )
        if results.test_results[0].execution_error:
            print(f"🚨 Error: {results.test_results[0].execution_error}")
        print(f"⏱️  Duration: {results.test_results[0].execution_time:.1f}s")
        print(
            f"💾 Results saved to: {results.captured_data_files[0] if results.captured_data_files else 'N/A'}",
        )

        return results

    async def evaluate_captured_data(
        self,
        captured_file: str,
        csv_path: str,
        component: str = "topic_splitting",
    ):
        """Evaluate captured data using LLM evaluators."""
        print("🔍 Evaluating captured data...")
        print(f"📁 File: {captured_file}")
        print(f"🧩 Component: {component}")

        # Setup data manager
        self._setup_components(csv_path, ".", ".")

        # Load captured data
        try:
            from demo_loader import WorkflowLoader

            loader = WorkflowLoader(captured_file)
        except Exception as e:
            print(f"❌ Failed to load captured data: {e}")
            return

        # Get the original query from captured data
        query = loader.query

        # Find the corresponding test case
        test_case = self.data_manager.get_test_case_by_query(query)
        if not test_case:
            print(f"❌ No test case found for query: {query}")
            return

        print(f"✅ Found test case: {test_case.test_id}")

        # Run evaluation based on component
        if component == "topic_splitting":
            await self._evaluate_topics(loader, test_case)
        elif component == "scientific_decomposition":
            await self._evaluate_decompositions(loader, test_case)
        else:
            print(f"❌ Evaluation not implemented for component: {component}")

    async def _evaluate_topics(self, loader, test_case: TestCase):
        """Evaluate topic splitting results."""
        print("\n🔍 EVALUATING TOPICS")

        try:
            # Get actual topics from captured data
            topics_data = loader.data.get("topics", [])
            if not topics_data:
                print("❌ No topics found in captured data")
                return

            # Extract topic objects
            actual_topics = [topic_data["topic"] for topic_data in topics_data]

            # Initialize evaluator
            evaluator = TopicEvaluator()

            # Run evaluation
            result = await evaluator.evaluate_detailed(
                actual_topics=actual_topics,
                expected_topics=test_case.expected_topics,
                query=test_case.query,
            )

            # Print results
            print("📊 EVALUATION RESULTS:")
            print(f"   Decision: {'✅ PASS' if result.passed else '❌ FAIL'}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Expected Topics: {len(test_case.expected_topics)}")
            print(f"   Actual Topics: {len(actual_topics)}")

            print("\n💭 REASONING:")
            print(f"   {result.reasoning}")

            # Print specific feedback
            if result.specific_feedback:
                coverage = result.specific_feedback.get("topic_coverage_analysis", {})
                if coverage:
                    print("\n📈 COVERAGE ANALYSIS:")
                    covered = coverage.get("covered_topics", [])
                    missing = coverage.get("missing_topics", [])

                    if covered:
                        print(f"   ✅ Covered topics: {len(covered)}")
                        for c in covered:
                            print(
                                f"      • {c['expected']} → {c['matched_with']} (score: {c['score']:.2f})",
                            )

                    if missing:
                        print(f"   ❌ Missing topics: {len(missing)}")
                        for m in missing:
                            print(f"      • {m}")

        except Exception as e:
            print(f"❌ Topic evaluation failed: {e}")

    async def _evaluate_decompositions(self, loader, test_case: TestCase):
        """Evaluate scientific decomposition results."""
        print("\n🔬 EVALUATING DECOMPOSITIONS")

        try:
            # Get decompositions from all topics
            all_decompositions = []
            topics_data = loader.data.get("topics", [])

            for topic_data in topics_data:
                decomp_results = topic_data.get("decompositions", [])
                for decomp_result in decomp_results:
                    decomp = decomp_result.get("decomposition")
                    if decomp:
                        all_decompositions.append(decomp)

            if not all_decompositions:
                print("❌ No decompositions found in captured data")
                return

            # Initialize evaluator
            evaluator = DecompositionEvaluator()

            # Run evaluation
            result = await evaluator.evaluate_detailed(
                actual_decompositions=all_decompositions,
                expected_decompositions=test_case.expected_decompositions,
                query=test_case.query,
            )

            # Print results
            print("📊 EVALUATION RESULTS:")
            print(f"   Decision: {'✅ PASS' if result.passed else '❌ FAIL'}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(
                f"   Expected Decompositions: {len(test_case.expected_decompositions)}",
            )
            print(f"   Actual Decompositions: {len(all_decompositions)}")

            min_expected = len(
                [d for d in test_case.expected_decompositions if d.is_minimum],
            )
            print(f"   Minimum Required: {min_expected}")

            print("\n💭 REASONING:")
            print(f"   {result.reasoning}")

            # Print minimum coverage
            if result.specific_feedback:
                min_coverage = result.specific_feedback.get("minimum_coverage", {})
                if min_coverage:
                    print("\n⭐ MINIMUM REQUIREMENTS:")
                    print(f"   Total minimum required: {min_coverage['total_minimum']}")
                    print(f"   Covered: {min_coverage['covered_count']}")
                    print(f"   Missing: {min_coverage['missing_count']}")

                    if min_coverage["covered"]:
                        print("   ✅ Covered minimum decompositions:")
                        for c in min_coverage["covered"]:
                            print(f"      • {c}")

                    if min_coverage["missing"]:
                        print("   ❌ Missing minimum decompositions:")
                        for m in min_coverage["missing"]:
                            print(f"      • {m}")

        except Exception as e:
            print(f"❌ Decomposition evaluation failed: {e}")

    def list_test_cases(self, csv_path: str, category: Optional[str] = None):
        """List available test cases."""
        print("📋 Available test cases...")

        # Setup data manager
        self._setup_components(csv_path, ".", ".")

        # Load and show summary
        self.data_manager.print_summary()

        # Show individual test cases
        test_cases = self.data_manager.load_test_cases()

        # Filter by category if specified
        if category:
            try:
                cat_enum = TestCategory(category.lower())
                test_cases = self.data_manager.get_test_cases_by_category(cat_enum)
                print(f"\n🏷️  Filtered to category: {category}")
            except ValueError:
                print(f"❌ Invalid category: {category}")
                return

        print("\n📝 Test Cases:")
        for i, tc in enumerate(test_cases, 1):
            min_count = len(tc.minimum_decompositions)
            print(f"   {i:2d}. {tc.query[:80]}...")
            print(
                f"       Topics: {len(tc.expected_topics)}, Decomps: {len(tc.expected_decompositions)} (min: {min_count})",
            )

    def list_test_runs(self, output_dir: str):
        """List available test runs."""
        print("📊 Available test runs...")

        self._setup_components(str(self.default_csv_path), output_dir, ".")
        run_ids = self.test_runner.list_test_runs()

        if not run_ids:
            print(f"❌ No test runs found in {output_dir}")
            return

        print(f"📁 Found {len(run_ids)} test runs:")
        for i, run_id in enumerate(run_ids, 1):
            print(f"   {i:2d}. {run_id}")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Testing framework for data search agent components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available test cases
  python test_cli.py --list

  # Run full test suite (limited to 2 test cases)
  python test_cli.py --mode full --max-tests 2

  # Run single test
  python test_cli.py --mode single --query "urbanization heat island"

  # Run fast smoke test (uses first test case)
  python test_cli.py --mode fast-smoke

  # Run fast smoke test with custom query
  python test_cli.py --mode fast-smoke --query "sea ice extent"

  # Evaluate captured data
  python test_cli.py --mode evaluate --captured-file captured_data/my_file.json --component topic_splitting

  # List test runs
  python test_cli.py --list-runs
        """,
    )

    # Core arguments
    parser.add_argument(
        "--mode",
        "-m",
        choices=["full", "single", "fast-smoke", "evaluate", "component"],
        help="Testing mode to run",
    )

    parser.add_argument(
        "--csv",
        default="test_data/queries.csv",
        help="Path to test data CSV file",
    )

    parser.add_argument(
        "--output-dir",
        default="test_runs",
        help="Directory for test run outputs",
    )

    parser.add_argument(
        "--capture-dir",
        default="captured_data",
        help="Directory for captured workflow data",
    )

    # Test selection
    parser.add_argument(
        "--max-tests",
        type=int,
        help="Maximum number of test cases to run",
    )

    parser.add_argument(
        "--category",
        help="Filter test cases by category (earth, space, etc.)",
    )

    parser.add_argument(
        "--query",
        help="Specific query to test (for single mode)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--captured-file",
        help="Path to captured workflow data file",
    )

    parser.add_argument(
        "--component",
        choices=[
            "topic_splitting",
            "scientific_decomposition",
            "known_parameters",
            "searchable_parameters",
        ],
        default="topic_splitting",
        help="Component to evaluate",
    )

    # List arguments
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available test cases",
    )

    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available test runs",
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = TestCLI()

    try:
        # Convert relative paths to absolute paths
        csv_path = str(Path(args.csv).resolve())
        output_dir = str(Path(args.output_dir).resolve())
        capture_dir = str(Path(args.capture_dir).resolve())

        if args.list:
            cli.list_test_cases(csv_path, args.category)

        elif args.list_runs:
            cli.list_test_runs(output_dir)

        elif args.mode == "full":
            await cli.run_full_suite(
                csv_path=csv_path,
                output_dir=output_dir,
                capture_dir=capture_dir,
                max_tests=args.max_tests,
                category=args.category,
            )

        elif args.mode == "single":
            if not args.query:
                print("❌ --query is required for single mode")
                return
            await cli.run_single_test(
                query=args.query,
                csv_path=csv_path,
                output_dir=output_dir,
                capture_dir=capture_dir,
            )

        elif args.mode == "fast-smoke":
            await cli.run_fast_smoke_test(
                query=args.query,  # Optional - uses first test case if not provided
                csv_path=csv_path,
                output_dir=output_dir,
                capture_dir=capture_dir,
            )

        elif args.mode == "evaluate":
            if not args.captured_file:
                print("❌ --captured-file is required for evaluate mode")
                return
            await cli.evaluate_captured_data(
                captured_file=args.captured_file,
                csv_path=csv_path,
                component=args.component,
            )

        else:
            # Default action - show help
            parser.print_help()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("CLI error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
