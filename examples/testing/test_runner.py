"""
Test Runner for executing test cases and capturing results.

This module provides the TestRunner class which executes test queries
using the data search agent and captures the results for evaluation.
"""

import json

# Import the demo infrastructure we'll reuse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from demo_capture import capture_fast_smoke_workflow, capture_full_workflow
from demo_loader import WorkflowLoader
from dotenv import load_dotenv

# Import our testing components
from test_models import ComponentTestResult, TestCase, TestRunResults, TestRunSummary

# Load environment
load_dotenv()


class TestRunner:
    """
    Executes test cases and captures results for evaluation.

    This class leverages the existing demo_capture infrastructure to run
    test queries and collect comprehensive workflow data.
    """

    def __init__(
        self,
        output_dir: str = "test_runs",
        capture_data_dir: str = "captured_data",
        debug: bool = True,
    ):
        """
        Initialize the test runner.

        Args:
            output_dir: Directory to save test run results
            capture_data_dir: Directory to save captured workflow data
            debug: Enable debug logging
        """
        self.output_dir = Path(output_dir)
        self.capture_data_dir = Path(capture_data_dir)
        self.debug = debug

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized TestRunner")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Capture dir: {self.capture_data_dir}")

    async def run_test_suite(
        self,
        test_cases: List[TestCase],
        run_id: Optional[str] = None,
        max_test_cases: Optional[int] = None,
    ) -> TestRunResults:
        """
        Run a complete test suite with multiple test cases.

        Args:
            test_cases: List of test cases to execute
            run_id: Optional custom run ID (generated if not provided)
            max_test_cases: Limit the number of test cases to run

        Returns:
            TestRunResults with complete test run data
        """
        if run_id is None:
            run_id = self._generate_run_id()

        logger.info(f"Starting test suite run: {run_id}")
        logger.info(f"Test cases to run: {len(test_cases)}")

        # Limit test cases if requested
        if max_test_cases and max_test_cases < len(test_cases):
            test_cases = test_cases[:max_test_cases]
            logger.info(f"Limited to first {max_test_cases} test cases")

        start_time = datetime.now()

        # Initialize summary
        summary = TestRunSummary(
            run_id=run_id,
            start_time=start_time,
            test_cases_count=len(test_cases),
            test_mode="full_suite",
        )

        # Execute each test case
        all_results = []
        captured_files = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test case {i}/{len(test_cases)}: {test_case.test_id}")

            try:
                result = await self.run_single_test(test_case, run_id)
                all_results.extend(result.test_results)
                captured_files.extend(result.captured_data_files)

            except Exception as e:
                logger.error(f"Test case {test_case.test_id} failed: {e}")
                # Create error result
                error_result = ComponentTestResult(
                    component_name="full_workflow",
                    test_case_id=test_case.test_id,
                    execution_time=0.0,
                    execution_success=False,
                    execution_error=str(e),
                    actual_output=None,
                    expected_output=test_case,
                )
                all_results.append(error_result)

        # Finalize summary
        end_time = datetime.now()
        summary.end_time = end_time
        summary.total_duration = (end_time - start_time).total_seconds()
        summary.total_tests = len(all_results)
        summary.passed_tests = len(
            [r for r in all_results if r.evaluation and r.evaluation.passed],
        )
        summary.failed_tests = len(
            [r for r in all_results if r.evaluation and not r.evaluation.passed],
        )
        summary.error_tests = len([r for r in all_results if not r.execution_success])

        if summary.total_tests > 0:
            summary.average_test_duration = summary.total_duration / summary.total_tests

        # Create complete results
        results = TestRunResults(
            summary=summary,
            test_results=all_results,
            test_cases=test_cases,
            captured_data_files=captured_files,
        )

        # Save results
        await self._save_test_run(results)

        logger.info(f"Test suite completed: {run_id}")
        logger.info(f"  Total tests: {summary.total_tests}")
        logger.info(f"  Passed: {summary.passed_tests}")
        logger.info(f"  Failed: {summary.failed_tests}")
        logger.info(f"  Errors: {summary.error_tests}")
        logger.info(f"  Duration: {summary.total_duration:.1f}s")

        return results

    async def run_single_test(
        self,
        test_case: TestCase,
        run_id: Optional[str] = None,
    ) -> TestRunResults:
        """
        Run a single test case and capture results.

        Args:
            test_case: Test case to execute
            run_id: Optional run ID for organizing results

        Returns:
            TestRunResults for this single test
        """
        if run_id is None:
            run_id = self._generate_run_id()

        logger.info(f"Running single test: {test_case.test_id}")
        logger.info(f"Query: {test_case.query}")

        start_time = datetime.now()

        try:
            # Execute the query using demo_capture workflow
            capture_start = time.time()
            captured_file = await capture_full_workflow(
                query=test_case.query,
                output_file=str(
                    self.capture_data_dir / f"{run_id}_{test_case.test_id}.json",
                ),
            )
            execution_time = time.time() - capture_start

            logger.info(f"Workflow captured to: {captured_file}")

            # Load the captured data
            loader = WorkflowLoader(captured_file)

            # Extract results for comparison
            # For now, we'll just capture the raw workflow results
            # Evaluation will be done by separate evaluator components
            result = ComponentTestResult(
                component_name="full_workflow",
                test_case_id=test_case.test_id,
                execution_time=execution_time,
                execution_success=True,
                execution_error=None,
                actual_output=loader.data,  # Full captured workflow data
                expected_output=test_case,  # Test case with expected results
            )

            # Create summary
            end_time = datetime.now()
            summary = TestRunSummary(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                test_cases_count=1,
                total_tests=1,
                test_mode="single_test",
                total_duration=(end_time - start_time).total_seconds(),
            )

            results = TestRunResults(
                summary=summary,
                test_results=[result],
                test_cases=[test_case],
                captured_data_files=[captured_file],
            )

            logger.info(f"Single test completed successfully: {test_case.test_id}")
            return results

        except Exception as e:
            logger.error(f"Single test failed: {e}")

            # Create error result
            error_result = ComponentTestResult(
                component_name="full_workflow",
                test_case_id=test_case.test_id,
                execution_time=0.0,
                execution_success=False,
                execution_error=str(e),
                actual_output=None,
                expected_output=test_case,
            )

            # Create summary
            end_time = datetime.now()
            summary = TestRunSummary(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                test_cases_count=1,
                total_tests=1,
                error_tests=1,
                test_mode="single_test",
                total_duration=(end_time - start_time).total_seconds(),
            )

            results = TestRunResults(
                summary=summary,
                test_results=[error_result],
                test_cases=[test_case],
                captured_data_files=[],
            )

            return results

    async def run_fast_smoke_test(
        self,
        test_case: TestCase,
        run_id: Optional[str] = None,
    ) -> TestRunResults:
        """
        Run a fast smoke test using single-path execution.

        This uses gpt-5-nano for all components and takes [0] at each branch
        for maximum speed. Skips repository routing.

        Args:
            test_case: Test case to execute
            run_id: Optional run ID for organizing results

        Returns:
            TestRunResults for this fast smoke test
        """
        if run_id is None:
            run_id = self._generate_run_id("fast_smoke")

        logger.info(f"Running fast smoke test: {test_case.test_id}")
        logger.info(f"Query: {test_case.query}")
        logger.info("Mode: Single-path, gpt-5-nano, [0] selection")

        start_time = datetime.now()

        try:
            # Execute the query using fast smoke workflow
            capture_start = time.time()
            captured_file = await capture_fast_smoke_workflow(
                query=test_case.query,
                output_file=str(
                    self.capture_data_dir / f"{run_id}_{test_case.test_id}.json",
                ),
            )
            execution_time = time.time() - capture_start

            logger.info(f"Fast smoke workflow captured to: {captured_file}")

            # Load the captured data
            loader = WorkflowLoader(captured_file)

            # Create result
            result = ComponentTestResult(
                component_name="fast_smoke_workflow",
                test_case_id=test_case.test_id,
                execution_time=execution_time,
                execution_success=True,
                execution_error=None,
                actual_output=loader.data,
                expected_output=test_case,
            )

            # Create summary
            end_time = datetime.now()
            summary = TestRunSummary(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                test_cases_count=1,
                total_tests=1,
                test_mode="fast_smoke",
                total_duration=(end_time - start_time).total_seconds(),
            )

            results = TestRunResults(
                summary=summary,
                test_results=[result],
                test_cases=[test_case],
                captured_data_files=[captured_file],
            )

            logger.info(f"Fast smoke test completed: {test_case.test_id}")
            logger.info(f"Duration: {execution_time:.1f}s")
            return results

        except Exception as e:
            logger.error(f"Fast smoke test failed: {e}")

            # Create error result
            error_result = ComponentTestResult(
                component_name="fast_smoke_workflow",
                test_case_id=test_case.test_id,
                execution_time=0.0,
                execution_success=False,
                execution_error=str(e),
                actual_output=None,
                expected_output=test_case,
            )

            # Create summary
            end_time = datetime.now()
            summary = TestRunSummary(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                test_cases_count=1,
                total_tests=1,
                error_tests=1,
                test_mode="fast_smoke",
                total_duration=(end_time - start_time).total_seconds(),
            )

            results = TestRunResults(
                summary=summary,
                test_results=[error_result],
                test_cases=[test_case],
                captured_data_files=[],
            )

            return results

    async def run_component_test(
        self,
        component_name: str,
        captured_data_file: str,
        test_case: TestCase,
        topic_idx: int = 0,
        decomp_idx: int = 0,
    ) -> ComponentTestResult:
        """
        Run a test for a specific component using captured data.

        Args:
            component_name: Name of component to test
            captured_data_file: Path to captured workflow data
            test_case: Original test case for context
            topic_idx: Topic index to test
            decomp_idx: Decomposition index to test

        Returns:
            ComponentTestResult for the specific component
        """
        logger.info(f"Running component test: {component_name}")
        logger.info(f"Data file: {captured_data_file}")
        logger.info(f"Path: Topic {topic_idx}, Decomp {decomp_idx}")

        try:
            # Load captured data
            loader = WorkflowLoader(captured_data_file)

            # Execute the component
            start_time = time.time()
            actual_output = await loader.test_component(
                component_name,
                topic_idx,
                decomp_idx,
            )
            execution_time = time.time() - start_time

            # Get expected output (this would be defined by the specific component)
            expected_output = self._get_expected_component_output(
                component_name,
                test_case,
                topic_idx,
                decomp_idx,
            )

            result = ComponentTestResult(
                component_name=component_name,
                test_case_id=test_case.test_id,
                execution_time=execution_time,
                execution_success=True,
                execution_error=None,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            logger.info(f"Component test completed: {component_name}")
            return result

        except Exception as e:
            logger.error(f"Component test failed: {e}")

            return ComponentTestResult(
                component_name=component_name,
                test_case_id=test_case.test_id,
                execution_time=0.0,
                execution_success=False,
                execution_error=str(e),
                actual_output=None,
                expected_output=None,
            )

    def load_test_run(self, run_id: str) -> TestRunResults:
        """
        Load a previously saved test run.

        Args:
            run_id: ID of the test run to load

        Returns:
            TestRunResults object
        """
        results_file = self.output_dir / f"{run_id}_results.json"

        if not results_file.exists():
            raise FileNotFoundError(f"Test run results not found: {results_file}")

        with open(results_file, "r") as f:
            data = json.load(f)

        # Reconstruct the results object
        # This would need proper deserialization logic
        logger.info(f"Loaded test run: {run_id}")
        return TestRunResults.model_validate(data)

    def list_test_runs(self) -> List[str]:
        """
        List all available test runs.

        Returns:
            List of run IDs
        """
        run_ids = []
        for file in self.output_dir.glob("*_results.json"):
            run_id = file.stem.replace("_results", "")
            run_ids.append(run_id)

        return sorted(run_ids)

    async def _save_test_run(self, results: TestRunResults):
        """Save test run results to disk."""
        run_id = results.summary.run_id

        # Save JSON results
        results_file = self.output_dir / f"{run_id}_results.json"
        with open(results_file, "w") as f:
            # Convert to dict for JSON serialization
            json.dump(results.model_dump(), f, indent=2, default=str)

        # Save summary markdown
        summary_file = self.output_dir / f"{run_id}_summary.md"
        with open(summary_file, "w") as f:
            f.write(self._generate_summary_markdown(results))

        logger.info(f"Test run saved: {results_file}")

    def _generate_run_id(self, prefix: str = "test_run") -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _get_expected_component_output(
        self,
        component_name: str,
        test_case: TestCase,
        topic_idx: int,
        decomp_idx: int,
    ) -> Any:
        """
        Get expected output for a specific component.

        This would return the relevant expected results from the test case
        based on the component being tested.
        """
        if component_name == "topic_splitting":
            return test_case.expected_topics

        elif component_name in [
            "scientific_decomposition",
            "known_parameters",
            "searchable_parameters",
        ]:
            # For decomposition-level components, return expected decompositions
            return test_case.expected_decompositions

        else:
            # For other components, return the full test case
            return test_case

    def _generate_summary_markdown(self, results: TestRunResults) -> str:
        """Generate a markdown summary of test results."""
        summary = results.summary

        md = f"""# Test Run Summary: {summary.run_id}

## Overview
- **Start Time**: {summary.start_time}
- **End Time**: {summary.end_time}
- **Duration**: {summary.total_duration:.1f}s ({summary.duration_minutes:.1f} minutes)
- **Test Mode**: {summary.test_mode}

## Results
- **Total Tests**: {summary.total_tests}
- **Passed**: {summary.passed_tests}
- **Failed**: {summary.failed_tests}
- **Errors**: {summary.error_tests}
- **Success Rate**: {summary.success_rate:.1f}%

## Test Cases
"""

        for test_case in results.test_cases:
            md += f"\n### {test_case.test_id}\n"
            md += f"**Query**: {test_case.query}\n\n"
            md += f"**Category**: {test_case.category.value}\n\n"

            # Show test results for this case
            case_results = [
                r for r in results.test_results if r.test_case_id == test_case.test_id
            ]
            for result in case_results:
                status = "✅ SUCCESS" if result.execution_success else "❌ ERROR"
                md += f"- **{result.component_name}**: {status}\n"

                if result.evaluation:
                    eval_status = "✅ PASS" if result.evaluation.passed else "❌ FAIL"
                    md += f"  - Evaluation: {eval_status} ({result.evaluation.confidence:.2f})\n"

        return md
