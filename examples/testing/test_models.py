"""
Data models for the testing framework.

This module defines the data structures used throughout the testing system,
including test cases, expected results, and evaluation outcomes.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestCategory(str, Enum):
    """Categories for organizing test cases."""

    EARTH_SCIENCE = "earth"
    SPACE_SCIENCE = "space"
    ATMOSPHERIC = "atmospheric"
    OCEANOGRAPHIC = "oceanographic"
    CLIMATE = "climate"
    UNKNOWN = "unknown"


class ExpectedTopic(BaseModel):
    """Expected topic from test data."""

    title: str = Field(..., description="Expected topic title")
    category: TestCategory = Field(
        default=TestCategory.UNKNOWN,
        description="Topic category",
    )

    def __str__(self) -> str:
        return f"Topic: {self.title} ({self.category.value})"


class ExpectedDecomposition(BaseModel):
    """Expected scientific decomposition from test data."""

    title: str = Field(..., description="Expected decomposition title")
    is_minimum: bool = Field(
        default=False,
        description="Whether this is a minimum requirement",
    )
    justification: Optional[str] = Field(
        None,
        description="Justification for the decomposition",
    )
    useful_background: Optional[str] = Field(
        None,
        description="Useful background information",
    )

    def __str__(self) -> str:
        minimum_flag = " (MINIMUM)" if self.is_minimum else ""
        return f"Decomposition: {self.title}{minimum_flag}"


class TestCase(BaseModel):
    """A single test case with expected results."""

    # Core test information
    query: str = Field(..., description="Research query to test")
    test_id: str = Field(..., description="Unique identifier for this test case")

    # Expected results
    expected_topics: List[ExpectedTopic] = Field(
        default_factory=list,
        description="Expected topics",
    )
    expected_decompositions: List[ExpectedDecomposition] = Field(
        default_factory=list,
        description="Expected decompositions",
    )

    # Test metadata
    category: TestCategory = Field(
        default=TestCategory.UNKNOWN,
        description="Test category",
    )
    description: Optional[str] = Field(
        None,
        description="Description of what this test validates",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for test organization",
    )

    def __str__(self) -> str:
        return f"TestCase[{self.test_id}]: {self.query[:80]}..."

    @property
    def minimum_decompositions(self) -> List[ExpectedDecomposition]:
        """Get only the decompositions marked as minimum requirements."""
        return [d for d in self.expected_decompositions if d.is_minimum]


class EvaluationResult(BaseModel):
    """Result of evaluating actual vs expected results."""

    # Core evaluation
    passed: bool = Field(..., description="Whether the evaluation passed")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from LLM judge",
    )
    reasoning: str = Field(..., description="Detailed reasoning from LLM judge")

    # Specific feedback
    specific_feedback: Dict[str, Any] = Field(
        default_factory=dict,
        description="Component-specific feedback",
    )

    # Metadata
    component: str = Field(..., description="Component being evaluated")
    evaluator_model: str = Field(..., description="LLM model used for evaluation")
    evaluation_time: datetime = Field(
        default_factory=datetime.now,
        description="When evaluation was performed",
    )

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status} ({self.confidence:.2f}) - {self.component}"


class ComponentTestResult(BaseModel):
    """Results from testing a specific component."""

    component_name: str = Field(..., description="Name of the component tested")
    test_case_id: str = Field(..., description="ID of the test case")

    # Test execution
    execution_time: float = Field(..., description="Execution time in seconds")
    execution_success: bool = Field(
        ...,
        description="Whether execution completed without errors",
    )
    execution_error: Optional[str] = Field(
        None,
        description="Error message if execution failed",
    )

    # Results
    actual_output: Any = Field(..., description="Actual output from component")
    expected_output: Any = Field(..., description="Expected output for comparison")

    # Evaluation
    evaluation: Optional[EvaluationResult] = Field(
        None,
        description="Evaluation result",
    )

    def __str__(self) -> str:
        status = "SUCCESS" if self.execution_success else "ERROR"
        eval_status = f" -> {self.evaluation}" if self.evaluation else ""
        return f"{self.component_name}: {status}{eval_status}"


class TestRunSummary(BaseModel):
    """Summary of a complete test run."""

    # Run identification
    run_id: str = Field(..., description="Unique identifier for this test run")
    start_time: datetime = Field(..., description="When the test run started")
    end_time: Optional[datetime] = Field(None, description="When the test run ended")

    # Test configuration
    test_cases_count: int = Field(..., description="Number of test cases executed")
    components_tested: List[str] = Field(
        default_factory=list,
        description="Components that were tested",
    )
    test_mode: str = Field(
        ...,
        description="Type of testing performed (full, component, regression)",
    )

    # Results summary
    total_tests: int = Field(0, description="Total number of individual tests")
    passed_tests: int = Field(0, description="Number of tests that passed")
    failed_tests: int = Field(0, description="Number of tests that failed")
    error_tests: int = Field(0, description="Number of tests that had execution errors")

    # Performance
    total_duration: float = Field(0.0, description="Total duration in seconds")
    average_test_duration: float = Field(
        0.0,
        description="Average test duration in seconds",
    )

    # Configuration
    llm_model_config: Dict[str, str] = Field(
        default_factory=dict,
        description="LLM model configuration used",
    )
    framework_version: Optional[str] = Field(None, description="Framework version")

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0

    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.total_duration / 60.0

    def __str__(self) -> str:
        return f"TestRun[{self.run_id}]: {self.passed_tests}/{self.total_tests} passed ({self.success_rate:.1f}%)"


class TestRunResults(BaseModel):
    """Complete results from a test run."""

    # Summary
    summary: TestRunSummary = Field(..., description="Test run summary")

    # Detailed results
    test_results: List[ComponentTestResult] = Field(
        default_factory=list,
        description="Individual test results",
    )

    # Test cases used
    test_cases: List[TestCase] = Field(
        default_factory=list,
        description="Test cases that were executed",
    )

    # Raw data files
    captured_data_files: List[str] = Field(
        default_factory=list,
        description="Paths to captured workflow data",
    )

    def get_results_by_component(
        self,
        component_name: str,
    ) -> List[ComponentTestResult]:
        """Get all results for a specific component."""
        return [r for r in self.test_results if r.component_name == component_name]

    def get_failed_tests(self) -> List[ComponentTestResult]:
        """Get all tests that failed evaluation."""
        return [
            r for r in self.test_results if r.evaluation and not r.evaluation.passed
        ]

    def get_error_tests(self) -> List[ComponentTestResult]:
        """Get all tests that had execution errors."""
        return [r for r in self.test_results if not r.execution_success]


class RegressionResult(BaseModel):
    """Result of comparing two test runs for regression analysis."""

    # Runs being compared
    baseline_run_id: str = Field(..., description="ID of the baseline run")
    current_run_id: str = Field(..., description="ID of the current run")

    # Comparison results
    improvement_count: int = Field(0, description="Number of tests that improved")
    regression_count: int = Field(0, description="Number of tests that regressed")
    unchanged_count: int = Field(0, description="Number of tests with no change")

    # Detailed changes
    improvements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed improvement information",
    )
    regressions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed regression information",
    )

    # Overall assessment
    overall_assessment: str = Field(..., description="Overall regression analysis")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis",
    )

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return self.regression_count > 0

    @property
    def net_improvement(self) -> int:
        """Calculate net improvement (improvements - regressions)."""
        return self.improvement_count - self.regression_count

    def __str__(self) -> str:
        return f"Regression: {self.improvement_count} improved, {self.regression_count} regressed"
