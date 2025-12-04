"""
Test Data Manager for loading and managing test cases from CSV files.

This module handles parsing the DataAgentQueries CSV file and converting it
into structured test cases that can be used by the testing framework.
"""

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from test_models import ExpectedDecomposition, ExpectedTopic, TestCase, TestCategory


class TestDataManager:
    """
    Manages test data loading and parsing from CSV files.

    The CSV format expected:
    - query: Research question
    - topic: Expected topic name
    - division: Category (earth, space, etc.)
    - decomps: Expected decomposition name
    - minimum: Whether this decomposition is required (yes/no)
    - justification: Justification for the decomposition
    - useful background: Additional background information
    """

    def __init__(self, csv_path: str):
        """
        Initialize with path to the test data CSV file.

        Args:
            csv_path: Path to the CSV file containing test data
        """
        self.csv_path = Path(csv_path)
        self._test_cases: Optional[List[TestCase]] = None
        self._queries_to_cases: Optional[Dict[str, TestCase]] = None

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Test data CSV not found: {csv_path}")

        logger.info(f"Initialized TestDataManager with {csv_path}")

    def load_test_cases(self) -> List[TestCase]:
        """
        Load and parse all test cases from the CSV file.

        Returns:
            List of TestCase objects
        """
        if self._test_cases is not None:
            return self._test_cases

        logger.info(f"Loading test cases from {self.csv_path}")

        # Read and group CSV data by query
        query_groups = self._parse_csv_by_query()

        # Convert grouped data to TestCase objects
        test_cases = []
        for query, rows in query_groups.items():
            test_case = self._create_test_case_from_rows(query, rows)
            test_cases.append(test_case)

        self._test_cases = test_cases
        self._queries_to_cases = {tc.query: tc for tc in test_cases}

        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def get_test_case_by_query(self, query: str) -> Optional[TestCase]:
        """
        Get a test case by its query string.

        Args:
            query: The research query to look for

        Returns:
            TestCase if found, None otherwise
        """
        if self._queries_to_cases is None:
            self.load_test_cases()

        return self._queries_to_cases.get(query)

    def get_test_cases_by_category(self, category: TestCategory) -> List[TestCase]:
        """
        Get all test cases for a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of TestCase objects in that category
        """
        if self._test_cases is None:
            self.load_test_cases()

        return [tc for tc in self._test_cases if tc.category == category]

    def get_test_cases_with_minimum_decomps(self) -> List[TestCase]:
        """
        Get all test cases that have minimum requirement decompositions.

        Returns:
            List of TestCase objects with minimum decompositions
        """
        if self._test_cases is None:
            self.load_test_cases()

        return [tc for tc in self._test_cases if tc.minimum_decompositions]

    def validate_test_cases(self) -> Tuple[List[str], List[str]]:
        """
        Validate all loaded test cases and return any issues found.

        Returns:
            Tuple of (warnings, errors) lists
        """
        if self._test_cases is None:
            self.load_test_cases()

        warnings = []
        errors = []

        for test_case in self._test_cases:
            # Check for empty queries
            if not test_case.query.strip():
                errors.append(f"Test case {test_case.test_id} has empty query")

            # Check for missing expected results
            if not test_case.expected_topics:
                warnings.append(f"Test case {test_case.test_id} has no expected topics")

            if not test_case.expected_decompositions:
                warnings.append(
                    f"Test case {test_case.test_id} has no expected decompositions",
                )

            # Check for very short queries
            if len(test_case.query.strip()) < 10:
                warnings.append(f"Test case {test_case.test_id} has very short query")

            # Check for missing minimum decompositions
            if not test_case.minimum_decompositions:
                warnings.append(
                    f"Test case {test_case.test_id} has no minimum decompositions marked",
                )

        return warnings, errors

    def _parse_csv_by_query(self) -> Dict[str, List[Dict]]:
        """
        Parse CSV file and group rows by query.

        Returns:
            Dictionary mapping query -> list of row data
        """
        query_groups = {}

        with open(self.csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                # Clean up the row data
                row = {k.strip(): v.strip() for k, v in row.items() if k}

                # Validate required fields
                if "query" not in row:
                    logger.warning(f"Row {row_num}: Missing 'query' field, skipping")
                    continue

                query = row["query"]
                if not query:
                    logger.warning(f"Row {row_num}: Empty query, skipping")
                    continue

                # Group by query
                if query not in query_groups:
                    query_groups[query] = []

                query_groups[query].append(
                    {
                        "row_num": row_num,
                        **row,
                    },
                )

        logger.info(f"Parsed {len(query_groups)} unique queries from CSV")
        return query_groups

    def _create_test_case_from_rows(self, query: str, rows: List[Dict]) -> TestCase:
        """
        Create a TestCase object from grouped CSV rows for a single query.

        Args:
            query: The research query
            rows: List of CSV row data for this query

        Returns:
            TestCase object
        """
        # Generate unique test ID from query
        test_id = self._generate_test_id(query)

        # Parse topics and decompositions from rows
        topics = self._extract_topics_from_rows(rows)
        decompositions = self._extract_decompositions_from_rows(rows)

        # Determine category (use most common division, default to earth)
        divisions = [self._normalize_category(row.get("division", "")) for row in rows]
        category = (
            max(set(divisions), key=divisions.count)
            if divisions
            else TestCategory.EARTH_SCIENCE
        )

        # Create tags from unique topics
        tags = list(set(topic.title.lower().replace(" ", "_") for topic in topics))

        return TestCase(
            query=query,
            test_id=test_id,
            expected_topics=topics,
            expected_decompositions=decompositions,
            category=category,
            description=f"Test case with {len(topics)} topics and {len(decompositions)} decompositions",
            tags=tags,
        )

    def _extract_topics_from_rows(self, rows: List[Dict]) -> List[ExpectedTopic]:
        """Extract unique topics from CSV rows."""
        seen_topics = set()
        topics = []

        for row in rows:
            topic_name = row.get("topic", "").strip()
            if topic_name and topic_name not in seen_topics:
                category = self._normalize_category(row.get("division", ""))
                topics.append(
                    ExpectedTopic(
                        title=topic_name,
                        category=category,
                    ),
                )
                seen_topics.add(topic_name)

        return topics

    def _extract_decompositions_from_rows(
        self,
        rows: List[Dict],
    ) -> List[ExpectedDecomposition]:
        """Extract decompositions from CSV rows."""
        seen_decomps = set()
        decompositions = []

        for row in rows:
            decomp_name = row.get("decomps", "").strip()
            if decomp_name and decomp_name not in seen_decomps:
                is_minimum = self._normalize_boolean(row.get("minimum", ""))
                justification = row.get("justification", "").strip() or None
                background = row.get("useful background", "").strip() or None

                decompositions.append(
                    ExpectedDecomposition(
                        title=decomp_name,
                        is_minimum=is_minimum,
                        justification=justification,
                        useful_background=background,
                    ),
                )
                seen_decomps.add(decomp_name)

        return decompositions

    def _generate_test_id(self, query: str) -> str:
        """Generate a unique test ID from the query."""
        # Create a slug from the query
        slug = re.sub(r"[^\w\s-]", "", query.lower())
        slug = re.sub(r"[-\s]+", "_", slug)
        slug = slug[:50]  # Limit length

        # Add hash for uniqueness
        hash_suffix = hashlib.md5(query.encode()).hexdigest()[:8]

        return f"{slug}_{hash_suffix}"

    def _normalize_category(self, division: str) -> TestCategory:
        """Normalize division string to TestCategory."""
        division = division.lower().strip()

        category_mapping = {
            "earth": TestCategory.EARTH_SCIENCE,
            "space": TestCategory.SPACE_SCIENCE,
            "atmospheric": TestCategory.ATMOSPHERIC,
            "atmosphere": TestCategory.ATMOSPHERIC,
            "ocean": TestCategory.OCEANOGRAPHIC,
            "oceanographic": TestCategory.OCEANOGRAPHIC,
            "climate": TestCategory.CLIMATE,
        }

        return category_mapping.get(division, TestCategory.EARTH_SCIENCE)

    def _normalize_boolean(self, value: str) -> bool:
        """Normalize string value to boolean."""
        value = value.lower().strip()
        return value in ("yes", "true", "1", "y")

    def get_summary_stats(self) -> Dict[str, int]:
        """
        Get summary statistics about the loaded test cases.

        Returns:
            Dictionary with statistics
        """
        if self._test_cases is None:
            self.load_test_cases()

        stats = {
            "total_test_cases": len(self._test_cases),
            "total_topics": sum(len(tc.expected_topics) for tc in self._test_cases),
            "total_decompositions": sum(
                len(tc.expected_decompositions) for tc in self._test_cases
            ),
            "minimum_decompositions": sum(
                len(tc.minimum_decompositions) for tc in self._test_cases
            ),
        }

        # Category breakdown
        for category in TestCategory:
            count = len([tc for tc in self._test_cases if tc.category == category])
            stats[f"category_{category.value}"] = count

        return stats

    def print_summary(self):
        """Print a summary of the loaded test data."""
        stats = self.get_summary_stats()

        print("\n📊 TEST DATA SUMMARY")
        print(f"📁 Source: {self.csv_path}")
        print(f"🧪 Test Cases: {stats['total_test_cases']}")
        print(f"📝 Topics: {stats['total_topics']}")
        print(f"🔬 Decompositions: {stats['total_decompositions']}")
        print(f"⭐ Minimum Required: {stats['minimum_decompositions']}")

        print("\n📊 Category Breakdown:")
        for category in TestCategory:
            count = stats[f"category_{category.value}"]
            if count > 0:
                print(f"   {category.value:15}: {count}")

        # Show some example test cases
        if self._test_cases:
            print("\n📋 Example Test Cases:")
            for i, tc in enumerate(self._test_cases[:3], 1):
                print(f"   {i}. {tc.query[:80]}...")
                print(
                    f"      Topics: {len(tc.expected_topics)}, Decomps: {len(tc.expected_decompositions)}",
                )
                if tc.minimum_decompositions:
                    print(
                        f"      Minimum required: {[d.title for d in tc.minimum_decompositions]}",
                    )
