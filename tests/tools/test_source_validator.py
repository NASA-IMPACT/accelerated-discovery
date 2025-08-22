#!/usr/bin/env python3
"""
Comprehensive tests for the ISSN-based source validation tool.

This test suite covers:
1. ISSN normalization and validation
2. Whitelist loading with multiple formats
3. CrossRef API integration (mocked)
4. DOI extraction from various URL formats
5. End-to-end validation workflow
6. Error handling and edge cases
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch

from akd.structures import SearchResultItem
from akd.tools.source_validator import (
    SourceValidator,
    SourceValidatorConfig,
    SourceValidatorInputSchema,
    SourceInfo,
    ValidationResult,
    create_source_validator,
)


class TestISSNNormalization:
    """Test ISSN normalization and validation logic."""

    def test_normalize_issn_valid_formats(self):
        """Test normalization of valid ISSN formats."""
        assert SourceValidator._normalize_issn("1234-5678") == "1234-5678"
        assert SourceValidator._normalize_issn("12345678") == "1234-5678"
        assert SourceValidator._normalize_issn("1234 5678") == "1234-5678"
        assert SourceValidator._normalize_issn("1234.5678") == "1234-5678"
        assert SourceValidator._normalize_issn("1234567x") == "1234-567X"
        assert SourceValidator._normalize_issn("1234567X") == "1234-567X"

    def test_normalize_issn_invalid_formats(self):
        """Test normalization rejects invalid ISSN formats."""
        assert SourceValidator._normalize_issn("123-456") is None
        assert SourceValidator._normalize_issn("12345678901") is None
        assert SourceValidator._normalize_issn("abcd-efgh") is None
        assert SourceValidator._normalize_issn("") is None
        assert SourceValidator._normalize_issn(None) is None


class TestWhitelistLoading:
    """Test loading of ISSN whitelists in various formats."""

    def test_load_flat_list_whitelist(self):
        """Test loading flat list of ISSNs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["1234-5678", "8765-4321", "1111222X"], f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert "1234-5678" in validator._allowed_issn_set
        assert "8765-4321" in validator._allowed_issn_set
        assert "1111-222X" in validator._allowed_issn_set
        assert len(validator._allowed_issn_set) == 3

        Path(whitelist_path).unlink()

    def test_load_object_with_issn_key(self):
        """Test loading object with 'issn' key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"issn": ["1234-5678", "8765-4321"]}, f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert "1234-5678" in validator._allowed_issn_set
        assert "8765-4321" in validator._allowed_issn_set
        assert len(validator._allowed_issn_set) == 2

        Path(whitelist_path).unlink()

    def test_load_categorized_whitelist(self):
        """Test loading categorized whitelist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "Bio": ["1234-5678", "1111-222X"],
                "CS": ["8765-4321"],
                "Physics": ["9999-8888"]
            }, f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert "1234-5678" in validator._allowed_issn_set
        assert "1111-222X" in validator._allowed_issn_set
        assert "8765-4321" in validator._allowed_issn_set
        assert "9999-8888" in validator._allowed_issn_set

        assert validator._issn_to_category["1234-5678"] == "Bio"
        assert validator._issn_to_category["8765-4321"] == "CS"
        assert validator._issn_to_category["9999-8888"] == "Physics"

        Path(whitelist_path).unlink()

    def test_load_nested_categories_format(self):
        """Test loading nested categories format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "categories": {
                    "Bio": ["1234-5678"],
                    "CS": ["8765-4321"]
                }
            }, f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert "1234-5678" in validator._allowed_issn_set
        assert "8765-4321" in validator._allowed_issn_set
        assert validator._issn_to_category["1234-5678"] == "Bio"
        assert validator._issn_to_category["8765-4321"] == "CS"

        Path(whitelist_path).unlink()

    def test_load_whitelist_file_not_found(self):
        """Test handling of missing whitelist file."""
        config = SourceValidatorConfig(whitelist_file_path="/nonexistent/path.json")
        validator = SourceValidator(config)

        assert len(validator._allowed_issn_set) == 0
        assert len(validator._issn_to_category) == 0


class TestDOIExtraction:
    """Test DOI extraction from various URL formats."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = SourceValidator()

    def test_extract_doi_from_standard_urls(self):
        """Test extraction from standard DOI URLs."""
        urls = [
            "https://doi.org/10.1016/j.actaastro.2023.05.014",
            "http://dx.doi.org/10.1029/2023GL104567",
            "https://www.dx.doi.org/10.1038/nature12345",
        ]
        expected = [
            "10.1016/j.actaastro.2023.05.014",
            "10.1029/2023GL104567",
            "10.1038/nature12345",
        ]

        for url, expected_doi in zip(urls, expected):
            assert self.validator._extract_doi_from_url(url) == expected_doi

    def test_extract_doi_from_publisher_urls(self):
        """Test extraction from publisher-specific URLs."""
        urls = [
            "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL104567",
            "https://www.nature.com/articles/doi:10.1038/nature12345",
            "https://example.com/paper?doi=10.1016/j.test.2023.01.001",
        ]
        expected = [
            "10.1029/2023GL104567",
            "10.1038/nature12345", 
            "10.1016/j.test.2023.01.001",
        ]

        for url, expected_doi in zip(urls, expected):
            assert self.validator._extract_doi_from_url(url) == expected_doi

    def test_extract_doi_no_match(self):
        """Test URLs without DOIs."""
        urls = [
            "https://example.com/paper-without-doi",
            "https://university.edu/research/paper.html",
            "https://arxiv.org/abs/1234.5678",
        ]

        for url in urls:
            assert self.validator._extract_doi_from_url(url) is None


class TestCrossRefIntegration:
    """Test CrossRef API integration (mocked)."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = SourceValidator()

    @pytest.mark.asyncio
    async def test_parse_crossref_response_complete(self):
        """Test parsing complete CrossRef response."""
        crossref_data = {
            "container-title": ["Nature"],
            "publisher": "Nature Publishing Group",
            "ISSN": ["0028-0836", "1476-4687"],
            "issn-type": [
                {"value": "1476-4687", "type": "electronic"},
                {"value": "0028-0836", "type": "print"}
            ],
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}]
        }

        result = self.validator._parse_crossref_response(
            crossref_data, 
            "10.1038/nature12345", 
            "https://nature.com/articles/nature12345"
        )

        assert result.title == "Nature"
        assert result.publisher == "Nature Publishing Group"
        assert result.doi == "10.1038/nature12345"
        assert result.url == "https://nature.com/articles/nature12345"
        assert "0028-0836" in result.issn
        assert "1476-4687" in result.issn
        assert result.is_open_access is True

    @pytest.mark.asyncio
    async def test_parse_crossref_response_minimal(self):
        """Test parsing minimal CrossRef response."""
        crossref_data = {
            "container-title": ["Test Journal"],
            "ISSN": ["1234-5678"]
        }

        result = self.validator._parse_crossref_response(
            crossref_data,
            "10.1000/test.2023.001",
            "https://test.com/paper"
        )

        assert result.title == "Test Journal"
        assert result.publisher is None
        assert result.issn == ["1234-5678"]
        assert result.is_open_access is None

    def test_crossref_response_parsing_covers_all_fields(self):
        """Test that CrossRef response parsing covers all expected fields."""
        # This test focuses on the parsing logic which is the core functionality
        crossref_data = {
            "container-title": ["Advanced Materials"],
            "publisher": "Wiley",
            "ISSN": ["0935-9648", "1521-4095"],
            "issn-type": [
                {"value": "0935-9648", "type": "print"},
                {"value": "1521-4095", "type": "electronic"}
            ],
            "license": [
                {"URL": "https://creativecommons.org/licenses/by/4.0/"},
                {"URL": "https://example.com/other-license"}
            ]
        }

        result = self.validator._parse_crossref_response(
            crossref_data,
            "10.1002/adma.202301234",
            "https://onlinelibrary.wiley.com/doi/10.1002/adma.202301234"
        )

        assert result.title == "Advanced Materials"
        assert result.publisher == "Wiley"
        assert result.doi == "10.1002/adma.202301234"
        assert result.url == "https://onlinelibrary.wiley.com/doi/10.1002/adma.202301234"
        assert "0935-9648" in result.issn
        assert "1521-4095" in result.issn
        assert len(result.issn) == 2  # No duplicates
        assert result.is_open_access is True  # Found CC license


class TestValidationLogic:
    """Test ISSN-based validation logic."""

    def setup_method(self):
        """Set up test validator with sample whitelist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "Bio": ["0028-0836", "1476-4687"],  # Nature ISSNs
                "CS": ["1234-5678"],
                "Physics": ["8765-4321"]
            }, f)
            self.whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=self.whitelist_path)
        self.validator = SourceValidator(config)

    def teardown_method(self):
        """Clean up temporary files."""
        Path(self.whitelist_path).unlink()

    def test_validate_whitelisted_source(self):
        """Test validation of whitelisted source."""
        source_info = SourceInfo(
            title="Nature",
            publisher="Nature Publishing Group",
            issn=["0028-0836", "1476-4687"],
            doi="10.1038/nature12345",
            url="https://nature.com/articles/nature12345"
        )

        is_whitelisted, category, confidence, matched_issn = self.validator._validate_against_whitelist(source_info)

        assert is_whitelisted is True
        assert category == "Bio"
        assert confidence == 1.0
        assert matched_issn in ["0028-0836", "1476-4687"]

    def test_validate_non_whitelisted_source(self):
        """Test validation of non-whitelisted source."""
        source_info = SourceInfo(
            title="Unknown Journal",
            issn=["9999-8888"],
            doi="10.1000/unknown.2023.001",
            url="https://unknown.com/paper"
        )

        is_whitelisted, category, confidence, matched_issn = self.validator._validate_against_whitelist(source_info)

        assert is_whitelisted is False
        assert category is None
        assert confidence == 0.0
        assert matched_issn is None

    def test_validate_arxiv_with_bypass(self):
        """Test arXiv validation with bypass enabled."""
        config = SourceValidatorConfig(
            whitelist_file_path=self.whitelist_path,
            allow_arxiv=True
        )
        validator = SourceValidator(config)

        source_info = SourceInfo(
            title="arXiv",
            issn=[],
            doi="10.48550/arXiv.2301.12345",
            url="https://arxiv.org/abs/2301.12345"
        )

        is_whitelisted, category, confidence, matched_issn = validator._validate_against_whitelist(source_info)

        assert is_whitelisted is True
        assert category == "arXiv"
        assert confidence == 1.0
        assert matched_issn is None

    def test_validate_empty_whitelist(self):
        """Test validation with empty whitelist (strict mode)."""
        config = SourceValidatorConfig(whitelist_file_path="/nonexistent/path.json")
        validator = SourceValidator(config)

        source_info = SourceInfo(
            title="Any Journal",
            issn=["1234-5678"],
            doi="10.1000/test.2023.001",
            url="https://test.com/paper"
        )

        is_whitelisted, category, confidence, matched_issn = validator._validate_against_whitelist(source_info)

        assert is_whitelisted is False
        assert category is None
        assert confidence == 0.0
        assert matched_issn is None


class TestEndToEndValidation:
    """Test complete validation workflow."""

    def setup_method(self):
        """Set up test validator and sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "Bio": ["0028-0836", "1476-4687"],
                "Physics": ["1234-5678"]
            }, f)
            self.whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=self.whitelist_path)
        self.validator = SourceValidator(config)

    def teardown_method(self):
        """Clean up temporary files."""
        Path(self.whitelist_path).unlink()

    @pytest.mark.asyncio
    async def test_validate_single_result_success(self):
        """Test successful single result validation."""
        search_result = SearchResultItem(
            url="https://doi.org/10.1038/nature12345",
            title="Test paper in Nature",
            content="Research content...",
            query="test query",
            category="science",
            doi="10.1038/nature12345",
            engine="test"
        )

        # Mock CrossRef response
        mock_crossref_data = {
            "container-title": ["Nature"],
            "publisher": "Nature Publishing Group",
            "ISSN": ["0028-0836", "1476-4687"]
        }

        with patch.object(self.validator, '_fetch_crossref_metadata', return_value=mock_crossref_data):
            session = Mock()
            result = await self.validator._validate_single_result(session, search_result)

            assert result.is_whitelisted is True
            assert result.whitelist_category == "Bio"
            assert result.confidence_score == 1.0
            assert result.matched_issn in ["0028-0836", "1476-4687"]
            assert result.source_info is not None
            assert result.source_info.title == "Nature"

    @pytest.mark.asyncio
    async def test_validate_single_result_no_doi(self):
        """Test validation when no DOI can be extracted."""
        search_result = SearchResultItem(
            url="https://example.com/paper-without-doi",
            title="Paper without DOI",
            content="Content...",
            query="test query",
            category="science",
            engine="test"
        )

        session = Mock()
        result = await self.validator._validate_single_result(session, search_result)

        assert result.is_whitelisted is False
        assert result.source_info is None
        assert "No DOI found" in result.validation_errors[0]
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_validate_single_result_crossref_failure(self):
        """Test validation when CrossRef API fails."""
        search_result = SearchResultItem(
            url="https://doi.org/10.1000/nonexistent",
            title="Paper with invalid DOI",
            content="Content...",
            query="test query",
            category="science",
            doi="10.1000/nonexistent",
            engine="test"
        )

        with patch.object(self.validator, '_fetch_crossref_metadata', return_value=None):
            session = Mock()
            result = await self.validator._validate_single_result(session, search_result)

            assert result.is_whitelisted is False
            assert result.source_info is None
            assert "Failed to fetch metadata from CrossRef" in result.validation_errors[0]
            assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_full_validation_workflow(self):
        """Test complete validation workflow with multiple results."""
        search_results = [
            SearchResultItem(
                url="https://doi.org/10.1038/nature12345",
                title="Nature paper",
                content="Content...",
                query="test query",
                category="science",
                doi="10.1038/nature12345",
                engine="test"
            ),
            SearchResultItem(
                url="https://doi.org/10.1000/unknown.2023.001",
                title="Unknown journal paper",
                content="Content...",
                query="test query",
                category="science",
                doi="10.1000/unknown.2023.001",
                engine="test"
            ),
        ]

        input_params = SourceValidatorInputSchema(
            search_results=search_results,
            whitelist_file_path=self.whitelist_path
        )

        # Mock CrossRef responses
        def mock_crossref_response(session, doi):
            if "nature" in doi:
                return {
                    "container-title": ["Nature"],
                    "ISSN": ["0028-0836", "1476-4687"]
                }
            else:
                return {
                    "container-title": ["Unknown Journal"],
                    "ISSN": ["9999-8888"]
                }

        with patch.object(self.validator, '_fetch_crossref_metadata', side_effect=mock_crossref_response):
            result = await self.validator._arun(input_params)

            assert len(result.validated_results) == 2
            assert result.summary["total_processed"] == 2
            assert result.summary["whitelisted_count"] == 1
            assert result.summary["whitelisted_percentage"] == 50.0

            # First result should be whitelisted (Nature)
            nature_result = result.validated_results[0]
            assert nature_result.is_whitelisted is True
            assert nature_result.whitelist_category == "Bio"

            # Second result should not be whitelisted
            unknown_result = result.validated_results[1]
            assert unknown_result.is_whitelisted is False
            assert unknown_result.whitelist_category is None


class TestFactoryFunction:
    """Test the factory function for creating validators."""

    def test_create_source_validator_default(self):
        """Test factory function with default parameters."""
        validator = create_source_validator()
        
        assert isinstance(validator, SourceValidator)
        assert validator.config.timeout_seconds == 30
        assert validator.config.max_concurrent_requests == 10
        assert validator.config.debug is False

    def test_create_source_validator_custom(self):
        """Test factory function with custom parameters."""
        validator = create_source_validator(
            timeout_seconds=60,
            max_concurrent_requests=5,
            debug=True
        )
        
        assert validator.config.timeout_seconds == 60
        assert validator.config.max_concurrent_requests == 5
        assert validator.config.debug is True


class TestConfigurationEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_whitelist_format(self):
        """Test handling of invalid whitelist format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump("invalid format", f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert len(validator._allowed_issn_set) == 0
        Path(whitelist_path).unlink()

    def test_malformed_json_whitelist(self):
        """Test handling of malformed JSON whitelist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        assert len(validator._allowed_issn_set) == 0
        Path(whitelist_path).unlink()

    def test_whitelist_with_invalid_issns(self):
        """Test whitelist loading with mix of valid and invalid ISSNs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                "1234-5678",  # Valid
                "invalid-issn",  # Invalid
                "8765-4321",  # Valid
                "123",  # Invalid
                None,  # Invalid
            ], f)
            whitelist_path = f.name

        config = SourceValidatorConfig(whitelist_file_path=whitelist_path)
        validator = SourceValidator(config)

        # Should only contain valid ISSNs
        assert len(validator._allowed_issn_set) == 2
        assert "1234-5678" in validator._allowed_issn_set
        assert "8765-4321" in validator._allowed_issn_set

        Path(whitelist_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])