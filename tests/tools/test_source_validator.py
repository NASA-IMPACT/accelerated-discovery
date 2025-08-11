from unittest.mock import AsyncMock
import pytest
import tempfile
import json
import os

from akd.structures import SearchResultItem
from akd.tools.source_validator import SourceValidator, SourceValidatorConfig, SourceInfo, SourceValidatorInputSchema  # <-- change this
from rapidfuzz.fuzz import token_set_ratio

# Shared fake whitelist data
FAKE_WHITELIST_DICT = {
    "data": {
        "AI": {
            "journals": [
                {"Journal Name": "Journal of Artificial Intelligence (JAI)"},
                {"Journal Name": "AI and Society"},
            ]
        }
    }
}


FAKE_CROSSREF_RESPONSE = {
    "message": {
        "container-title": ["Journal of Artificial Intelligence"],
        "publisher": "AI Publisher",
        "ISSN": ["1234-5678"],
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}]
    }
}


@pytest.fixture
def validator_with_temp_whitelist_fuzzy_enabled():
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
        json.dump(FAKE_WHITELIST_DICT, tmp_file)
        tmp_file_path = tmp_file.name

    # Instantiate validator with config
    config = SourceValidatorConfig(
        whitelist_file_path=tmp_file_path,
        use_fuzzy_match=True,
        fuzzy_match_method="token_set",
        fuzzy_threshold=85,
        debug=False,
    )
    validator = SourceValidator(config=config)

    yield validator  # Give the test access to the validator

    os.remove(tmp_file_path)


@pytest.fixture
def validator_with_temp_whitelist_fuzzy_disabled():
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
        json.dump(FAKE_WHITELIST_DICT, tmp_file)
        tmp_file_path = tmp_file.name

    config = SourceValidatorConfig(
        whitelist_file_path=tmp_file_path,
        use_fuzzy_match=False,
        debug=False,
    )
    validator = SourceValidator(config=config)

    yield validator

    os.remove(tmp_file_path)



def test_exact_match_from_file(validator_with_temp_whitelist_fuzzy_enabled):
    source = SourceInfo(
        title="Journal of Artificial Intelligence (JAI)",
        publisher=None,
        issn=None,
        is_open_access=None,
        doi="10.1000/test"
    )
    result = validator_with_temp_whitelist_fuzzy_enabled._validate_against_whitelist(source)
    assert result == (True, "AI", 1.0)

def test_fuzzy_match_from_file(validator_with_temp_whitelist_fuzzy_enabled):
    source = SourceInfo(
        title="Journal of Artificial Intelligences",  # minor variation
        publisher=None,
        issn=None,
        is_open_access=None,
        doi="10.1000/test"
    )
    result = validator_with_temp_whitelist_fuzzy_enabled._validate_against_whitelist(source)
    assert result[0] is True
    assert result[1] == "AI"
    assert result[2] >= 0.85

def test_no_match_from_file(validator_with_temp_whitelist_fuzzy_enabled):
    source = SourceInfo(
        title="Some Unknown Journal",
        publisher=None,
        issn=None,
        is_open_access=None,
        doi="10.1000/test"
    )
    result = validator_with_temp_whitelist_fuzzy_enabled._validate_against_whitelist(source)
    assert result == (False, None, 0.0)

def test_match_pass_fuzzy_disabled(validator_with_temp_whitelist_fuzzy_disabled):
    source = SourceInfo(
        title="Journal of Artificial Intelligence",
        publisher=None,
        issn=None,
        is_open_access=None,
        doi="10.1000/test"
    )
    result = validator_with_temp_whitelist_fuzzy_disabled._validate_against_whitelist(source)
    assert result[0] is True
    assert result[1] == "AI"

def test_fuzzy_candidate_fails_when_fuzzy_disabled(validator_with_temp_whitelist_fuzzy_disabled):
    source = SourceInfo(
        title="Journal of Artificial Intelligences",  # minor plural change
        publisher=None,
        issn=None,
        is_open_access=None,
        doi="10.1000/test"
    )
    result = validator_with_temp_whitelist_fuzzy_disabled._validate_against_whitelist(source)
    assert result == (False, None, 0.0)

@pytest.mark.asyncio
async def test_arun_exact_match(monkeypatch, validator_with_temp_whitelist_fuzzy_enabled):
    monkeypatch.setattr(
        validator_with_temp_whitelist_fuzzy_enabled,
        "_fetch_crossref_metadata",
        AsyncMock(return_value=FAKE_CROSSREF_RESPONSE["message"])
    )

    # Patch _extract_doi_from_url to return a fake DOI
    monkeypatch.setattr(
        validator_with_temp_whitelist_fuzzy_enabled,
        "_extract_doi_from_url",
        lambda url: "10.1000/test"
    )

    search_result = SearchResultItem(
        title="Some title",
        url="https://doi.org/10.1000/test",
        query="test query",
    )

    input_data = SourceValidatorInputSchema(
        search_results=[search_result],
    )

    result = await validator_with_temp_whitelist_fuzzy_enabled._arun(params=input_data)

    assert len(result.validated_results) == 1
    validation = result.validated_results[0]
    assert validation.is_whitelisted is True
    assert validation.whitelist_category == "AI"
    assert validation.confidence_score == 1.0
    assert result.summary["whitelisted_count"] == 1
    assert result.summary["error_count"] == 0