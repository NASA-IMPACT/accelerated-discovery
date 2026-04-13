"""Shared fixtures for search agent tests."""

import json
import os

import pytest
import requests


@pytest.fixture(scope="module")
def requires_searxng():
    """Skip test if SearxNG is unavailable. Only checks when a test uses this fixture."""
    url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    try:
        response = requests.head(url, timeout=5)
        if response.status_code >= 400:
            pytest.skip(f"SearxNG returned {response.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip(f"SearxNG unreachable at {url}")


@pytest.fixture(scope="module")
def requires_sde_api():
    """Skip test if SDE API is unavailable. Only checks when a test uses this fixture."""
    url = "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search"
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"page": 0, "pageSize": 1, "search_term": "test", "search_type": "keyword"}),
            timeout=5,
        )
        if response.status_code >= 400:
            pytest.skip(f"SDE API returned {response.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip(f"SDE API unreachable at {url}")
