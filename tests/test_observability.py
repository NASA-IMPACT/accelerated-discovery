from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset_observability_module():
    import akd.observability as obs

    obs._INITIALIZED = False
    yield
    obs._INITIALIZED = False


def test_init_observability_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LOGFIRE_ENABLED", raising=False)
    import akd.observability as obs

    importlib.reload(obs)
    with patch.object(obs.logfire, "configure") as configure_mock, patch.object(
        obs.logfire, "instrument_litellm"
    ) as litellm_mock:
        obs.init_observability()
        configure_mock.assert_not_called()
        litellm_mock.assert_not_called()


def test_init_observability_configures_litellm_when_enabled(monkeypatch):
    monkeypatch.setenv("LOGFIRE_ENABLED", "true")
    monkeypatch.setenv("LOGFIRE_ENV", "test")
    monkeypatch.setenv("LOGFIRE_TOKEN", "test-token")
    monkeypatch.setenv("LOGFIRE_INSTRUMENT_HTTPX", "false")
    import akd.observability as obs

    importlib.reload(obs)
    with patch.object(obs.logfire, "configure") as configure_mock, patch.object(
        obs.logfire, "instrument_litellm"
    ) as litellm_mock, patch.object(obs.logfire, "instrument_httpx") as httpx_mock:
        obs.init_observability(service_name="accelerated-discovery-test")
        configure_mock.assert_called_once()
        kwargs = configure_mock.call_args.kwargs
        assert kwargs["service_name"] == "accelerated-discovery-test"
        assert kwargs["environment"] == "test"
        assert kwargs["inspect_arguments"] is False
        assert kwargs["token"] == "test-token"
        litellm_mock.assert_called_once()
        httpx_mock.assert_not_called()
