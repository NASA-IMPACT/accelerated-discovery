"""Tests for streaming support."""

from unittest.mock import AsyncMock, patch

import pytest

from .conftest import LiteLLMTestInputSchema, LiteLLMTestOutputSchema, TestLiteLLMAgent


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM instructor."""
    with patch("instructor.from_litellm") as mock:
        client = AsyncMock()
        mock.return_value = client
        client.chat.completions.create = AsyncMock(
            return_value=LiteLLMTestOutputSchema(response="Test", confidence=0.9),
        )
        yield client


@pytest.fixture
def agent(mock_litellm, litellm_config):
    """Create agent with mock."""
    return TestLiteLLMAgent(config=litellm_config)


class TestStreaming:
    """Streaming tests."""

    @pytest.mark.asyncio
    async def test_astream_emits_events(self, agent, mock_litellm):
        """Test astream emits STARTING, RUNNING, COMPLETED."""
        events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        assert [e.event_type for e in events] == ["starting", "running", "completed"]

    @pytest.mark.asyncio
    async def test_astream_sets_source(self, agent, mock_litellm):
        """Test source is set on all events."""
        events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        assert all(e.source == "TestLiteLLMAgent" for e in events)

    @pytest.mark.asyncio
    async def test_astream_completed_has_output(self, agent, mock_litellm):
        """Test COMPLETED event has output."""
        events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        assert events[-1].output.response == "Test"

    @pytest.mark.asyncio
    async def test_astream_error_yields_failed(self, litellm_config):
        """Test error yields FAILED event."""
        with patch("instructor.from_litellm") as mock:
            client = AsyncMock()
            mock.return_value = client
            client.chat.completions.create = AsyncMock(side_effect=ValueError("Error!"))

            agent = TestLiteLLMAgent(config=litellm_config)
            events = []

            with pytest.raises(ValueError):
                async for e in agent.astream(LiteLLMTestInputSchema(query="test")):
                    events.append(e)

            assert events[-1].event_type == "failed"
            assert events[-1].error == "Error!"
