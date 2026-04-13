"""Tests for streaming support."""

from unittest.mock import MagicMock, patch

import pytest

from akd._base.streaming import StreamEvent, StreamEventType

from .conftest import LiteLLMTestInputSchema, TestLiteLLMAgent


def make_chunk(content: str = "", reasoning: str | None = None):
    """Create a mock streaming chunk."""
    chunk = MagicMock()
    chunk.usage = None
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.reasoning_content = reasoning
    chunk.choices[0].delta.thinking = None
    chunk.choices[0].delta.tool_calls = None
    return chunk


async def make_stream(*chunks):
    """Create async generator from chunks."""
    for chunk in chunks:
        yield chunk


class TestStreaming:
    """Streaming tests."""

    @pytest.mark.asyncio
    async def test_astream_emits_lifecycle_events(self, litellm_config):
        """Test astream emits STARTING, RUNNING, COMPLETED."""
        json_output = '{"response": "Test", "confidence": 0.9}'

        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.return_value = make_stream(make_chunk(json_output))

            agent = TestLiteLLMAgent(config=litellm_config)
            events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        event_types = [e.event_type for e in events]
        assert event_types[0] == "starting"
        assert "running" in event_types
        assert event_types[-1] == "completed"

    @pytest.mark.asyncio
    async def test_astream_sets_source(self, litellm_config):
        """Test source is set on all events."""
        json_output = '{"response": "Test", "confidence": 0.9}'

        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.return_value = make_stream(make_chunk(json_output))

            agent = TestLiteLLMAgent(config=litellm_config)
            events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        assert all(e.source == "TestLiteLLMAgent" for e in events)

    @pytest.mark.asyncio
    async def test_astream_completed_has_output(self, litellm_config):
        """Test COMPLETED event has output."""
        json_output = '{"response": "Test", "confidence": 0.9}'

        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.return_value = make_stream(make_chunk(json_output))

            agent = TestLiteLLMAgent(config=litellm_config)
            events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        assert events[-1].output.response == "Test"
        assert events[-1].output.confidence == 0.9

    @pytest.mark.asyncio
    async def test_astream_emits_thinking_events(self, litellm_config):
        """Test THINKING events for reasoning tokens."""
        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.return_value = make_stream(
                make_chunk(reasoning="Let me think..."),
                make_chunk('{"response": "Test", "confidence": 0.9}'),
            )

            agent = TestLiteLLMAgent(config=litellm_config)
            events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        thinking_events = [e for e in events if e.event_type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].thinking_content == "Let me think..."

    @pytest.mark.asyncio
    async def test_astream_emits_streaming_token_events(self, litellm_config):
        """Test StreamingTokenEvent for text deltas."""
        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.return_value = make_stream(
                make_chunk('{"response": "Test", "confidence": 0.9}'),
            )

            agent = TestLiteLLMAgent(config=litellm_config)
            events = [e async for e in agent.astream(LiteLLMTestInputSchema(query="test"))]

        streaming_events = [e for e in events if e.event_type == "streaming"]
        assert len(streaming_events) >= 1

    @pytest.mark.asyncio
    async def test_astream_error_yields_failed(self, litellm_config):
        """Test error yields FAILED event."""
        with patch("akd.agents._base._base.acompletion") as mock_acompletion:
            mock_acompletion.side_effect = ValueError("API Error!")

            agent = TestLiteLLMAgent(config=litellm_config)
            events = []

            with pytest.raises(ValueError):
                async for e in agent.astream(LiteLLMTestInputSchema(query="test")):
                    events.append(e)

        assert events[-1].event_type == "failed"
        assert events[-1].error == "API Error!"


class TestStreamEventProperties:
    """Tests for StreamEvent convenience properties."""

    def test_thinking_content_property(self):
        """Test thinking_content property returns data value."""
        event = StreamEvent(
            event_type=StreamEventType.THINKING,
            source="TestAgent",
            data={"thinking_content": "Let me think..."},
        )
        assert event.thinking_content == "Let me think..."

    def test_partial_output_property(self):
        """Test partial_output property returns data value."""
        event = StreamEvent(
            event_type=StreamEventType.PARTIAL,
            source="TestAgent",
            data={"partial_output": {"response": "partial"}},
        )
        assert event.partial_output == {"response": "partial"}

    def test_properties_return_none_when_missing(self):
        """Test properties return None when data keys are missing."""
        event = StreamEvent(
            event_type=StreamEventType.THINKING,
            source="TestAgent",
        )
        assert event.thinking_content is None
        assert event.partial_output is None
        assert event.output is None
        assert event.error is None
