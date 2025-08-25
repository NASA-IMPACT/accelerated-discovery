from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd.agents.search.aspect_search import (
    AspectSearchAgent,
    AspectSearchConfig,
    AspectSearchInputSchema,
    AspectSearchOutputSchema,
)
from akd.agents.search.aspect_search.structures import Editor, Perspectives
from akd.configs.project import get_project_settings
from akd.structures import SearchResultItem


@pytest.fixture
def agent():
    project_settings = get_project_settings()
    openai_key = project_settings.model_config_settings.api_keys.openai
    config = AspectSearchConfig(
        model_name="gpt-4o-mini",
        api_key=openai_key,
        max_turns=2,
        num_editors=1,
    )
    aspect_agent = AspectSearchAgent(config)
    return aspect_agent


@pytest.fixture
def dummy_perspectives():
    return Perspectives(
        editors=[
            Editor(
                affiliation="University Researcher",
                name="Dr. Alice Chen",
                role="Machine Learning Researcher",
                description="Dr. Chen will focus on the theoretical foundations of attention mechanisms in large language models (LLMs), exploring how attention distributions can be interpreted to understand model behavior and decision-making processes.",
            ),
        ],
    )


@pytest.fixture
def dummy_topic():
    return "llm attribution mechanisms using attention distribution"


@pytest.fixture
def dummy_interview_result(dummy_perspectives):
    return (
        [
            SearchResultItem(
                url="http://example.com",
                title="test",
                query="test",
                content="content",
            ),
        ],
        {"http://example.com": "content"},
        dummy_perspectives,
    )


@pytest.fixture
def dummy_interview():
    return [
        {
            "messages": [
                MagicMock(content="test message", name="Subject_Matter_Expert"),
            ],
            "search_results": [
                SearchResultItem(
                    url="http://example.com",
                    title="test",
                    query="test",
                    content="content",
                ),
            ],
            "references": {"http://example.com": "content"},
        },
    ]


@pytest.mark.asyncio
async def test_agent_defaults(agent):
    """Tests the agents configuration."""
    assert agent.config.max_turns == 2
    assert agent.config.num_editors == 1
    assert agent.config.top_n_wiki_results == 3


@pytest.mark.asyncio
@patch("akd.agents.search.aspect_search.aspect_search.survey_subjects")
async def test_get_perspectives(
    mock_survey_subjects,
    agent,
    dummy_perspectives,
    dummy_topic,
):
    """Tests that perspectives are retrieved correctly."""
    mock_survey_subjects.ainvoke = AsyncMock(return_value=dummy_perspectives)
    perspectives = await agent.get_perspectives(topic=dummy_topic)
    assert isinstance(perspectives, Perspectives)
    assert isinstance(perspectives.editors[0], Editor)
    assert isinstance(perspectives.editors, List)
    assert len(perspectives.editors) == agent.config.num_editors


@pytest.mark.asyncio
@patch("akd.agents.search.aspect_search.aspect_search.survey_subjects")
async def test_conduct_interviews(
    mock_survey_subjects,
    agent,
    dummy_topic,
    dummy_perspectives,
    dummy_interview,
):
    """Tests interview graph"""
    mock_survey_subjects.ainvoke = AsyncMock(return_value=dummy_perspectives)

    agent.interview_graph = MagicMock()
    agent.interview_graph.abatch = AsyncMock(return_value=dummy_interview)
    search_results, references, perspectives = await agent._conduct_interviews(
        dummy_topic,
    )

    assert isinstance(search_results, list)
    assert isinstance(references, dict)
    assert isinstance(perspectives, Perspectives)
    assert len(references) <= len(search_results)
    assert "http://example.com" in references
    assert perspectives.editors[0].affiliation == "University Researcher"


@pytest.mark.asyncio
async def test_get_response_async_single_topic(
    agent,
    dummy_topic,
    dummy_interview_result,
):
    """Tests agent for a single topic."""
    agent._conduct_interviews = AsyncMock(return_value=dummy_interview_result)
    params = AspectSearchInputSchema(topic=dummy_topic)
    result = await agent.get_response_async(params)

    assert isinstance(result, AspectSearchOutputSchema)
    assert isinstance(result.search_results, List)
    assert isinstance(result.references, Dict)
    assert isinstance(result.perspectives, Perspectives)
    assert isinstance(result.perspectives.editors[0], Editor)


@pytest.mark.asyncio
async def test_get_response_async_multiple_topics(
    agent,
    dummy_interview_result,
    dummy_topic,
):
    """Tests agent for multiple topics."""
    agent._conduct_interviews = AsyncMock(side_effect=[dummy_interview_result] * 2)

    params = AspectSearchInputSchema(topic=[dummy_topic] * 2)
    result = await agent.get_response_async(params)

    assert isinstance(result, AspectSearchOutputSchema)
    assert isinstance(result.search_results, List)
    assert isinstance(result.references, List)
    assert isinstance(result.perspectives, List)
    assert isinstance(result.perspectives[0], Perspectives)
    assert len(result.search_results) == 2
    assert len(result.references) == 2


@pytest.mark.asyncio
async def test_arun(agent, dummy_topic):
    """Tests _arun simply calls get_response_async."""
    params = AspectSearchInputSchema(topic=dummy_topic)
    agent.get_response_async = AsyncMock(
        return_value=AspectSearchOutputSchema(
            search_results=[],
            references={},
            perspectives=Perspectives(editors=[]),
        ),
    )
    result = await agent._arun(params)
    assert isinstance(result, AspectSearchOutputSchema)
