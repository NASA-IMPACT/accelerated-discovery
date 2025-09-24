from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akd.agents.search.aspect_search.structures import Editor, Perspectives
from akd.agents.storm import StormAgent, StormInputSchema, StormOutputSchema
from akd.agents.storm.prompts import DRAFT_OUTLINE_PROMPT, REFINE_OUTLINE_PROMPT
from akd.agents.storm.structures import (
    ArticleSection,
    Outline,
    OutlineSection,
    ResearchState,
)
from akd.agents.storm.tools import get_draft_outline, get_refined_outline, retrieve
from akd.configs.project import get_project_settings
from akd.tools.search import SearchResultItem


@pytest.fixture
def dummy_topic():
    return "llm attribution mechanisms using attention distribution"


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
def dummy_outline(dummy_topic):
    return Outline(
        page_title=dummy_topic,
        sections=[OutlineSection(section_title="test", description="test")],
    )


@pytest.fixture
def dummy_state(dummy_topic, dummy_outline, dummy_perspectives):
    return ResearchState(
        topic=dummy_topic,
        outline=dummy_outline,
        perspectives=dummy_perspectives,
        article="This is a dummy article",
        references={
            "https://url1.com/": "content_1",
            "https://url2.com/": "content_2",
        },
        search_results=[
            SearchResultItem(
                title="dummy",
                url="https://url1.com/",
                query=dummy_topic,
                content="content_1",
            ),
            SearchResultItem(
                title="dummy",
                url="https://url2.com/",
                query=dummy_topic,
                content="content_2",
            ),
        ],
        sections=[
            ArticleSection(section_title="dummy", content="dummy"),
            ArticleSection(section_title="dummy", content="dummy"),
        ],
    )


@pytest.fixture
def agent():
    """Creates an agent for further tests."""
    project_settings = get_project_settings()
    openai_key = project_settings.model_config_settings.api_keys.openai
    agent = StormAgent(
        model_name="gpt-4o-mini",
        api_key=openai_key,
    )
    return agent


# =============================================================================
# Initialisation tests
# =============================================================================


@pytest.mark.asyncio
async def test_agent_input(dummy_topic):
    """Tests the input schema alias."""
    schema_with_alias = StormInputSchema(query=dummy_topic)
    assert schema_with_alias.topic == dummy_topic
    assert "configurable" in schema_with_alias.config.keys()
    schema_default = StormInputSchema(topic=dummy_topic)
    assert schema_default.topic == dummy_topic
    assert "configurable" in schema_default.config.keys()


@pytest.mark.asyncio
def test_storm_agent_initialisation(agent):
    """Test that StormAgent initializes correctly"""
    assert agent.fast_llm is not None
    assert agent.long_context_llm is not None
    assert agent.vectorstore is not None
    assert agent.retriever is not None
    assert agent.storm is not None


# =============================================================================
# Tool tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_draft_outline(dummy_topic, dummy_outline):
    """Test get_draft_outline functions returns Outline"""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.invoke.return_value = dummy_outline
    mock_llm.with_structured_output.return_value = MagicMock()
    with patch.object(
        type(DRAFT_OUTLINE_PROMPT),
        "__or__",
        return_value=mock_pipeline,
    ):
        result = get_draft_outline(dummy_topic, mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(Outline)
    mock_pipeline.invoke.assert_called_once_with({"topic": dummy_topic})
    assert result == dummy_outline
    assert isinstance(result, Outline)


@pytest.mark.asyncio
async def test_get_refined_outline(dummy_topic, dummy_outline):
    """Test get_refined_outline functions returns Outline"""
    mock_llm = MagicMock()
    mock_pipeline = AsyncMock()
    mock_pipeline.ainvoke.return_value = dummy_outline
    mock_llm.with_structured_output.return_value = MagicMock()
    dummy_old_outline = dummy_outline
    dummy_conversations = "dummy conversations"

    with patch.object(
        type(REFINE_OUTLINE_PROMPT),
        "__or__",
        return_value=mock_pipeline,
    ):
        result = await get_refined_outline(
            dummy_topic,
            dummy_old_outline,
            dummy_conversations,
            mock_llm,
        )
    mock_llm.with_structured_output.assert_called_once_with(Outline)
    mock_pipeline.ainvoke.assert_awaited_once_with(
        {
            "topic": dummy_topic,
            "old_outline": dummy_old_outline,
            "conversations": dummy_conversations,
        },
    )
    assert result == dummy_outline
    assert isinstance(result, Outline)


@pytest.mark.asyncio
async def test_retrieve_formats_docs_correctly(dummy_topic):
    """Test retriever returns docs"""
    dummy_inputs = {"topic": dummy_topic, "section": "test section"}
    dummy_docs = [
        MagicMock(metadata={"source": "source1"}, page_content="content"),
        MagicMock(metadata={"source": "source2"}, page_content="content"),
    ]
    expected_refs = {"source1": "content", "source2": "content"}
    expected_docs = (
        '<Document href="source1"/>\ncontent\n</Document>\n'
        '<Document href="source2"/>\ncontent\n</Document>'
    )
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = dummy_docs
    result = await retrieve(dummy_inputs, mock_retriever)
    mock_retriever.ainvoke.assert_awaited_once_with(f"{dummy_topic}: test section")
    assert result["docs"] == expected_docs
    assert result["topic"] == dummy_topic
    assert result["references"] == expected_refs


# =============================================================================
# Pipeline tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_response_async(agent, dummy_topic, dummy_state):
    """Test get_response_async"""
    if agent.config.debug:
        pytest.skip("Debug has to be set to false to test default ainvoke.")
    agent.storm.ainvoke = AsyncMock(return_value=dummy_state)
    input_params = StormInputSchema(topic=dummy_topic)
    result = await agent.get_response_async(input_params)
    assert isinstance(result, StormOutputSchema)
    assert isinstance(result.article, str)
    assert isinstance(result.perspectives, Perspectives)
    assert isinstance(result.references, Dict)
    assert isinstance(result.search_results, List)
    assert isinstance(result.outline, Outline)
    assert isinstance(result.sections, List)

    assert len(result.article) != 0
    assert len(result.perspectives.editors) != 0
    assert len(result.references) != 0
    assert len(result.search_results) != 0
    assert len(result.sections) != 0


@pytest.mark.asyncio
async def test_get_response_async_debug(agent, dummy_state, dummy_topic):
    """Test get_response_async in debug mode"""
    agent.config.debug = True
    dummy_storm = AsyncMock()

    async def astream_mock(*args, **kwargs):
        for chunk in [
            {**dummy_state, "step": 1},
            {**dummy_state, "step": 2},
        ]:
            yield chunk

    dummy_storm.astream = astream_mock
    dummy_state_obj = MagicMock()
    dummy_state_obj.values = dummy_state
    dummy_storm.get_state = MagicMock(return_value=dummy_state_obj)

    agent.storm = dummy_storm
    input_params = StormInputSchema(topic=dummy_topic)
    result = await agent.get_response_async(input_params)

    assert isinstance(result, StormOutputSchema)
    assert isinstance(result.article, str)
    assert isinstance(result.perspectives, Perspectives)
    assert isinstance(result.references, Dict)
    assert isinstance(result.search_results, List)
    assert isinstance(result.outline, Outline)
    assert isinstance(result.sections, List)

    assert len(result.article) != 0
    assert len(result.perspectives.editors) != 0
    assert len(result.references) != 0
    assert len(result.search_results) != 0
    assert len(result.sections) != 0

    dummy_storm.get_state.assert_called_once_with(config=input_params.config)


@pytest.mark.asyncio
async def test_arun(agent, dummy_topic, dummy_state):
    """Tests _arun simply calls get_response_async."""
    params = StormInputSchema(topic=dummy_topic)
    agent.get_response_async = AsyncMock(
        return_value=StormOutputSchema(
            article=dummy_state["article"],
            references=dummy_state["references"],
            perspectives=dummy_state["perspectives"],
            search_results=dummy_state["search_results"],
            outline=dummy_state["outline"],
            sections=dummy_state["sections"],
        ),
    )
    result = await agent.arun(params)
    assert isinstance(result, StormOutputSchema)
