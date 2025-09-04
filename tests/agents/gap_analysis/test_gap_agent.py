from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest
from pydantic import AnyUrl

from akd.agents.gap_analysis.gap_analysis import (
    GapAgent,
    GapAgentConfig,
    GapInputSchema,
    GapOutputSchema,
)
from akd.agents.gap_analysis.structures import ParsedPaper, Section
from akd.configs.project import get_project_settings
from akd.structures import PaperDataItem, SearchResultItem
from akd.tools.scrapers import DoclingScraperConfig
from akd.tools.search import SemanticScholarSearchToolConfig

BASE_PAPER_DATA = dict(
    paper_id="b62aa6e18a5bd37f54af6fe9ab29fc265b8cc078",
    corpus_id=None,
    external_ids={
        "DBLP": "conf/aaai/",
        "ArXiv": "2504.06136",
        "DOI": "10.1609/aaai.v39i28.35362",
        "CorpusId": 277628287,
    },
    url="http://arxiv.org/abs/2406.12031v2",
    title="QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform",
    abstract=None,
    venue=None,
    publication_venue=None,
    year=None,
    reference_count=None,
    citation_count=None,
    influential_citation_count=None,
    is_open_access=False,
    open_access_pdf={"url": "", "status": None, "license": None},
    fields_of_study=None,
    s2_fields_of_study=None,
    publication_types=None,
    publication_date=None,
    journal=None,
    citation_styles=None,
    authors=None,
    citations=None,
    references=None,
    embedding=None,
    tldr=None,
    external_id="2504.06136",
)


@pytest.fixture
def dummy_search_results():
    """Creates dummy search results."""
    return [
        SearchResultItem(
            url=AnyUrl("http://arxiv.org/abs/2504.06136v1"),
            title="QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform",
            query="test query",
            pdf_url=AnyUrl("http://arxiv.org/pdf/2504.06136v1"),
            content="We present QGen Studio: an adaptive question-an...",
        ),
    ]


@pytest.fixture
def dummy_paper_item():
    "Creates a dummy PaperDataItem"
    return [PaperDataItem(**BASE_PAPER_DATA)]


@pytest.fixture
def dummy_parsed_paper():
    """Create a dummy ParsedPaper"""
    return [
        ParsedPaper(
            **BASE_PAPER_DATA,
            figures=None,
            tables=None,
            section_titles=[["Introduction"], ["Related Work"]],
            sections=[
                Section(
                    title="Introduction",
                    content="Large language models (LLMs) have shown remarkable ...",
                    subsections=[],
                ),
                Section(
                    title="Related Work",
                    content="The use of LLMs for generating synthetic...",
                    subsections=[],
                ),
            ],
        ),
    ]


@pytest.fixture
def agent():
    """Creates an agent for further tests."""
    project_settings = get_project_settings()
    openai_key = project_settings.model_config_settings.api_keys.openai
    docling_config = DoclingScraperConfig(
        do_table_structure=True,
        pdf_mode="accurate",
        export_type="html",
        debug=False,
    )
    s2_config = SemanticScholarSearchToolConfig(
        debug=False,
        external_id="ARXIV",
        fields=["paperId", "title", "externalIds", "isOpenAccess", "openAccessPdf"],
    )
    gap_agent_config = GapAgentConfig(
        docling_config=docling_config,
        s2_tool_config=s2_config,
        model_name="gpt-4o-mini",
        api_key=openai_key,
        debug=True,
    )
    gap_agent = GapAgent(gap_agent_config)
    return gap_agent


@pytest.mark.asyncio
async def test_fetch_paper_items(agent, dummy_search_results, dummy_paper_item):
    """Tests fetch paper items given a list of search results and paper items."""
    agent.semantic_search_tool.fetch_paper_by_external_id = AsyncMock(
        return_value=dummy_paper_item,
    )
    paper_items, search_results = await agent._fetch_paper_items(dummy_search_results)
    assert len(paper_items) == 1
    assert len(paper_items) == len(search_results)


@pytest.mark.asyncio
async def test_fetch_parsed_pdfs(agent, dummy_search_results):
    """Tests fetch parsed pdfs given a list of search results."""
    dummy_result = MagicMock()
    dummy_result.content = "<html>dummy content</html>"
    agent.docling_scraper.arun = AsyncMock(return_value=dummy_result)
    parsed_pdfs = await agent._fetch_parsed_pdfs(dummy_search_results)
    assert isinstance(parsed_pdfs, list)
    assert "<html>" in parsed_pdfs[0]


@pytest.mark.asyncio
async def test_fetch_parsed_papers(agent, dummy_parsed_paper, dummy_paper_item):
    """Tests whether parsed pdfs are fetched properly."""
    dummy_parsed_pdfs = ["<html>paper content</html>"]
    agent._create_parsed_paper = AsyncMock(side_effect=dummy_parsed_paper)
    parsed_papers = await agent._fetch_parsed_papers(
        dummy_parsed_pdfs,
        dummy_paper_item,
    )
    assert isinstance(parsed_papers, list)
    assert all(p.__class__.__name__ == "ParsedPaper" for p in parsed_papers)
    assert parsed_papers[0].title == dummy_parsed_paper[0].title
    assert len(parsed_papers) == len(dummy_parsed_pdfs)


@pytest.mark.asyncio
async def test_create_graph(agent, dummy_parsed_paper):
    """Tests graph creation."""
    G = await agent.create_graph(dummy_parsed_paper)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) > 0


@pytest.mark.asyncio
@patch(
    "akd.agents.gap_analysis.gap_analysis.generate_final_answer",
    new_callable=AsyncMock,
)
@patch("akd.agents.gap_analysis.gap_analysis.select_nodes", new_callable=AsyncMock)
async def test_arun_full_pipeline(
    select_nodes,
    generate_final_answer,
    agent,
    dummy_paper_item,
    dummy_search_results,
    dummy_parsed_paper,
):
    """Tests the whole pipeline with a short paper."""
    select_nodes.return_value = [["node1"], ["node2"]]
    generate_final_answer.return_value = (
        "final_answer_mock",
        {"node1": "source", "node2": "source"},
    )
    dummy_graph = nx.Graph()
    dummy_graph.add_nodes_from(["node1", "node2"])
    dummy_graph.add_edge("node1", "node2")
    agent._fetch_paper_items = AsyncMock(
        return_value=(dummy_paper_item, dummy_search_results),
    )
    agent._fetch_parsed_pdfs = AsyncMock(return_value=["<html>paper content</html>"])
    agent._fetch_parsed_papers = AsyncMock(return_value=dummy_parsed_paper)
    agent.create_graph = AsyncMock(return_value=dummy_graph)

    params = GapInputSchema(search_results=dummy_search_results, gap="evidence")
    result = await agent.arun(params)

    assert isinstance(result, GapOutputSchema)

    assert isinstance(result.graph, Dict)
    assert isinstance(result.attributed_source_answers, Dict)
    assert isinstance(result.output, str)

    assert len(result.output) > 0
    assert len(result.attributed_source_answers) > 0
