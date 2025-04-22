import pytest


import instructor
import openai

from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig
from akd.tools.scrapers.web_scrapers import (
    SimpleWebScraper,
    WebpageScraperToolConfig,
    Crawl4AIWebScraper,
)
from akd.tools.scrapers.pdf_scrapers import SimplePDFScraper
from akd.tools.scrapers.composite import CompositeWebScraper
from akd.tools.scrapers.resolvers import ArxivResolver, ADSResolver, IdentityResolver
from akd.tools.scrapers.composite import ResearchArticleResolver
from akd.agents.factory import create_query_agent
from akd.agents.extraction import (
    IntentBasedExtractionSchemaMapper,
    EstimationExtractionAgent,
)
from akd.agents.litsearch import LitAgent, LitAgentInputSchema, LitAgentOutputSchema
from akd.agents.intents import IntentAgent
from akd.agents._base import BaseAgentConfig

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import assert_test, evaluate

import pytest_asyncio
from pydantic import BaseModel


import os
import json


@pytest.fixture(scope="module")
def lit_agent():

    search_tool = SearxNGSearchTool(
        config=SearxNGSearchToolConfig(
            base_url="http://localhost:8080",
            max_results=5,
            engines=["google", "arxiv", "google_scholar"],
            debug=True,
        )
    )
    scraper_cfg = WebpageScraperToolConfig()
    scraper = CompositeWebScraper(
        SimpleWebScraper(scraper_cfg),
        Crawl4AIWebScraper(scraper_cfg),
        SimplePDFScraper(scraper_cfg),
        debug=True,
    )
    article_resolver = ResearchArticleResolver(
        ArxivResolver(), ADSResolver(), IdentityResolver()
    )
    intent_agent = IntentAgent(
        config=BaseAgentConfig(client=instructor.from_openai(openai.AsyncOpenAI()))
    )
    query_agent = create_query_agent()
    extraction_agent = EstimationExtractionAgent()
    schema_mapper = IntentBasedExtractionSchemaMapper()
    agent = LitAgent(
        intent_agent=intent_agent,
        schema_mapper=schema_mapper,
        query_agent=query_agent,
        extraction_agent=extraction_agent,
        search_tool=search_tool,
        web_scraper=scraper,
        article_resolver=article_resolver,
    )
    agent.clear_history()

    return agent


@pytest.fixture(scope="module")
def query():
    return "methods and estimation to map landslides in Nepal"


@pytest_asyncio.fixture(scope="module")
async def result(lit_agent, query):
    # Save the model dump to disk for caching
    # TODO: figure out caching with ser-de for LitAgentOutputSchema
    cache_path = "tests/lit_agent_output.json"
    # if os.path.exists(cache_path):
    #     with open(cache_path, "r") as f:
    #         cached = json.load(f)
    #         output = LitAgentOutputSchema.model_validate(cached)
    #     return cached

    output = await lit_agent.arun(LitAgentInputSchema(query=query))
    with open(cache_path, "w") as f:
        f.write(json.dumps([o.model_dump(mode="json") for o in output]))

    return output


@pytest_asyncio.fixture(scope="module")
async def test_cases(query, result):
    # Only include test cases where r.result.estimations is not empty
    return [
        LLMTestCase(
            input=query,
            actual_output=" - ".join(
                [e.answer for r in result for e in r.result.estimations]
            ),
        )
    ]


@pytest_asyncio.fixture(scope="module")
async def full_test_cases(query, result):

    return [LLMTestCase(input=query, actual_output=str(r.model_dump())) for r in result]


@pytest_asyncio.fixture(scope="module")
async def dataset(test_cases):
    return EvaluationDataset(test_cases=test_cases)


@pytest_asyncio.fixture(scope="module")
async def full_dataset(full_test_cases):
    return EvaluationDataset(test_cases=full_test_cases)


@pytest.mark.asyncio
async def test_answer_relevancy(dataset):
    metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1", include_reason=True)
    assert_test(dataset, [metric], run_async=False)


# @pytest.mark.asyncio
# async def test_faithfulness(dataset):
#    metric = FaithfulnessMetric(threshold=0.7, model="gpt-4.1", include_reason=True)
#    assert_test(dataset, [metric], run_async=False)


@pytest.mark.asyncio
async def test_geval_correctness(dataset):
    metric = GEval(
        name="Correctness",
        criteria="Correctness - determine if the actual output is a correct and helpful answer to the given query",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        strict_mode=True,
    )
    assert_test(dataset, [metric], run_async=False)


@pytest.mark.asyncio
async def test_full_geval_correctness(full_dataset):
    metric = GEval(
        name="Correctness",
        criteria=(
            "Correctness - determine if the actual output represents a thorough and accurate literature review in response to the given query. "
            "An output is considered correct if the answer is relevant to the user's query, adequate reasoning is provided, and if sources are likely to be relevant and helpful to the user's research question"
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        strict_mode=True,
    )
    assert_test(full_dataset, [metric], run_async=False)


@pytest.mark.asyncio
async def test_full_geval_geography(full_dataset):
    metric = GEval(
        name="Correctness",
        criteria=(
            "Correctness - determine if the actual output represents a thorough and accurate literature review in response to the given query. "
            "An output is considered correct if the answer is relevant to the user's query, adequate reasoning is provided, and if sources are likely to be relevant and helpful to the user's research question. "
            "Note that any geographic information in the user's query is EXTREMELY important, therefore any source that does not explicitly reference a geography in the user's query should be considered incorrect"
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        strict_mode=True,
    )

    assert_test(full_dataset, [metric], run_async=False)
