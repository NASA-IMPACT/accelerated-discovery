import argparse
import asyncio
import json

import instructor
import openai

from akd.agents._base import BaseAgentConfig
from akd.agents.extraction import (
    EstimationExtractionAgent,
    IntentBasedExtractionSchemaMapper,
)
from akd.agents.factory import create_query_agent
from akd.agents.intents import IntentAgent
from akd.agents.litsearch import LitAgent, LitAgentInputSchema
from akd.configs.lit_config import get_lit_agent_settings
from akd.configs.project import CONFIG
from akd.tools.scrapers.composite import CompositeWebScraper, ResearchArticleResolver
from akd.tools.scrapers.pdf_scrapers import SimplePDFScraper
from akd.tools.scrapers.resolvers import ADSResolver, ArxivResolver, IdentityResolver
from akd.tools.scrapers.web_scrapers import Crawl4AIWebScraper, SimpleWebScraper
from akd.tools.search import SearxNGSearchTool
from akd.tools.vector_database import VectorDBSearchTool


async def main(args):
    lit_agent_config = get_lit_agent_settings(args.config)
    search_config = lit_agent_config.search
    scraper_config = lit_agent_config.scraper
    vector_search_config = lit_agent_config.vectordb_search

    search_tool = SearxNGSearchTool(config=search_config)
    vector_search_tool = VectorDBSearchTool(config=vector_search_config)

    scraper = CompositeWebScraper(
        SimpleWebScraper(scraper_config),
        Crawl4AIWebScraper(scraper_config),
        SimplePDFScraper(scraper_config),
        debug=True,
    )

    article_resolver = ResearchArticleResolver(
        ArxivResolver(),
        ADSResolver(),
        IdentityResolver(),
    )

    intent_agent = IntentAgent(
        config=BaseAgentConfig(client=instructor.from_openai(openai.AsyncOpenAI())),
    )

    query_agent = create_query_agent()
    schema_mapper = IntentBasedExtractionSchemaMapper()
    extraction_agent = EstimationExtractionAgent()

    lit_agent = LitAgent(
        intent_agent=intent_agent,
        schema_mapper=schema_mapper,
        query_agent=query_agent,
        extraction_agent=extraction_agent,
        search_tool=search_tool,
        vector_search_tool=vector_search_tool,
        web_scraper=scraper,
        article_resolver=article_resolver,
    )

    lit_agent.clear_history()

    result = await lit_agent.arun(LitAgentInputSchema(query=args.query))

    print(result[0].model_dump())
    breakpoint()

    with open("test_lit_agent.json", "w") as f:
        f.write(json.dumps([r.model_dump(mode="json") for r in result], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LitAgent pipeline")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query to run through the LitAgent pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lit_agent.toml",
        help="Path to the TOML config file for LitAgent",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
