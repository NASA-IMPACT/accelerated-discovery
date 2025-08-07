import argparse
import asyncio
import json

from loguru import logger

from akd.agents.extraction import (
    EstimationExtractionAgent,
    IntentBasedExtractionSchemaMapper,
)
from akd.agents.factory import create_query_agent
from akd.agents.intents import IntentAgent
from akd.agents.litsearch import LitAgent, LitAgentInputSchema
from akd.tools.scrapers.composite import CompositeScraper, ResearchArticleResolver
from akd.tools.scrapers.pdf_scrapers import SimplePDFScraper
from akd.tools.scrapers.resolvers import ADSResolver, ArxivResolver, IdentityResolver
from akd.tools.scrapers.web_scrapers import Crawl4AIWebScraper, SimpleWebScraper
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig
from akd.tools.text_splitter import TextSplitterTool
from akd.tools.vector_db_tool import VectorDBTool, VectorDBToolConfig


async def main(args):
    # search_config = SearxNGSearchToolConfig(max_results=3)
    search_tool = SearxNGSearchTool(
        # config=search_config
        )

    scraper = CompositeScraper(
        SimpleWebScraper(),
        Crawl4AIWebScraper(),
        SimplePDFScraper(),
        debug=True,
    )

    article_resolver = ResearchArticleResolver(
        ArxivResolver(),
        ADSResolver(),
        IdentityResolver(),
    )

    text_splitter = TextSplitterTool()
    vector_db_config = VectorDBToolConfig(
        db_path="./",
        collection_name="lit_agent_demo",
    )
    vector_db_tool = VectorDBTool(config=vector_db_config)

    intent_agent = IntentAgent()
    query_agent = create_query_agent()

    schema_mapper = IntentBasedExtractionSchemaMapper()
    extraction_agent = EstimationExtractionAgent()

    lit_agent = LitAgent(
        intent_agent=intent_agent,
        schema_mapper=schema_mapper,
        query_agent=query_agent,
        extraction_agent=extraction_agent,
        search_tool=search_tool,
        web_scraper=scraper,
        article_resolver=article_resolver,
        text_splitter=text_splitter,
        vector_db_tool=vector_db_tool,
    )

    lit_agent.clear_history()

    result = await lit_agent.arun(
        LitAgentInputSchema(query=args.query, max_search_results=5),
    )
    logger.info(result.model_dump())

    with open("./temp/test_lit_agent.json", "w") as f:
        f.write(json.dumps(result.model_dump(mode="json")["results"], indent=2))


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
