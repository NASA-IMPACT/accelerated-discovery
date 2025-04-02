from typing import List

from atomic_agents.agents.base_agent import BaseIOSchema
from loguru import logger
from pydantic import Field

from ..structures import ExtractionDTO
from ..tools.scrapers.resolvers import BaseArticleResolver, ResolverInputSchema
from ..tools.scrapers.web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)
from ..tools.search import SearxNGSearchTool, SearxNGSearchToolInputSchema
from ._base import BaseAgent, BaseAgentConfig
from .extraction import (
    EstimationExtractionAgent,
    ExtractionInputSchema,
    ExtractionSchemaMapper,
)
from .factory import create_extraction_agent
from .intents import IntentAgent, IntentInputSchema
from .query import QueryAgent, QueryAgentInputSchema


class LitAgentInputSchema(BaseIOSchema):
    """
    Input schema for the LitAgent
    """

    query: str = Field(..., description="Query to search for relevant web pages")


class LitAgentOutputSchema(BaseIOSchema):
    """
    Output schema for the LitAgent
    """

    results: List[ExtractionDTO] = Field(
        ...,
        description="List of extracted information",
    )


class LitAgent(BaseAgent):
    input_schema = LitAgentInputSchema
    output_schema = LitAgentOutputSchema

    def __init__(
        self,
        intent_agent: IntentAgent,
        schema_mapper: ExtractionSchemaMapper,
        query_agent: QueryAgent,
        extraction_agent: EstimationExtractionAgent,
        search_tool: SearxNGSearchTool,
        web_scraper: WebScraperToolBase,
        article_resolver: BaseArticleResolver,
        n_queries: int = 3,
        debug: bool = False,
    ) -> None:
        self.intent_agent = intent_agent
        self.schema_mapper = schema_mapper
        self.query_agent = query_agent
        self.extraction_agent = extraction_agent

        self.search_tool = search_tool
        self.web_scraper = web_scraper
        self.article_resolver = article_resolver

        self.n_queries = n_queries
        super().__init__(debug=debug)

    async def arun(self, params: LitAgentInputSchema) -> LitAgentOutputSchema:
        # intent_output = self.intent_agent.run(IntentInputSchema(query=params.query))
        # logger.debug(f"query={params.query} | intent={intent_output.intent}")

        # schema = self.schema_mapper(intent_output.intent)
        # logger.debug(f"Extraction schema={schema}")
        # extraction_agent = create_extraction_agent(schema)
        self.extraction_agent.reset_memory()

        logger.info("Analyzing input query to generate relevant search queries...")
        query_agent_output = await self.query_agent.arun(
            QueryAgentInputSchema(query=params.query, num_queries=self.n_queries),
        )
        query_agent_output.queries.insert(0, params.query)
        logger.debug("Generated search queries:")
        for i, query in enumerate(query_agent_output.queries, 1):
            logger.debug(f"Query {i}: {query}")

        # Perform the search
        logger.info("Searching across the web using SearxNG...")
        search_results = await self.search_tool.arun(
            SearxNGSearchToolInputSchema(
                queries=query_agent_output.queries,
                category="science",
            ),
        )

        # Log search results
        logger.info(f"Found {len(search_results.results)} relevant web pages:")
        contents = []
        for i, result in enumerate(search_results.results, 1):
            logger.debug(f"Result {i} : Scraping the url {result.url}")
            resolver_output = await self.article_resolver.arun(
                ResolverInputSchema(url=result.url),
            )
            url = resolver_output.url
            try:
                scraped_content = await self.web_scraper.arun(
                    WebpageScraperToolInputSchema(url=url),
                )
            except:
                logger.warning("Fallback to search result content, not webpage")
                scraped_content = WebpageScraperToolOutputSchema(
                    content=result.content,
                    metadata=WebpageMetadata(**result.model_dump()),
                )

            content = scraped_content.content or result.content
            logger.debug(
                f"Result {i}: {result.title} | "
                f"{url} | {content[:100]}.. | words={len(content.split())}",
            )
            contents.append(ExtractionDTO(source=str(url), result=content))

        results = []
        for content in contents:
            self.extraction_agent.reset_memory()
            answer = await self.extraction_agent.arun(
                ExtractionInputSchema(query=query, content=content.result),
            )
            logger.debug(f"Source={content.source} | Answer={answer}")
            if answer:
                content.result = answer
                results.append(content)
        return results

    def clear_history(self) -> None:
        logger.warning("Clearing history for all the agents")
        self.query_agent.reset_memory()
        self.intent_agent.reset_memory()
        try:
            self.query_agent.get_context_provider("scraped_content").content_items = []
            self.intent_agent.get_context_provider("scraped_content").content_items = []
        except:
            pass
