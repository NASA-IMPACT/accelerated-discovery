from typing import List

from langchain_core.documents import Document
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import BaseAgent
from akd.structures import ExtractionDTO
from akd.tools.scrapers.resolvers import BaseArticleResolver, ResolverInputSchema
from akd.tools.scrapers.web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolInputSchema

# --- MODIFIED: Re-importing the TextSplitterTool and its schema ---
from akd.tools.text_splitter import TextSplitterInputSchema, TextSplitterTool
from akd.tools.vector_db_tool import VectorDBTool

from .extraction import (
    EstimationExtractionAgent,
    ExtractionInputSchema,
    ExtractionSchemaMapper,
)
from .intents import IntentAgent
from .query import QueryAgent, QueryAgentInputSchema


class LitAgentInputSchema(InputSchema):
    """
    Input schema for the LitAgent
    """

    query: str = Field(..., description="Query to search for relevant web pages")
    max_search_results: int = Field(
        5,
        description="Maximum number of search results to retrieve",
    )


class LitAgentOutputSchema(OutputSchema):
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
        text_splitter: TextSplitterTool,
        vector_db_tool: VectorDBTool,
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
        self.text_splitter = text_splitter
        self.vector_db_tool = vector_db_tool

        self.n_queries = n_queries
        super().__init__(debug=debug)

    async def get_response_async(self, *args, **kwargs) -> LitAgentOutputSchema:
        """
        This method is required by the BaseAgent but is not used by LitAgent,
        which is an orchestrator. The main entry point is the `_arun` method.
        """
        raise NotImplementedError()

    async def _arun(
        self,
        params: LitAgentInputSchema,
        **kwargs,
    ) -> LitAgentOutputSchema:
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
                max_results=params.max_search_results,
            ),
        )

        # Log search results
        logger.info(f"Found {len(search_results.results)} relevant web pages:")
        contents = []
        docs_to_split = []
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
            except Exception:
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

            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "id": str(url),
                        "source": str(url),
                        "title": result.title,
                    },
                )
                docs_to_split.append(doc)

        # Split and index the documents
        if docs_to_split:
            logger.info(f"Splitting {len(docs_to_split)} documents into chunks...")
            splitter_output = await self.text_splitter.arun(
                TextSplitterInputSchema(
                    documents=docs_to_split,
                ),
            )
            docs_to_index = splitter_output.chunks

            if docs_to_index:
                logger.info(f"Indexing {len(docs_to_index)} document chunks...")
                try:
                    self.vector_db_tool.index(documents=docs_to_index)
                except Exception as e:
                    logger.error(f"Failed to index document chunks in VectorDB: {e}")

        results = []
        for content in contents:
            self.extraction_agent.reset_memory()
            try:
                answer = await self.extraction_agent.arun(
                    ExtractionInputSchema(query=query, content=content.result),
                )
                logger.debug(f"Source={content.source} | Answer={answer}")
                if answer:
                    content.result = answer
                    results.append(content)
            except KeyboardInterrupt:
                break
            except Exception:
                continue
        return LitAgentOutputSchema(results=results)

    def clear_history(self) -> None:
        logger.warning("Clearing history for all the agents")
        self.query_agent.reset_memory()
        self.intent_agent.reset_memory()
