from typing import List, Optional

from atomic_agents.agents.base_agent import BaseIOSchema
from langchain_core.documents import Document
from loguru import logger
from pydantic import Field

from akd.structures import ExtractionDTO
from akd.tools.fact_reasoner_tool import (
    FactReasonerInputSchema,
    FactReasonerOutputSchema,
    FactReasonerTool,
)
from akd.tools.scrapers.resolvers import BaseArticleResolver, ResolverInputSchema
from akd.tools.scrapers.web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolInputSchema
from akd.tools.vector_database import (
    VectorDBIndexInputSchema,
    VectorDBQueryInputSchema,
    VectorDBQueryOutputSchema,
    VectorDBSearchTool,
)

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
        n_queries: int = 1,
        debug: bool = False,
        vector_search_tool: Optional[VectorDBSearchTool] = None,
        fact_reasoner_tool: Optional[FactReasonerTool] = None,
    ) -> None:
        self.intent_agent = intent_agent
        self.schema_mapper = schema_mapper
        self.query_agent = query_agent
        self.extraction_agent = extraction_agent

        self.search_tool = search_tool
        self.web_scraper = web_scraper
        self.article_resolver = article_resolver
        self.vector_search_tool = vector_search_tool
        self.fact_reasoner_tool = fact_reasoner_tool

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

        if self.vector_search_tool is not None:
            # Store Search results in Vector DB
            langchain_docs = []
            assert len(search_results.results) == len(
                contents,
            ), "Search results and contents are out of sync!"

            for search_result, content in zip(search_results.results, contents):
                metadata = search_result.model_dump()
                # Remove large or redundant fields
                metadata.pop("content", None)
                metadata.pop("extra", None)

                sanitized_metadata = {}
                for key, value in metadata.items():
                    try:
                        if value is None:
                            continue
                        value_as_str = str(value)
                        sanitized_metadata[key] = value_as_str
                    except Exception:
                        continue

                langchain_docs.append(
                    Document(page_content=content.result, metadata=sanitized_metadata),
                )

            logger.info(f"Indexing {len(langchain_docs)} documents into Vector DB...")
            await self.vector_search_tool.arun_index(
                VectorDBIndexInputSchema(
                    documents=langchain_docs,
                ),
            )

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

        # Run FactReasoner on the results
        if self.fact_reasoner_tool is not None:
            sample_result = results[0]
            foobar = self.fact_reasoner_tool.arun(FactReasonerInputSchema(
                response=sample_result
            ))

            print(foobar)




        # TODO: could add FR as verification to 'longform_answer'
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
