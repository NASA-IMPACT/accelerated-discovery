from typing import List
import json

from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import BaseAgent
from akd.structures import ExtractionDTO
from akd.tools.scrapers.resolvers import BaseArticleResolver, ResolverInputSchema
from akd.tools.scrapers._base import ScrapedMetadata
from akd.tools.scrapers.web_scrapers import (
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WebScraper,
)
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolInputSchema
from akd.tools.evaluator.base_evaluator import (
    LLMEvaluator,
    LLMEvaluatorInputSchema,
    LLMEvaluatorConfig,
)

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
        web_scraper: WebScraper,
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

    # get_response_async is required by the BaseAgent interface,
    # but it seems that the LitAgent runtime logic is all implemented in _arun.
    # Therefore, we can leave this method unimplemented or raise NotImplementedError.
    # TODO: Migrate the logic from _arun to get_response_async, and have
    # _arun call get_response_async.
    async def get_response_async(self, *args, **kwargs):
        pass

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
        for i, result in enumerate(search_results.results, 1):
            logger.debug(f"Result {i} : Scraping the url {result.url}")
            resolver_output = await self.article_resolver.arun(
                ResolverInputSchema(url=result.url),
            )
            url = resolver_output.url
            try:
                scraped_content = await self.web_scraper.arun(
                    ScraperToolInputSchema(url=url),
                )
            except Exception:
                logger.warning("Fallback to search result content, not webpage")
                scraped_content = ScraperToolOutputSchema(
                    content=result.content,
                    metadata=ScrapedMetadata(**result.model_dump()),
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
            try:
                answer = await self.extraction_agent.arun(
                    ExtractionInputSchema(query=query, content=content.result),
                )
                logger.debug(f"Source={content.source} | Answer={answer}")

                if not answer:
                    logger.warning(
                        f"No answer extracted for content from {content.source}."
                    )
                    continue

                content.result = answer
                results.append(content)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.warning(f"Error processing content from {content.source}: {e}")
                continue

        extraction_quality_metric = {
            "name": "Extraction Evaluation",
            "criteria": "Evaluate wether or not the answer correctly extracts and summarizes the content, in a way that is relevant to the input query",
        }
        extraction_structure_metric = {
            "name": "Extraction Structure Evaluation",
            "criteria": "Does the content includes a title and an abstract?",
        }
        extraction_evaluator_config = LLMEvaluatorConfig(
            custom_metrics=[extraction_quality_metric, extraction_structure_metric],
            threshold=0.5,
        )
        extraction_evaluator = LLMEvaluator(
            config=extraction_evaluator_config, debug=self.debug
        )
        evaluator_config = LLMEvaluatorConfig(threshold=5.0)
        evaluator = LLMEvaluator(config=evaluator_config, debug=self.debug)

        _results = []
        for content in results:
            evaluator_input = LLMEvaluatorInputSchema(
                input=params.query, output=json.dumps(content.result.model_dump())
            )

            extraction_evaluation = await extraction_evaluator._arun(evaluator_input)

            if not extraction_evaluation.success:
                logger.warning(
                    f"Low extraction evaluation score ({extraction_evaluation.score}) for content from {content.source}, skipping."
                )
                continue

            evaluation = await evaluator._arun(evaluator_input)

            if not evaluation.success:
                logger.warning(
                    f"Low evaluation score ({evaluation.score}) for content from {content.source}, skipping."
                )
                continue
            _results.append(content)

        return LitAgentOutputSchema(results=_results)

    def clear_history(self) -> None:
        logger.warning("Clearing history for all the agents")
        self.query_agent.reset_memory()
        self.intent_agent.reset_memory()
