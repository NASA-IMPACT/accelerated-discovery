from loguru import logger
from pydantic import Field
from akd._base import InputSchema, OutputSchema
from akd.tools.code_search import (
    CodeSearchTool,
    CodeSearchToolInputSchema,
    CodeSearchToolOutputSchema,
)
from akd.agents.query import (
    QueryAgent,
    QueryAgentInputSchema,
    FollowUpQueryAgent,
    FollowUpQueryAgentInputSchema,
)
from akd.agents.relevancy import (
    MultiRubricRelevancyAgent,
    MultiRubricRelevancyInputSchema,
)


class CodeSearchAgentInputSchema(InputSchema):
    """Input schema for the CodeSearchAgent."""

    user_query: str = Field(..., description="The user query to be processed.")
    num_queries: int = Field(
        default=3, description="The number of reformulated queries to generate."
    )
    max_results: int = Field(
        default=5,
        description="The maximum number of results to return from the code search tool.",
    )
    max_iterations: int = Field(
        default=5,
        description="The maximum number of iterations to run the relevancy agent.",
    )


class CodeSearchAgentOutputSchema(OutputSchema):
    """Output schema for the CodeSearchAgent."""

    final_queries: list[str] = Field(..., description="The final queries generated.")
    relevancy: dict = Field(
        ..., description="The relevancy scores for the final queries."
    )
    code_results: CodeSearchToolOutputSchema = Field(
        ..., description="The code search results."
    )


class CodeSearchAgent:
    """Agent for searching code repositories."""

    def __init__(
        self,
        query_agent: QueryAgent,
        followup_agent: FollowUpQueryAgent,
        relevancy_agent: MultiRubricRelevancyAgent,
        code_search_tool: CodeSearchTool,
        debug: bool = False,
    ) -> None:
        self.query_agent = query_agent
        self.followup_agent = followup_agent
        self.relevancy_agent = relevancy_agent
        self.code_search_tool = code_search_tool
        self.debug = debug

    async def _arun(
        self, params: CodeSearchAgentInputSchema, **kwargs
    ) -> CodeSearchAgentOutputSchema:
        user_query = params.user_query
        num_queries = params.num_queries
        max_results = params.max_results
        max_iterations = params.max_iterations

        if self.debug:
            logger.debug(f"\nUser Query: {params}\n")

        # Step 0: Generate initial queries
        current_queries = (
            await self.query_agent.arun(
                QueryAgentInputSchema(query=user_query, num_queries=num_queries)
            )
        ).queries

        if self.debug:
            logger.debug(f"\nInitial Reformulated Queries: {current_queries}\n")

        # Loop through the relevancy iterations
        for iteration in range(max_iterations):
            if self.debug:
                logger.debug(f"\nIteration {iteration + 1}\n")

            # Step 1: Code search
            code_results = await self.code_search_tool.arun(
                CodeSearchToolInputSchema(
                    queries=current_queries, max_results=max_results
                )
            )

            if not code_results.results:
                if self.debug:
                    logger.debug("No results returned from code search tool.")
                break

            content = "---\n".join(
                [
                    f"Repo {idx}: URL={item.url} | Content={item.content}"
                    for (idx, item) in enumerate(code_results.results, start=1)
                ]
            )

            # Step 2: Relevancy check
            queries_str = "\n".join(current_queries)
            relevancy_output = await self.relevancy_agent.arun(
                MultiRubricRelevancyInputSchema(
                    query=queries_str,
                    content=content,
                )
            )

            relevancy_scores = relevancy_output.model_dump()
            if self.debug:
                logger.debug(f"\nRelevancy Scores: {relevancy_scores}\n")

            # Stopping Criteria: If the topic alignment is aligned, return the results (Need to extend this to other rubrics)
            if relevancy_scores["topic_alignment"].value == "aligned":
                if self.debug:
                    logger.debug("Topic alignment is 'aligned'. Exiting loop early.")
                return CodeSearchAgentOutputSchema(
                    final_queries=current_queries,
                    relevancy=relevancy_scores,
                    code_results=code_results,
                )

            # Step 3: Follow-up query generation using the relevancy scores
            merged_content = (
                content
                + "\n"
                + "\n".join([f"{k}: {v}" for k, v in relevancy_scores.items()])
            )

            followup_queries = await self.followup_agent.arun(
                FollowUpQueryAgentInputSchema(
                    original_queries=current_queries,
                    content=merged_content,
                    num_queries=num_queries,
                )
            )

            current_queries = followup_queries.followup_queries
            if self.debug:
                logger.debug(f"Next Queries: {current_queries}")

        if self.debug:
            logger.debug("iterations completed. Returning final results.")

        return CodeSearchAgentOutputSchema(
            final_queries=current_queries,
            relevancy=relevancy_scores,
            code_results=code_results,
        )
