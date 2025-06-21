import itertools
from typing import List

from loguru import logger
from pydantic import Field

from akd._base import OutputSchema
from akd.agents.relevancy import (
    RelevancyAgent,
    RelevancyAgentInputSchema,
    RelevancyAgentOutputSchema,
)
from akd.structures import RelevancyLabel
from akd.tools._base import BaseTool, BaseToolConfig


class RelevancyCheckerInputSchema(RelevancyAgentInputSchema):
    """Input schema for the RelevancyChecker."""

    pass


class RelevancyCheckerOutputSchema(OutputSchema):
    """Output schema for the RelevancyChecker."""

    score: float = Field(
        ...,
        description="The relevance score between the query and the content.",
        ge=0.0,
        le=1.0,
    )
    reasoning_steps: List[str] = Field(
        ...,
        description="The reasoning steps leading to the relevance check.",
    )


class RelevancyCheckerConfig(BaseToolConfig):
    """Configuration for the RelevancyChecker."""

    model_config = {"extra": "allow"}

    debug: bool = Field(
        default=True,
        description="Boolean flag for debug mode",
    )
    n_iter: int = Field(
        default=1,
        description="Number of iterations for the relevancy check.",
    )
    swapping: bool = Field(
        default=False,
        description="Boolean flag for swapping query and content.",
    )


class RelevancyChecker(
    BaseTool[RelevancyCheckerInputSchema, RelevancyCheckerOutputSchema],
):
    input_schema = RelevancyCheckerInputSchema
    output_schema = RelevancyCheckerOutputSchema

    config_schema = RelevancyCheckerConfig

    def __init__(
        self,
        config: RelevancyCheckerConfig | None = None,
        agent: RelevancyAgent | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config, debug=debug)
        self.agent = agent or RelevancyAgent()
        self._input_schema = self.agent.input_schema

    async def _arun(
        self,
        params: RelevancyCheckerInputSchema,
        **kwargs,
    ) -> RelevancyCheckerOutputSchema:
        logger.info(f"Running relevancy check for query: {params.query}")
        outputs = await self._run_agent(params, n_iter=self.n_iter)
        return await self._ensemble(outputs)

    async def _run_agent(
        self,
        params: RelevancyCheckerInputSchema,
        n_iter: int = 3,
    ) -> List[RelevancyAgentOutputSchema]:
        outputs = []
        if self.debug:
            logger.debug(f"Type of params: {type(params)}")
        for i in range(n_iter):
            output = await self.agent.arun(params)
            if self.debug:
                logger.debug(f"Relevancy check {i + 1}: {output}")
            outputs.append(output)
            self.agent.reset_memory()
        return outputs

    async def _ensemble(
        self,
        outputs: List[RelevancyAgentOutputSchema],
    ) -> RelevancyCheckerOutputSchema:
        logger.info(f"Ensembling {len(outputs)} outputs")
        relevance = len(
            list(filter(lambda r: r.label == RelevancyLabel.RELEVANT, outputs)),
        ) / len(outputs)
        return RelevancyCheckerOutputSchema(
            score=relevance,
            reasoning_steps=list(
                itertools.chain(*map(lambda r: r.reasoning_steps, outputs)),
            ),
        )
