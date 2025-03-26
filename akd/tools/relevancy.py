import itertools
from typing import List, Optional

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from loguru import logger
from pydantic import Field

from ..agents.relevancy import (
    RelevancyAgent,
    RelevancyAgentInputSchema,
    RelevancyAgentOutputSchema,
)
from ..structures import RelevancyLabel


class RelevancyCheckerInputSchema(RelevancyAgentInputSchema):
    """Input schema for the RelevancyChecker."""

    pass


class _RelevancyCheckerSwappedInputSchema(BaseIOSchema):
    """Schema with swapped field order"""

    content: str = Field(
        ...,
        description="The content to check for relevance.",
    )
    query: str = Field(
        ...,
        description="The query to check for relevance.",
    )


class RelevancyCheckerOutputSchema(BaseIOSchema):
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

    model_config = {"arbitrary_types_allowed": True}

    debug: bool = Field(
        default=True,
        description="Boolean flag for debug mode",
    )
    n_iter: int = Field(
        default=2,
        description="Number of iterations for the relevancy check.",
    )
    swapping: bool = Field(
        default=True,
        description="Boolean flag for swapping query and content.",
    )
    agent: RelevancyAgent = Field(
        default_factory=RelevancyAgent,
        description="Relevancy agent to use for the check.",
    )


class RelevancyChecker(BaseTool):
    input_schema = RelevancyCheckerInputSchema
    output_schema = RelevancyCheckerOutputSchema

    def __init__(self, config: Optional[RelevancyCheckerConfig] = None):
        config = config or RelevancyCheckerConfig()
        super().__init__(config)
        self.debug = config.debug
        self.config = config

    def run(self, param: RelevancyCheckerInputSchema) -> RelevancyCheckerOutputSchema:
        logger.info(f"Running relevancy check for query: {param.query}")
        outputs = self._run(param, n_iter=self.config.n_iter)
        if self.config.swapping:
            logger.info(f"Running swapping pass. Query and Content swapped")
            param_swapped = _RelevancyCheckerSwappedInputSchema(
                content=param.content,
                query=param.query,
            )
            outputs.extend(self._run(param_swapped, n_iter=self.config.n_iter))
        return self._ensemble(outputs)

    def _run(
        self,
        param: RelevancyCheckerInputSchema,
        n_iter: int = 3,
    ) -> List[RelevancyAgentOutputSchema]:
        outputs = []
        for i in range(n_iter):
            output = self.config.agent.run(param)
            if self.debug:
                logger.debug(f"Relevancy check {i+1}: {output}")
            outputs.append(output)
            self.config.agent.reset_memory()
        return outputs

    def _ensemble(
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
