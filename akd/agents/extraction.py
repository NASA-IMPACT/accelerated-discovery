import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Union

from atomic_agents.agents.base_agent import BaseAgent, BaseIOSchema
from loguru import logger
from pydantic import Field

from ..structures import ExtractionSchema, SingleEstimation
from ..utils import AsyncRunMixin
from ._base import BaseAgent
from .intents import Intent


class ExtractionSchemaMapper(ABC, AsyncRunMixin):
    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    def __call__(self, *args, **kwargs) -> Any:
        return self.run(*args, **kwargs)


class IntentBasedExtractionSchemaMapper(ExtractionSchemaMapper):
    """
    If Intent is ESTIMATION, return a type `List[SingleEstimation]`.

    If GENERAL, return base ExtractionSchema
    """

    async def arun(
        self,
        intent: Intent,
        **kwargs,
    ) -> Union[ExtractionSchema, List[SingleEstimation]]:
        res = ExtractionSchema
        if intent == Intent.ESTIMATION:
            res = List[SingleEstimation]
        if self.debug:
            logger.debug(f"Intent={intent} | Schema={res}")
        return res


class ExtractionInputSchema(BaseIOSchema):
    """Information Extraction input schema"""

    query: str = Field(..., description="Query that is used for answering/extraction")
    content: str = Field(
        ...,
        description="Actual text/content to extract information from",
    )


class EstimationExtractionOutputSchema(BaseIOSchema):
    """Estimation Extraction output schema"""

    estimations: List[SingleEstimation] = Field(
        ...,
        description="List of estimations extracted from the query and the content",
    )


class EstimationExtractionAgent(BaseAgent):
    input_schema = ExtractionInputSchema
    output_schema = EstimationExtractionOutputSchema
