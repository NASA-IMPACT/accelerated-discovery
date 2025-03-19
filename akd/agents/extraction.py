from abc import ABC, abstractmethod
from typing import Any, List, Union

from atomic_agents.agents.base_agent import BaseIOSchema
from loguru import logger
from pydantic import Field

from ..structures import ExtractionSchema, SingleEstimation
from .intents import Intent


class ExtractionSchemaMapper(ABC):
    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        return self.run(*args, **kwargs)


class IntentBasedExtractionSchemaMapper(ExtractionSchemaMapper):
    """
    If Intent is ESTIMATION, return a type `List[SingleEstimation]`.

    If GENERAL, return base ExtractionSchema
    """

    def run(self, intent: Intent) -> Union[ExtractionSchema, List[SingleEstimation]]:
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
