"""
Repository Router Component.

Uses LLM to determine which data sources can provide information for each topic
identified in the scientific research question.
"""

from enum import Enum
from typing import List, Union

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema

from ..utils.prompt_loader import load_and_format_prompt
from ._base import BaseDataSearchComponent
from .topic_splitting import Topic


class NASARepositoryEnum(str, Enum):
    """NASA repository names that can be processed by this agent."""

    CMR = "CMR"
    PDS4 = "PDS4"
    GCN = "GCN"
    HEK = "HEK"
    NAVO = "NAVO"
    ORDR = "ORDR"
    SPASE = "SPASE"


class Route(BaseModel):
    """Routing decision for a single topic."""

    repositories: List[Union[NASARepositoryEnum, str]] = Field(
        ...,
        description="Repositories to consult for this topic - use NASARepositoryEnum for NASA repos, strings for external",
        min_items=1,
    )
    rationales: List[str] = Field(
        ...,
        description="Short rationales aligned to repositories",
        min_items=1,
    )


class RepositoryRoutingInputSchema(InputSchema):
    """Input schema for repository routing."""

    original_query: str = Field(
        ...,
        description="Original research question for context",
    )
    topic: Topic = Field(..., description="Single topic to route to data sources")


class RepositoryRoutingOutput(BaseModel):
    """Single topic routing output."""

    route: Route = Field(..., description="Route for the input topic")


class RepositoryRouterComponent(
    BaseDataSearchComponent[RepositoryRoutingInputSchema, RepositoryRoutingOutput],
):
    """
    Component for routing topics to appropriate data repositories using LLM.

    Takes scientific research topics and uses LLM expertise to determine which
    data repositories (CMR, USGS, EPA, NOAA) can best provide the needed datasets.
    """

    input_schema = RepositoryRoutingInputSchema
    output_schema = RepositoryRoutingOutput

    # Base class configuration
    template_name = "repository_routing"
    default_temperature = 0.1  # Low temperature for consistent routing decisions

    async def process(
        self,
        original_query: str,
        topic: Topic,
    ) -> RepositoryRoutingOutput:
        """
        Route a single topic to appropriate data repositories using LLM analysis.

        Args:
            original_query: Original scientific research question for context
            topic: Single topic to route to data sources

        Returns:
            RepositoryRoutingOutput with route for the topic
        """
        if self.debug:
            logger.debug(
                f"Routing topic '{topic.title}' to data repositories using LLM",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(original_query, topic)

        # Add user message to memory
        self._add_user_message(user_prompt)

        # Execute with retry logic
        response = await self._execute_with_retry(
            operation_name="route topics",
            custom_error_prefix="Failed to route topics",
        )

        if self.debug:
            logger.debug(
                f"LLM produced route for topic: {response.route.repositories}",
            )

        return response

    def _format_user_prompt(self, original_query: str, topic: Topic) -> str:
        """Format the user prompt with research context and single topic."""
        return load_and_format_prompt(
            "repository_routing_user",
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
        )

    async def _arun(
        self,
        params: RepositoryRoutingInputSchema,
        **kwargs,
    ) -> RepositoryRoutingOutput:
        """Execute the repository routing analysis."""
        return await self.process(params.original_query, params.topic)
