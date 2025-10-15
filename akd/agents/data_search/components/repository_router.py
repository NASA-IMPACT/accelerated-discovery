"""
Repository Router Component.

Uses LLM to determine which data source can best provide information for each
scientific decomposition.
"""

from enum import Enum
from typing import Union

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema

from ..utils.prompt_loader import load_and_format_prompt
from ._base import BaseDataSearchComponent
from .scientific_decomposition import ScientificDecomposition
from .topic_splitting import Topic


class NASARepositoryEnum(str, Enum):
    """NASA repository names that can be processed by this agent."""

    CMR = "CMR"
    PDS4 = "PDS4"
    # GCN = "GCN"
    # HEK = "HEK"
    # NAVO = "NAVO"
    # ORDR = "ORDR"
    # SPASE = "SPASE"


class Route(BaseModel):
    """Routing decision for a single scientific decomposition."""

    repository: Union[NASARepositoryEnum, str] = Field(
        ...,
        description="Single best repository - NASA repo (CMR, PDS4, etc.) or external source name (USGS, NOAA, etc.)",
    )
    rationale: str = Field(
        ...,
        description="Explanation for why this repository was selected",
    )
    is_external: bool = Field(
        ...,
        description="True if repository is external (non-NASA), False if NASA repository",
    )


class RepositoryRoutingInputSchema(InputSchema):
    """Input schema for repository routing."""

    original_query: str = Field(
        ...,
        description="Original research question for context",
    )
    topic: Topic = Field(..., description="Parent topic providing context")
    decomposition: ScientificDecomposition = Field(
        ...,
        description="Scientific decomposition to route to a data source",
    )


class RepositoryRoutingOutput(BaseModel):
    """Routing output for a single scientific decomposition."""

    route: Route = Field(..., description="Route for the input decomposition")


class RepositoryRouterComponent(
    BaseDataSearchComponent[RepositoryRoutingInputSchema, RepositoryRoutingOutput],
):
    """
    Component for routing scientific decompositions to appropriate data repositories.

    Takes scientific decompositions and uses LLM expertise to determine the single
    best data source (NASA repository like CMR/PDS4, or external source like USGS/NOAA).
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
        decomposition: ScientificDecomposition,
    ) -> RepositoryRoutingOutput:
        """
        Route a scientific decomposition to the best data repository.

        Args:
            original_query: Original scientific research question for context
            topic: Parent topic providing context
            decomposition: Scientific decomposition to route

        Returns:
            RepositoryRoutingOutput with single best repository
        """
        if self.debug:
            logger.debug(
                f"Routing decomposition '{decomposition.title}' to data repository using LLM",
            )

        # Format the user prompt
        user_prompt = self._format_user_prompt(original_query, topic, decomposition)

        # Save prompt to file for debugging
        self._save_prompt_to_file(
            user_prompt,
            "repository_routing",
            decomposition.title,
        )

        # Add user message to memory
        self._add_user_message(user_prompt)

        # Execute with retry logic
        response = await self._execute_with_retry(
            operation_name="route decomposition",
            custom_error_prefix="Failed to route decomposition",
        )

        if self.debug:
            logger.debug(
                f"LLM routed to: {response.route.repository} "
                f"(external: {response.route.is_external})",
            )

        return response

    def _format_user_prompt(
        self,
        original_query: str,
        topic: Topic,
        decomposition: ScientificDecomposition,
    ) -> str:
        """Format the user prompt with research context, topic, and decomposition."""
        # Dynamically get available NASA repositories
        from akd.agents.data_search.handlers import get_implemented_repositories

        available_repos = get_implemented_repositories()
        nasa_repositories_available = ", ".join(
            [repo.value for repo in available_repos],
        )

        return load_and_format_prompt(
            "repository_routing_user",
            original_query=original_query,
            topic_title=topic.title,
            topic_context=topic.functional_context,
            decomposition_title=decomposition.title,
            decomposition_justification=decomposition.scientific_justification,
            nasa_repositories_available=nasa_repositories_available,
        )

    async def _arun(
        self,
        params: RepositoryRoutingInputSchema,
        **kwargs,
    ) -> RepositoryRoutingOutput:
        """Execute the repository routing analysis."""
        return await self.process(
            params.original_query,
            params.topic,
            params.decomposition,
        )
