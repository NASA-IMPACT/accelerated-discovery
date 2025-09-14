from __future__ import annotations

from loguru import logger
from pydantic import Field
from typing import Optional

from akd.tools.code_search import (
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
    CombinedCodeSearchTool,
    CombinedCodeSearchToolConfig,
)
from akd.agents.query import FollowUpQueryAgent, QueryAgent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.agents.search import (
    ControlledSearchAgent,
    ControlledSearchAgentConfig,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
)
from akd.configs.code_prompts import CODE_QUERY_PROMPT, CODE_RELEVANCY_PROMPT

from ._base import BaseAgentConfig


class CodeSearchAgentConfig(ControlledSearchAgentConfig):
    """Configuration for CodeSearchAgent extending ControlledSearchAgentConfig."""

    # Model configurations
    subagent_model: str = Field(
        default="gpt-4o-mini", description="Model for query and relevancy agents"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
        description="Re-ranker model for combined search",
    )

    # Local search configuration
    embedding_model_name: str = Field(
        default="thenlper/gte-large", description="Embedding model for local search"
    )
    data_file: Optional[str] = Field(
        default=None, description="Path to local repository data file"
    )
    google_drive_file_id: Optional[str] = Field(
        default="15kxTyLeBCPL82WjMTyDytcXag85vglCP",
        description="Google Drive file ID for repository database, uses gte-large embeddings",
    )

    # SDE search configuration
    sde_base_url: str = Field(
        default="https://d2kqty7z3q8ugg.cloudfront.net/api/code/search",
        description="SDE search API base URL",
    )
    sde_search_type: str = Field(
        default="vector", description="SDE search type (vector, hybrid, keyword)"
    )
    sde_page_size: int = Field(default=100, description="SDE search page size")

    # Tool selection
    use_local_search: bool = Field(
        default=True, description="Enable local repository search"
    )
    use_sde_search: bool = Field(
        default=True, description="Enable SDE repository search"
    )


class CodeSearchAgent:
    """
    Wrapper for code repository search using ControlledSearchAgent.
    """

    def __init__(
        self,
        config: Optional[CodeSearchAgentConfig] = None,
        search_tool: Optional[CombinedCodeSearchTool] = None,
        query_agent: Optional[QueryAgent] = None,
        followup_query_agent: Optional[FollowUpQueryAgent] = None,
        relevancy_agent: Optional[MultiRubricRelevancyAgent] = None,
        debug: Optional[bool] = None,
    ):
        """Initialize CodeSearchAgent with custom components."""

        self.config = config or CodeSearchAgentConfig()
        if debug is not None:
            self.config.debug = debug

        # Setup search tool
        self.search_tool = search_tool or self._setup_search_tool()

        # Setup agents
        self.query_agent = query_agent or self._setup_query_agent()
        self.followup_query_agent = (
            followup_query_agent or self._setup_followup_query_agent()
        )
        self.relevancy_agent = relevancy_agent or self._setup_relevancy_agent()

        # Create the underlying ControlledSearchAgent
        self._controlled_agent = ControlledSearchAgent(
            config=self.config,
            search_tool=self.search_tool,
            query_agent=self.query_agent,
            followup_query_agent=self.followup_query_agent,
            relevancy_agent=self.relevancy_agent,
            debug=self.config.debug,
        )

    def _setup_search_tool(self) -> CombinedCodeSearchTool:
        """Setup the combined search tool with local and SDE components."""

        tools = []

        # Setup local search tool
        if self.config.use_local_search:
            local_config = LocalRepoCodeSearchToolConfig(
                embedding_model_name=self.config.embedding_model_name
            )

            # Use data_file if provided, otherwise use google_drive_file_id
            if self.config.data_file:
                local_config.data_file = self.config.data_file
            elif self.config.google_drive_file_id:
                local_config.google_drive_file_id = self.config.google_drive_file_id

            local_tool = LocalRepoCodeSearchTool(config=local_config)
            tools.append(local_tool)

        # Setup SDE search tool
        if self.config.use_sde_search:
            sde_config = SDECodeSearchToolConfig(
                base_url=self.config.sde_base_url,
                debug=self.config.debug,
                search_mode=self.config.sde_search_type,
                page_size=self.config.sde_page_size,
            )
            sde_tool = SDECodeSearchTool(config=sde_config)
            tools.append(sde_tool)

        if not tools:
            logger.error("No search tools enabled")
            raise ValueError("No search tools enabled")

        # Create combined tool
        combined_config = CombinedCodeSearchToolConfig(
            reranker_model_name=self.config.reranker_model
        )

        return CombinedCodeSearchTool(config=combined_config, tools=tools)

    def _setup_query_agent(self) -> QueryAgent:
        """Setup query agent with code-specific prompt."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model, system_prompt=CODE_QUERY_PROMPT
        )
        return QueryAgent(config=config)

    def _setup_followup_query_agent(self) -> FollowUpQueryAgent:
        """Setup follow-up query agent."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model, system_prompt=CODE_QUERY_PROMPT
        )
        return FollowUpQueryAgent(config=config)

    def _setup_relevancy_agent(self) -> MultiRubricRelevancyAgent:
        """Setup relevancy agent."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model, system_prompt=CODE_RELEVANCY_PROMPT
        )
        return MultiRubricRelevancyAgent(config=config)

    async def arun(
        self, input_schema: LitSearchAgentInputSchema
    ) -> LitSearchAgentOutputSchema:
        """
        Perform code repository search.

        Args:
            input_schema: Input schema containing query and search parameters

        Returns:
            Search results from ControlledSearchAgent
        """

        return await self._controlled_agent.arun(input_schema)
