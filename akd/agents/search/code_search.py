from __future__ import annotations

import os
from typing import Literal

from loguru import logger
from pydantic import Field

from akd.agents.query import FollowUpQueryAgent, QueryAgent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.configs.code_prompts import CODE_QUERY_PROMPT, CODE_RELEVANCY_PROMPT
from akd.tools.reranker import RerankerToolConfig
from akd.tools.search.code_search import (
    CodeSearchTool,
    CombinedCodeSearchTool,
    CombinedCodeSearchToolConfig,
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
)
from akd.utils import is_server_available

from ._base import BaseAgentConfig
from .controlled import ControlledSearchAgent, ControlledSearchAgentConfig


class CodeSearchAgentConfig(ControlledSearchAgentConfig):
    """Configuration for CodeSearchAgent extending ControlledSearchAgentConfig."""

    # Model configurations
    subagent_model: str = Field(default="gpt-4o-mini", description="Model for query and relevancy agents")
    reranker_tool: str = Field(
        default="cross-encoder",
        description="The tool to use for reranking the combined results.",
    )
    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
        description="The model to use with the cross-encoder tool for reranking the combined results.",
    )

    # Local search configuration
    embedding_model_name: str = Field(default="thenlper/gte-large", description="Embedding model for local search")
    data_file: str | None = Field(default=None, description="Path to local repository data file")
    google_drive_file_id: str | None = Field(
        default="19Oi2gAE5aGyp11mIxCXsaGp1PA08_Aw3",
        description="Google Drive file ID for repository database, uses gte-large embeddings",
    )

    # SDE search configuration
    sde_base_url: str = Field(
        default=os.getenv("SDE_BASE_URL", "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search"),
        description="SDE search API base URL",
    )
    sde_search_type: Literal["vector", "hybrid", "keyword"] = Field(default="vector", description="SDE search type")
    sde_page_size: int = Field(default=100, description="SDE search page size")

    # Tool selection
    use_local_search: bool = Field(default=True, description="Enable local repository search")
    use_sde_search: bool = Field(default=True, description="Enable SDE repository search")

    # System prompts
    query_prompt: str = Field(default=CODE_QUERY_PROMPT, description="System prompt for query agent")
    followup_query_prompt: str = Field(default=CODE_QUERY_PROMPT, description="System prompt for follow-up query agent")
    relevancy_prompt: str = Field(default=CODE_RELEVANCY_PROMPT, description="System prompt for relevancy agent")


class CodeSearchAgent(ControlledSearchAgent):
    """
    Wrapper for code repository search using ControlledSearchAgent.
    """

    def __init__(
        self,
        config: CodeSearchAgentConfig | None = None,
        search_tool: CodeSearchTool | None = None,
        query_agent: QueryAgent | None = None,
        followup_query_agent: FollowUpQueryAgent | None = None,
        relevancy_agent: MultiRubricRelevancyAgent | None = None,
        debug: bool = False,
    ):
        """Initialize CodeSearchAgent with custom components."""

        self.config = config or CodeSearchAgentConfig()
        self.config.debug = debug

        self.search_tool = search_tool or self._setup_search_tool()

        # Setup agents
        self.query_agent = query_agent or self._setup_query_agent()
        self.followup_query_agent = followup_query_agent or self._setup_followup_query_agent()
        self.relevancy_agent = relevancy_agent or self._setup_relevancy_agent()

        # Create the underlying ControlledSearchAgent
        super().__init__(
            config=self.config,
            search_tool=self.search_tool,
            query_agent=self.query_agent,
            followup_query_agent=self.followup_query_agent,
            relevancy_agent=self.relevancy_agent,
            debug=self.config.debug,
        )

    def _setup_local_tool(self) -> LocalRepoCodeSearchTool | None:
        """Setup local repository search tool with error handling."""
        if not self.config.use_local_search:
            return None

        try:
            local_config = LocalRepoCodeSearchToolConfig(embedding_model_name=self.config.embedding_model_name)

            if self.config.data_file:
                if not os.path.exists(self.config.data_file):
                    if self.config.debug:
                        logger.warning(
                            f"[CodeSearchAgent] Local data_file not found: {self.config.data_file}. Skipping local search tool.",
                        )
                    return None
                else:
                    local_config.data_file = self.config.data_file
            elif self.config.google_drive_file_id:
                local_config.google_drive_file_id = self.config.google_drive_file_id
            else:
                if self.config.debug:
                    logger.warning(
                        "[CodeSearchAgent] No data_file or google_drive_file_id provided for local search. Skipping local search tool.",
                    )
                return None

            return LocalRepoCodeSearchTool(config=local_config)

        except Exception as e:
            if self.config.debug:
                logger.warning(
                    f"[CodeSearchAgent] LocalRepoCodeSearchTool unavailable; continuing without it. Reason: {e}",
                )
            return None

    def _setup_sde_tool(self) -> SDECodeSearchTool | None:
        """Setup SDE search tool with error handling."""
        if not self.config.use_sde_search:
            return None

        try:
            url = self.config.sde_base_url
            reachable = is_server_available(url)
            if not reachable:
                return None

            sde_config = SDECodeSearchToolConfig(
                base_url=url,
                debug=self.config.debug,
                search_mode=self.config.sde_search_type,
                page_size=self.config.sde_page_size,
            )
            return SDECodeSearchTool(config=sde_config)

        except Exception as e:
            if self.config.debug:
                logger.warning(f"[CodeSearchAgent] SDECodeSearchTool unavailable; continuing without it. Reason: {e}")
            return None

    def _setup_search_tool(self) -> CombinedCodeSearchTool:
        """Setup the combined search tool with local and SDE components."""

        tools = []

        # Setup local search tool
        local_tool = self._setup_local_tool()
        if local_tool:
            tools.append(local_tool)

        # Setup SDE search tool
        sde_tool = self._setup_sde_tool()
        if sde_tool:
            tools.append(sde_tool)

        # Finalize or fail
        if not tools:
            logger.error("[CodeSearchAgent] No search tools could be initialized (local and SDE both unavailable).")
            raise ValueError("No search tools available")

        # Create combined tool with correct field names
        combined_config = CombinedCodeSearchToolConfig(
            reranker_type=self.config.reranker_tool,
            reranker_config=RerankerToolConfig(
                model_name=self.config.cross_encoder_model_name,
            ),
        )

        return CombinedCodeSearchTool(config=combined_config, tools=tools)

    def _setup_query_agent(self) -> QueryAgent:
        """Setup query agent with code-specific prompt."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model,
            system_prompt=self.config.query_prompt,
            debug=self.config.debug,
        )
        return QueryAgent(config=config)

    def _setup_followup_query_agent(self) -> FollowUpQueryAgent:
        """Setup follow-up query agent."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model,
            system_prompt=self.config.followup_query_prompt,
            debug=self.config.debug,
        )
        return FollowUpQueryAgent(config=config)

    def _setup_relevancy_agent(self) -> MultiRubricRelevancyAgent:
        """Setup relevancy agent."""
        config = BaseAgentConfig(
            model_name=self.config.subagent_model,
            system_prompt=self.config.relevancy_prompt,
            debug=self.config.debug,
        )
        return MultiRubricRelevancyAgent(config=config)
