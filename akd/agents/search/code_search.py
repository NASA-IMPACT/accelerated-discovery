from __future__ import annotations

from loguru import logger
from pydantic import Field
from typing import Literal
import requests
import os

from akd.tools.code_search import (
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
    CombinedCodeSearchTool,
    CombinedCodeSearchToolConfig,
    CodeSearchTool,
)
from akd.agents.query import FollowUpQueryAgent, QueryAgent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.agents.search import (
    ControlledSearchAgent,
    ControlledSearchAgentConfig,
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
    data_file: str | None = Field(
        default=None, description="Path to local repository data file"
    )
    google_drive_file_id: str | None = Field(
        default="15kxTyLeBCPL82WjMTyDytcXag85vglCP",
        description="Google Drive file ID for repository database, uses gte-large embeddings",
    )

    # SDE search configuration
    sde_base_url: str = Field(
        default=os.getenv(
            "SDE_BASE_URL", "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search"
        ),
        description="SDE search API base URL",
    )
    sde_search_type: Literal["vector", "hybrid", "keyword"] = Field(
        default="vector", description="SDE search type"
    )
    sde_page_size: int = Field(default=100, description="SDE search page size")

    # Tool selection
    use_local_search: bool = Field(
        default=True, description="Enable local repository search"
    )
    use_sde_search: bool = Field(
        default=True, description="Enable SDE repository search"
    )


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

        search_tool = search_tool or self._setup_search_tool()

        # Setup agents
        self.query_agent = query_agent or self._setup_query_agent()
        self.followup_query_agent = (
            followup_query_agent or self._setup_followup_query_agent()
        )
        self.relevancy_agent = relevancy_agent or self._setup_relevancy_agent()

        # Create the underlying ControlledSearchAgent
        super().__init__(
            config=self.config,
            search_tool=search_tool,
            query_agent=query_agent,
            followup_query_agent=followup_query_agent,
            relevancy_agent=relevancy_agent,
            debug=self.config.debug,
        )

    def _setup_search_tool(self) -> CombinedCodeSearchTool:
        """Setup the combined search tool with local and SDE components."""

        tools = []

        # Setup local search tool
        if self.config.use_local_search:
            try:
                local_config = LocalRepoCodeSearchToolConfig(
                    embedding_model_name=self.config.embedding_model_name
                )

                if self.config.data_file:
                    if not os.path.exists(self.config.data_file):
                        if self.config.debug:
                            logger.warning(
                                f"[CodeSearchAgent] Local data_file not found: {self.config.data_file}. Skipping local search tool."
                            )
                        local_config = None
                    else:
                        local_config.data_file = self.config.data_file
                elif self.config.google_drive_file_id:
                    local_config.google_drive_file_id = self.config.google_drive_file_id
                else:
                    if self.config.debug:
                        logger.warning(
                            "[CodeSearchAgent] No data_file or google_drive_file_id provided for local search. Skipping local search tool."
                        )
                    local_config = None

                if local_config is not None:
                    local_tool = LocalRepoCodeSearchTool(config=local_config)
                    tools.append(local_tool)

            except Exception as e:
                if self.config.debug:
                    logger.warning(
                        f"[CodeSearchAgent] LocalRepoCodeSearchTool unavailable; continuing without it. Reason: {e}"
                    )

        # Setup SDE search tool
        if self.config.use_sde_search:
            try:
                url = self.config.sde_base_url
                # Sanity check on URL
                if not (url.startswith("http://") or url.startswith("https://")):
                    if self.config.debug:
                        logger.warning(
                            f"[CodeSearchAgent] sde_base_url '{url}' is not a valid URL."
                        )

                reachable = True
                try:
                    # Check if the URL is reachable
                    requests.head(url, timeout=5, allow_redirects=True)
                    if self.config.debug:
                        logger.info(
                            f"[CodeSearchAgent] Verified reachability of SDE URL: {url}"
                        )
                except requests.RequestException as e:
                    reachable = False
                    if self.config.debug:
                        logger.warning(
                            f"[CodeSearchAgent] Could not reach SDE URL '{url}'. Skipping SDE tool. Reason: {e}"
                        )

                if reachable:
                    sde_config = SDECodeSearchToolConfig(
                        base_url=url,
                        debug=self.config.debug,
                        search_mode=self.config.sde_search_type,
                        page_size=self.config.sde_page_size,
                    )
                    sde_tool = SDECodeSearchTool(config=sde_config)
                    tools.append(sde_tool)

            except Exception as e:
                if self.config.debug:
                    logger.warning(
                        f"[CodeSearchAgent] SDECodeSearchTool unavailable; continuing without it. Reason: {e}"
                    )

        # Finalize or fail
        if not tools:
            logger.error(
                "[CodeSearchAgent] No search tools could be initialized (local and SDE both unavailable)."
            )
            raise ValueError("No search tools available")

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
