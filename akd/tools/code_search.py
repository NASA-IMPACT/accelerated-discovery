from __future__ import annotations

from typing import List, Optional

from loguru import logger
from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.networks import HttpUrl
import os
import gdown
import pandas as pd

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig

from akd.tools.gcd.embeddings import Embedder
from akd.tools.gcd.vectorizer import RepoFinder
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig, SearxNGSearchToolInputSchema, SearxNGSearchToolOutputSchema

class CodeSearchToolInputSchema(InputSchema):
    """
    Input schema for the repository search tool.
    """

    queries: List[str] = Field(..., description="A list of search queries for finding repositories")
    top_k: int = Field(
        10,
        description="The maximum number of repository results to return.",
    )

class CodeSearchResultItem(BaseModel):
    """Schema for a single repository search result item."""
    url: str = Field(..., description="The URL of the found repository.")
    content: str = Field(
        ...,
        description="A snippet of content or description from the repository.",
    )

class CodeSearchOutputSchema(OutputSchema):
    """Output schema for the repository search tool."""
    results: List[CodeSearchResultItem] = Field(
        ...,
        description="A list of repository search result items aggregated from all queries.",
    )

class CodeSearchToolConfig(BaseToolConfig):
    """Configuration for the repository search tool."""
    data_file: str = os.getenv("CODE_SEARCH_DATA_FILE", ".code_data/repositories_with_embeddings.csv") # TODO: Replace with SDE API in future.
    embedding_model_name: str = os.getenv("CODE_SEARCH_MODEL", "all-MiniLM-L6-v2")
    debug: bool = False

"""
Tool for performing semantic code search for local code repositories. 
"""
class CodeSearchTool(BaseTool[CodeSearchToolInputSchema, CodeSearchOutputSchema]):
    """
    Tool for performing semantic code search. 
    It automatically downloads the necessary data file if it's not found locally.
    """
    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchOutputSchema
    config_schema = CodeSearchToolConfig

    def __init__(self, config: CodeSearchToolConfig | None = None, debug: bool = False):
        """
        Initializes the tool. If the data file is not found, it will be
        downloaded from Google Drive before loading the models.
        """
        config = config or self.config_schema()
        super().__init__(config, debug)
        self.repo_finder = None

        try:
            logger.info("Initializing CodeSearchTool...")
            self._ensure_data_file_exists() # Check for and download the data file

            logger.info("Loading data and embedding model...")
            repo_data = pd.read_csv(self.config.data_file)
            embedder = Embedder(self.config.embedding_model_name)
            self.repo_finder = RepoFinder(embedder=embedder, data=repo_data, debug=self.debug)

            logger.info("CodeSearchTool initialization complete.")
        except Exception as e:
            logger.error(f"Error during CodeSearchTool initialization: {e}")

    def _ensure_data_file_exists(self):
        """
        Checks if the data file exists and downloads it if it does not. 
        Will be replaced with SDE API in future.
        """
        data_file_path = self.config.data_file
        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found at '{data_file_path}'. Downloading...")
            
            # Ensure the target directory exists
            data_dir = os.path.dirname(data_file_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)
            
            # Download from Google Drive
            file_id = "1nPaEWD9Wuf115aEmqJQusCvJlPc7AP7O"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", data_file_path, quiet=False)
            logger.info(f"Data file downloaded successfully to '{data_file_path}'.")
        else:
            logger.info(f"Data file already exists at '{data_file_path}'.")

    async def _arun(self, params: CodeSearchToolInputSchema, **kwargs) -> CodeSearchOutputSchema:
        """
        Runs the in-memory code search for a list of queries.
        """
        if not self.repo_finder:
            logger.error("Cannot perform search because the tool is not initialized.")
            return self.output_schema(results=[])

        all_results_data = []
        for query in params.queries:
            if self.debug:
                logger.debug(f"Searching for query: '{query}' with top_k={params.top_k}")
            
            try:
                results = self.repo_finder.find_repo(query=query, top_k=params.top_k)
                if results:
                    all_results_data.extend(results)
            except Exception as e:
                logger.error(f"Error during search for query '{query}': {e}")
        
        formatted_results = [
            CodeSearchResultItem(
                url=result.get('URL'),
                content=result.get('text')
            )
            for result in all_results_data
        ]
        
        return self.output_schema(results=formatted_results)
    
"""
Tool for performing targeted searches on GitHub using SearxNG.
This tool is a wrapper around SearxNGSearchTool.
"""
class GitHubSearchTool(SearxNGSearchTool):
    """
    A specialized search tool for GitHub, using SearxNG as the backend.

    This tool is a wrapper around the general SearxNGSearchTool, but is
    hardcoded to search only the 'github' engine and the 'technology' category.
    """

    def __init__(
        self,
        config: SearxNGSearchToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initializes the GitHubSearchTool.

        This constructor enforces the 'github' engine for all searches.

        Args:
            config (SearxNGSearchToolConfig):
                Configuration for the tool. The `engines` property will
                be overridden.
            debug (bool): Enable debug logging.
        """
        config = config or SearxNGSearchToolConfig()

        # Hardcode the configuration for GitHub searching
        config.engines = ["github"]
        # Optional: Give the tool a more specific default title/description
        config.title = "GitHub Search"
        config.description = (
            "Tool for performing targeted searches on GitHub "
            "for code, repositories, and issues."
        )

        super().__init__(config, debug)

    @classmethod
    def from_params(
        cls,
        base_url: Optional[HttpUrl] = "http://localhost:8080",
        max_results: int = 10,
        # engines parameter is omitted as it's hardcoded
        max_pages: int = 5,
        results_per_page: int = 10,
        score_cutoff: float = 0.1,
        debug: bool = False,
    ) -> "GitHubSearchTool":
        """
        Creates a GitHubSearchTool instance from individual parameters,
        enforcing the 'github' engine.
        """
        base_url = base_url or os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
        config = SearxNGSearchToolConfig(
            base_url=base_url,
            max_results=max_results,
            # Enforce the specific engine for this tool
            engines=["github"],
            max_pages=max_pages,
            results_per_page=results_per_page,
            score_cutoff=score_cutoff,
            debug=debug,
            title="GitHub Search",
            description="Performs a targeted search on GitHub.",
        )
        return cls(config, debug)

    async def _arun(
        self,
        params: SearxNGSearchToolInputSchema,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> SearxNGSearchToolOutputSchema:
        """
        Runs the search tool, forcing the search category to 'technology'.

        This method intercepts the input parameters, sets the category,
        and then calls the parent class's `_arun` method to perform the
        actual search.

        Args:
            params (SearxNGSearchToolInputSchema):
                The input parameters for the tool. The 'category' field
                will be ignored and overridden.
            max_results (Optional[int]):
                The maximum number of search results to return.

        Returns:
            SearxNGSearchToolOutputSchema:
                The output of the tool, adhering to the output schema.
        """
        # Hardcode the category to 'technology' for every call
        params.category = "technology"

        if self.debug:
            logger.debug(
                f"GitHubSearchTool: Forcing category to '{params.category}' "
                f"and engines to {self.config.engines}"
            )

        # Call the parent's _arun method with the modified parameters
        return await super()._arun(params=params, max_results=max_results, **kwargs)