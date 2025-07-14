from __future__ import annotations

from typing import List, Optional

from loguru import logger
from pydantic.fields import Field
from pydantic.networks import HttpUrl
from pydantic import computed_field
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from akd.tools._base import BaseTool, BaseToolConfig

from akd.tools.misc import Embedder, HttpUrlAdapter
from akd.tools.search import SearxNGSearchTool, SearxNGSearchToolConfig, SearxNGSearchToolInputSchema, SearxNGSearchToolOutputSchema

from akd.structures import SearchResultItem
from akd.tools.search import SearchToolInputSchema, SearchToolOutputSchema
from akd.utils import get_akd_root, google_drive_downloader

"""
Define abstract base classes for code search tools.
"""
class CodeSearchToolInputSchema(SearchToolInputSchema):
    """
    Input schema for the code search tool.
    """
    pass

class CodeSearchToolOutputSchema(SearchToolOutputSchema):
    """
    Output schema for the code search tool.
    """
    pass

class CodeSearchToolConfig(BaseToolConfig):
    """Configuration for the code search tool."""
    pass
    
class CodeSearchTool(BaseTool[CodeSearchToolInputSchema, CodeSearchToolOutputSchema]):
    """
    Abstract base class for all code search tools.
    """
    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = CodeSearchToolConfig

"""
Tool for performing semantic code search for local code repositories. 
"""
class LocalRepoCodeSearchToolInputSchema(CodeSearchToolInputSchema):
    """
    Input schema for the local repository code search tool.
    """
    @computed_field
    @property
    def top_k(self) -> int:
        return self.max_results

class LocalRepoCodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the local repository code search tool.
    """
    data_file: str = str(get_akd_root() / "docs" / "repositories_with_embeddings.csv") 
    google_drive_file_id: str = os.getenv("CODE_SEARCH_FILE_ID", "1nPaEWD9Wuf115aEmqJQusCvJlPc7AP7O")
    embedding_model_name: str = os.getenv("CODE_SEARCH_MODEL", "all-MiniLM-L6-v2")
    debug: bool = False

class LocalRepoCodeSearchTool(CodeSearchTool):
    """
    Tool for performing semantic code search. 
    It automatically downloads the necessary data file if it's not found locally.
    """
    input_schema = LocalRepoCodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = LocalRepoCodeSearchToolConfig

    def __init__(self, config: LocalRepoCodeSearchToolConfig | None = None, debug: bool = False):
        """
        Initializes the tool. If the data file is not found, it will be
        downloaded from Google Drive before loading the models.
        """
        config = config or self.config_schema()
        super().__init__(config, debug)

        try:
            logger.info("Initializing CodeSearchTool...")
            self._ensure_data_file_exists() # Check for and download the data file

            logger.info("Loading data and embedding model...")
            self.repo_data = pd.read_csv(self.config.data_file)
            self.embedder = Embedder(self.config.embedding_model_name)

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
            
            if self.debug:
                logger.debug(f"Downloading data file from Google Drive (file_id={self.config.google_drive_file_id})...")
            # Download from Google Drive
            file_id = self.config.google_drive_file_id
            google_drive_downloader(file_id, data_file_path, quiet=False)
            if self.debug:
                logger.debug("Download complete.")
        else:
            logger.info(f"Data file already exists at '{data_file_path}'.")

    def find_repo(
        self,
        query: str,
        top_k: int = 25,
        embeddings_column: str = "embeddings",
        debug: bool = False,
    ) -> list[dict]:
        """
        Perform similarity search against cached embeddings using vectorized computation. # noqa

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of dictionaries with top results and similarity scores
        """
        if self.repo_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if embeddings_column not in self.repo_data.columns:
            raise ValueError(
                "No embeddings found. Run generate_embeddings() first.",
            )

        # Get query embedding
        query_embedding = self.embedder.embed_texts([query])
        if debug:
            logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Parse embeddings if they are in string format
        if self.repo_data[embeddings_column].dtype == "object":
            self.repo_data[embeddings_column] = self.repo_data[embeddings_column].apply(self.embedder._parse_embedding)

        # Stack all embeddings into a matrix
        embeddings_matrix = np.vstack(
            self.repo_data[embeddings_column].tolist(),
        )
        if debug:
            logger.debug(
                f"Embeddings matrix shape: {embeddings_matrix.shape}",
            )
       
        # Compute cosine distances using cdist (more efficient)
        # cdist with 'cosine' gives cosine distance (1 - cosine_similarity)
        cosine_distances = cdist(
            query_embedding.reshape(1, -1),
            embeddings_matrix,
            metric="cosine",
        )[0]  # Extract the single row

        # Convert cosine distances to similarity scores (0-1 range)
        similarities = np.clip(1 - cosine_distances, 0, 1)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = self.repo_data.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]

        return results.reset_index(drop=True).to_dict("records")

    async def _arun(self, params: CodeSearchToolInputSchema, **kwargs) -> CodeSearchToolOutputSchema:
        """
        Runs the in-memory code search for a list of queries.
        """
        all_results_data = []
        for query in params.queries:
            if self.debug:
                logger.debug(f"Searching for query: '{query}' with top_k={params.top_k}")
            
            try:
                results = self.find_repo(query=query, 
                                         top_k=params.top_k, 
                                         embeddings_column="embeddings", 
                                         debug=self.debug)
                if results:
                    all_results_data.extend(results)
            except Exception as e:
                logger.error(f"Error during search for query '{query}': {e}")
        
        formatted_results = [
            SearchResultItem(
                title="GitHub Repository",
                url=HttpUrlAdapter.validate_python(result.get('URL')),
                content=result.get('text'),
                query=query,
            )
            for result in all_results_data
        ]
        
        return self.output_schema(results=formatted_results)
    
"""
Tool for performing targeted searches on GitHub using SearxNG.
This tool is a wrapper around SearxNGSearchTool.
"""
class GitHubCodeSearchTool(CodeSearchTool, SearxNGSearchTool):
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
        base_url: Optional[HttpUrl] = None,
        max_results: int = 10,
        # engines parameter is omitted as it's hardcoded
        max_pages: int = 5,
        results_per_page: int = 10,
        score_cutoff: float = 0.1,
        debug: bool = False,
    ) -> "GitHubCodeSearchTool":
        """
        Creates a GitHubSearchTool instance from individual parameters,
        enforcing the 'github' engine.
        """
        base_url = base_url or HttpUrlAdapter.validate_python(os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"))
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