from __future__ import annotations

from typing import List

from loguru import logger
from pydantic import BaseModel
from pydantic.fields import Field
import os
import gdown
import pandas as pd

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig

from akd.tools.gcd.embeddings import Embedder
from akd.tools.gcd.vectorizer import RepoFinder

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