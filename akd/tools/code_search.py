from __future__ import annotations

import os
from typing import Literal, Optional
import requests
import json
import time

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ValidationError, computed_field, Field
from pydantic.networks import HttpUrl
from scipy.spatial.distance import cdist
from tenacity import retry, stop_after_attempt
from sentence_transformers import CrossEncoder

from akd.errors import SchemaValidationError
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.tools.misc import Embedder, HttpUrlAdapter, OpenAIEmbedder
from akd.tools.search import (
    SearchToolInputSchema,
    SearchToolOutputSchema,
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
)
from akd.utils import get_akd_root, google_drive_downloader


class CodeSearchToolInputSchema(SearchToolInputSchema):
    """
    Input schema for the code search tool.
    """

    @computed_field
    def top_k(self) -> int:
        """Returns the number of top results to return."""
        return self.max_results


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

    def _validate_input(
        self,
        params: CodeSearchToolInputSchema | SearchToolInputSchema | dict,
    ) -> CodeSearchToolInputSchema:
        """Validate and convert input parameters."""
        if isinstance(params, self.input_schema):
            return params

        if isinstance(params, dict):
            try:
                params = self.input_schema(**params)
            except ValidationError as e:
                raise SchemaValidationError(f"Invalid input parameters: {e}") from e
        # convert searxng input schema to code search input schema internally
        elif isinstance(params, SearchToolInputSchema):
            if self.debug:
                logger.warning(
                    f"Converting SearxNGSearchToolInputSchema to {self.input_schema.__name__}",
                )
            params = self.input_schema(**params.model_dump())
        else:
            raise TypeError(
                f"params must be an instance of {self.input_schema.__name__}",
            )
        return params

    def _validate_output(
        self,
        output: CodeSearchToolOutputSchema | SearchToolOutputSchema,
    ) -> CodeSearchToolOutputSchema:
        """Validate output against schema."""

        if isinstance(output, self.output_schema):
            return output
        if isinstance(output, SearchToolOutputSchema):
            if self.debug:
                logger.warning(
                    f"Converting SearchToolOutputSchema to {self.output_schema.__name__}",
                )
            output = self.output_schema(**output.model_dump())
        if not isinstance(output, self.output_schema):
            raise TypeError(
                f"Output must be an instance of {self.output_schema.__name__}",
            )
        return output

    def _deduplicate_results(
        self,
        results: list[SearchResultItem],
        key: str = "url",
    ) -> list[SearchResultItem]:
        """
        Deduplicate results based on a unique key (default is URL).
        """
        seen = set()
        deduped = []
        for result in results:
            val = str(getattr(result, key, ""))
            if val and val not in seen:
                seen.add(val)
                deduped.append(result)
        return deduped

    def _sort_results(
        self,
        results: list[SearchResultItem],
        sort_by: str = "score",
    ) -> list[SearchResultItem]:
        """
        Sort results by the specified key. First checks for the key directly in the dict,
        then checks in the 'extra' field if it exists. Returns unsorted if key not found.
        """

        def __get_sort_key(result):
            # First check if sort_by key exists directly in the dict
            if sort_by in result:
                return result[sort_by]

            # Then check if 'extra' field exists and contains the sort_by key
            if (
                result.extra
                and isinstance(result.extra, dict)
                and sort_by in result.extra
            ):
                return result.extra[sort_by]

            # If key not found anywhere, return a default value that will sort last
            # Using float('inf') for numerical sorting or empty string for string sorting
            return float("-inf")

        try:
            # Sort in descending order (highest score first)
            # Change reverse=False if you want ascending order
            return sorted(results, key=__get_sort_key, reverse=True)
        except TypeError:
            # If sorting fails (mixed types), return as is
            return results


class CombinedCodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the combined code search tool.
    """

    reranker_model_name: str = Field(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        description="The model to use for reranking the combined results.",
    )


class CombinedCodeSearchTool(CodeSearchTool):
    """
    Tool for performing combined code search using multiple sub-tools and CrossEncoder reranking.
    """

    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = CombinedCodeSearchToolConfig

    def __init__(
        self,
        config: CombinedCodeSearchToolConfig | None = None,
        tools: Optional[list[CodeSearchTool]] = None,
        debug: bool = False,
    ):
        super().__init__(config, debug)
        self.tools = tools or [
            LocalRepoCodeSearchTool(debug=debug),
            GitHubCodeSearchTool(debug=debug),
            SDECodeSearchTool(debug=debug),
        ]
        self.reranker_model = CrossEncoder(config.reranker_model_name)

    async def _arun(
        self,
        params: CodeSearchToolInputSchema,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """Run the combined code search tool and aggregate results from all tools."""
        all_results: list[SearchResultItem] = []

        for tool in self.tools:
            try:
                if self.debug:
                    logger.debug(f"Running tool: {tool.__class__.__name__}")
                result = await tool._arun(params)
                for res in result.results:
                    res.extra["tool"] = tool.__class__.__name__
                all_results.extend(result.results)
            except Exception as e:
                logger.error(f"Error running tool {tool.__class__.__name__}: {e}")

        reranked = self._rerank_results(all_results, params.queries[0])[: params.top_k]
        return self.output_schema(results=reranked, category="technology")

    def _rerank_results(
        self, results: list[SearchResultItem], query: str
    ) -> list[SearchResultItem]:
        """
        Rerank results using a CrossEncoder model based on the query and result content.
        """

        try:
            # Generate (query, content) pairs
            pairs = [(query, result.content) for result in results]

            # Get similarity scores from CrossEncoder
            scores = self.reranker_model.predict(pairs)

            # Attach scores
            for score, result in zip(scores, results):
                result.extra["score"] = score

            # Sort results by score
            deduped = self._deduplicate_results(results, key="url")
            return self._sort_results(deduped, sort_by="score")
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results


class LocalRepoCodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the local repository code search tool.
    """

    data_file: str = str(
        get_akd_root() / "docs" / "repositories_with_embeddings_v2.csv"
    )
    google_drive_file_id: str = os.getenv(
        "CODE_SEARCH_FILE_ID",
        "1H15aL-y6DsQl-bt08cU0FaZf5NuwUzmD",
    )
    embedder_type: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    wait_time: int = 1
    embedding_model_name: str = os.getenv("CODE_SEARCH_MODEL", "all-MiniLM-L6-v2")
    remove_embedding_column: bool = True
    text_column: str = "text"
    desc_column: str = "description"
    embeddings_column: str = "embeddings"
    debug: bool = False


class LocalRepoCodeSearchTool(CodeSearchTool):
    """
    Tool for performing semantic code search.
    It automatically downloads the necessary data file if it's not found locally.
    """

    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = LocalRepoCodeSearchToolConfig

    def __init__(
        self,
        config: LocalRepoCodeSearchToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initializes the tool. If the data file is not found, it will be
        downloaded from Google Drive before loading the models.
        """

        config = config or self.config_schema()
        super().__init__(config, debug)

        try:
            logger.info("Initializing CodeSearchTool...")
            self._ensure_data_file_exists()  # Check for and download the data file

            logger.info("Loading data and embedding model...")
            self.repo_data = pd.read_csv(self.config.data_file)
            if self.config.embedder_type == "sentence-transformers":
                self.embedder = Embedder(self.config.embedding_model_name)
            elif self.config.embedder_type == "openai":
                self.embedder = OpenAIEmbedder(
                    model_name=self.config.embedding_model_name
                )

            if self.config.embeddings_column not in self.repo_data.columns:
                logger.warning(
                    f"No embeddings found in column '{self.config.embeddings_column}'. Generating them now..."
                )
                self.generate_embeddings(
                    force_regenerate=False,
                    batch_size=32,
                )

            # Parse embeddings if they are in string format
            if self.debug:
                logger.debug(
                    f"Embeddings column dtype: {self.repo_data[self.config.embeddings_column].dtype}"
                )
            if self.repo_data[self.config.embeddings_column].dtype == "object":
                self.repo_data[self.config.embeddings_column] = self.repo_data[
                    self.config.embeddings_column
                ].apply(
                    self.embedder._parse_embedding,
                )

            # Stack all embeddings into a matrix
            if self.debug:
                logger.debug(
                    f"Embeddings column: {self.repo_data[self.config.embeddings_column].head()}"
                )
            self.embeddings_matrix = np.vstack(
                self.repo_data[self.config.embeddings_column].tolist(),
            )
            if self.debug:
                logger.debug(
                    f"Embeddings matrix shape: {self.embeddings_matrix.shape}",
                )

            logger.info("CodeSearchTool initialization complete.")
        except Exception as e:
            logger.error(f"Error during CodeSearchTool initialization: {e}")

    def _ensure_data_file_exists(self):
        """
        Checks if the data file exists and downloads it if it does not.
        """

        data_file_path = self.config.data_file
        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found at '{data_file_path}'. Downloading...")

            # Ensure the target directory exists
            data_dir = os.path.dirname(data_file_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)

            # Download from Google Drive
            file_id = self.config.google_drive_file_id
            google_drive_downloader(file_id, data_file_path, quiet=False)
        else:
            logger.info(f"Data file already exists at '{data_file_path}'.")

    def generate_embeddings(
        self,
        force_regenerate: bool = False,
        batch_size: int = 32,
    ) -> None:
        """
        Generate embeddings for a given text column if not already present.

        Args:
            text_column: Name of the column containing text to embed
            embeddings_column: Name of the column to store embeddings
            force_regenerate: If True, regenerate embeddings even if column exists
            batch_size: Size of batches for embedding generation
        """
        # Check if embeddings already exist
        if (
            self.config.embeddings_column in self.repo_data.columns
            and not force_regenerate
        ):
            logger.info(
                f"Embeddings column '{self.config.embeddings_column}' already exists. Skipping generation."
            )
            return

        logger.info(
            f"Generating embeddings for {len(self.repo_data)} texts in batches of {batch_size} using {self.config.embedder_type}..."
        )

        # Get texts to embed
        texts = (
            (
                self.repo_data[self.config.desc_column].fillna("")
                + " "
                + self.repo_data[self.config.text_column].fillna("")
            )
            .astype(str)
            .tolist()
        )

        # Process in batches
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_index = i // batch_size + 1
            logger.debug(f"Processing batch {batch_index}/{total_batches}")

            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedder.embed_texts(
                batch_texts,
                batch_size=batch_size,
            )
            embeddings.extend(batch_embeddings)

            if self.config.embedder_type == "openai":
                # Wait for 1 second between batches to avoid rate limit
                time.sleep(self.config.wait_time)

        # Store embeddings in memory
        self.repo_data[self.config.embeddings_column] = embeddings
        logger.info("Embeddings generation completed.")

        # Prepare data for saving
        save_data = self.repo_data.copy()

        # Convert numpy arrays to string representation for CSV storage
        save_data[self.config.embeddings_column] = save_data[
            self.config.embeddings_column
        ].apply(
            lambda x: ",".join(map(str, x)) if isinstance(x, np.ndarray) else x,
        )

        # Persist to disk
        save_data.to_csv(self.config.data_file, index=False)
        logger.info(f"Saved updated data with embeddings to {self.config.data_file}")
        self.repo_data = save_data

    def find_repo(
        self,
        query: str,
        top_k: int = 25,
        remove_embedding_column: bool = True,
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
            raise ValueError("No data loaded. Check if the data file exists.")

        # Get query embedding
        query_embedding = self.embedder.embed_texts([query])
        if self.debug:
            logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Compute cosine distances using cdist (more efficient)
        # cdist with 'cosine' gives cosine distance (1 - cosine_similarity)
        cosine_distances = cdist(
            query_embedding.reshape(1, -1),
            self.embeddings_matrix,
            metric="cosine",
        )[0]  # Extract the single row

        # Convert cosine distances to similarity scores (0-1 range)
        similarities = np.clip(1 - cosine_distances, 0, 1)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = self.repo_data.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]
        results["score"] = results["score"].astype(float)

        if remove_embedding_column:
            results = results.drop(columns=[self.config.embeddings_column])

        return results.reset_index(drop=True).to_dict("records")

    async def _arun(
        self,
        params: CodeSearchToolInputSchema,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """
        Runs the in-memory code search for a list of queries.
        """

        all_results_data = []
        for query in params.queries:
            if self.debug:
                logger.debug(
                    f"Searching for query: '{query}' with top_k={params.top_k}",
                )

            try:
                results = self.find_repo(
                    query=query,
                    top_k=params.top_k,
                    remove_embedding_column=self.config.remove_embedding_column,
                )
                if results:
                    all_results_data.extend(results)
            except Exception as e:
                logger.error(f"Error during search for query '{query}': {e}")

        formatted_results = [
            SearchResultItem(
                title=f"GitHub Repository for {query}",
                url=HttpUrlAdapter.validate_python(result.pop("URL", "")),
                content=result.pop("text", ""),
                query=query,
                extra=result,
            )
            for result in all_results_data
        ]
        try:
            deduped = self._deduplicate_results(formatted_results, key="url")
        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")

        try:
            sorted_results = self._sort_results(
                deduped,
                sort_by="score",
            )
        except Exception as e:
            logger.error(f"Error sorting repo list by score: {e}")

        return self.output_schema(results=sorted_results, category="technology")


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

        base_url = base_url or HttpUrlAdapter.validate_python(
            os.getenv("SEARXNG_BASE_URL", "http://localhost:8080"),
        )
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
        params: CodeSearchToolInputSchema,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
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
                f"and engines to {self.config.engines}",
            )

        # Call the parent's _arun method with the modified parameters
        output = await super()._arun(params=params, max_results=max_results, **kwargs)

        try:
            deduped = self._deduplicate_results(output.results, key="url")
        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")

        try:
            sorted_results = self._sort_results(deduped, sort_by="score")
        except Exception as e:
            logger.error(f"Error sorting repo list by score: {e}")

        return self.output_schema(results=sorted_results, category="technology")


class SDECodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the SDE code search tool.
    """

    base_url: str = os.getenv(
        "SDE_BASE_URL", "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search"
    )
    page_size: int = 10
    max_pages: int = 1
    headers: dict = Field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        description="Headers for the SDE API",
    )
    debug: bool = False


class SDECodeSearchTool(CodeSearchTool):
    """
    Tool for code search using SDE API.
    """

    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = SDECodeSearchToolConfig

    @retry(stop=stop_after_attempt(2))
    def sde_search(self, page: int, query: str):
        """
        Search for code using SDE REST API.
        """

        payload = {
            "filters": {},
            "page": page,
            "pageSize": self.page_size,
            "search_term": query,
        }
        if self.debug:
            logger.debug(f"Payload: {payload}")
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(payload)
        )
        if self.debug:
            logger.debug(f"Response: {response.json()}")
        return response.json()["documents"]

    async def _arun(
        self,
        params: CodeSearchToolInputSchema,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """
        Run the SDE code search tool.
        """

        all_results_data = []
        for query in params.queries:
            if self.debug:
                logger.debug(
                    f"Searching for query: '{query}' with top_k={params.max_results}"
                )

            try:
                for page in range(self.max_pages):
                    try:
                        results = self.sde_search(page=page, query=query)
                        if results:
                            all_results_data.extend(results)
                        else:
                            break
                    except Exception as e:
                        logger.error(
                            f"Error during search for query '{query}' on page {page}: {e}"
                        )
                        continue  # continue to the next page
                all_results_data = all_results_data[: params.max_results]
            except Exception as e:
                logger.error(f"Error during search for query '{query}': {e}")

        formatted_results = [
            SearchResultItem(
                title=f"SDE Code Search for {query}",
                url=HttpUrlAdapter.validate_python(result.pop("url", "")),
                content=result.pop("full_text", ""),
                query=query,
                extra=result,
            )
            for result in all_results_data
        ]
        try:
            deduped = self._deduplicate_results(formatted_results, key="url")
        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")

        try:
            sorted_results = self._sort_results(
                deduped,
                sort_by="score",
            )
        except Exception as e:
            logger.error(f"Error sorting repo list by score: {e}")
        return self.output_schema(results=sorted_results, category="technology")
