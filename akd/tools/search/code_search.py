from __future__ import annotations

import json
import os
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger
from pydantic import Field, ValidationError, computed_field
from scipy.spatial.distance import cdist
from tenacity import retry, stop_after_attempt

from akd._base.errors import SchemaValidationError
from akd.structures import SearchResultItem
from akd.tools.misc import Embedder, HttpUrlAdapter, OpenAIEmbedder
from akd.tools.reranker import RerankerToolConfig, RerankerType
from akd.utils import get_akd_root, google_drive_downloader

from ._base import (
    SearchTool,
    SearchToolConfig,
    SearchToolInputSchema,
    SearchToolOutputSchema,
)
from .composite import CompositeSearchTool, CompositeSearchToolConfig
from .searxng import SearxNGSearchTool, SearxNGSearchToolConfig


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


class CodeSearchToolConfig(SearchToolConfig):
    """Configuration for the code search tool."""

    # only use "url" for code search
    rrf_keys: list[str] = Field(
        default_factory=lambda: ["url"],
        description=(
            "List of attribute names for RRF deduplication (cascaded OR logic). "
            "Matches if ANY key matches. Priority order: doi > title > url."
        ),
    )

    # disabled for code search
    result_normalization: bool = Field(
        default=False,
        description=(
            "Enable automatic normalization of results after each query. "
            "Results are enriched with DOI resolution, URL normalization, and metadata. "
            "Uses CompositeResolver with default chain if no custom resolver provided."
        ),
    )

    deduplication_keys: list[str] = Field(
        default_factory=lambda: ["url"],  # default to url for code search
    )


class CodeSearchTool(SearchTool):
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
            """
            Gets the sorting key from the result object, checking the direct
            attribute first, then the 'extra' dictionary.
            """

            # 1. Try to get the attribute directly from the object.
            # We use a default of `None` to distinguish "doesn't exist"
            # from a valid "falsy" value like 0, False, or [].
            if (value := getattr(result, sort_by, None)) is not None:
                return value

            # 2. If not found (or was None), check the 'extra' attribute.
            # Safely get 'extra', defaulting to an empty dict if it's None or missing.
            extra = getattr(result, "extra", None)

            # 3. If 'extra' is a dict, try to .get() the key.
            # .get() safely returns None if the key doesn't exist.
            if isinstance(extra, dict):
                if (value := extra.get(sort_by)) is not None:
                    return value

            # 4. If not found in either place, return the default sorting value.
            return float("-inf")

        try:
            # Sort in descending order (highest score first)
            # Change reverse=False if you want ascending order
            return sorted(results, key=__get_sort_key, reverse=True)
        except TypeError:
            # If sorting fails (mixed types), return as is
            return results

    async def _arun_single_query(self, *args, **kwargs) -> CodeSearchToolOutputSchema:
        raise NotImplementedError()


class CompositeCodeSearchToolConfig(CompositeSearchToolConfig):
    """
    Configuration for the combined code search tool.

    Inherits fusion_strategy from CompositeSearchToolConfig.
    Default uses flatten_rerank_rrf with cross-encoder for optimal semantic ranking.
    """

    # Override defaults for code search use case
    fusion_strategy: Literal["direct_rrf_blackbox", "flatten_rerank_rrf"] = Field(
        default="flatten_rerank_rrf",
        description="Fusion strategy for combining code search results from multiple tools.",
    )
    reranker_type: RerankerType = Field(
        default="cross_encoder",
        description="Type of reranker to use for combining results from multiple search tools.",
    )
    reranker_config: RerankerToolConfig | None = Field(
        default_factory=lambda: RerankerToolConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L12-v2",
        ),
        description="Configuration for the reranker tool.",
    )


class CompositeCodeSearchTool(CodeSearchTool, CompositeSearchTool):
    """
    Tool for performing combined code search using multiple sub-tools.

    Combines results from LocalRepo, GitHub, and SDE code search tools using
    configurable fusion strategies (direct RRF or flatten+rerank+RRF).
    """

    input_schema = CodeSearchToolInputSchema
    output_schema = CodeSearchToolOutputSchema
    config_schema = CompositeCodeSearchToolConfig

    def __init__(
        self,
        config: CompositeCodeSearchToolConfig | None = None,
        tools: Optional[list[CodeSearchTool]] = None,
        debug: bool = False,
    ):
        """
        Initialize combined code search tool.

        Args:
            config: Configuration for the tool.
            tools: Optional list of search tools to combine. Defaults to LocalRepo, GitHub, and SDE.
            debug: Enable debug logging.
        """
        # Initialize default tools if not provided
        search_tools = tools or [
            LocalRepoCodeSearchTool(debug=debug),
            GitHubCodeSearchTool(debug=debug),
            SDECodeSearchTool(debug=debug),
        ]

        # Initialize composite search tool with all tools
        super().__init__(*search_tools, config=config, debug=debug)


class LocalRepoCodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the local repository code search tool.
    """

    data_file: str = str(
        get_akd_root() / "docs" / os.getenv("REPO_EMBEDDINGS_FILE", "repositories_with_embeddings_v6.csv"),
    )
    google_drive_file_id: str = os.getenv(
        "CODE_SEARCH_FILE_ID",
        "1XwH4N-HJeak4Pfp6r0Nhdz0d5tQD99jE",
    )
    embedder_type: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    wait_time: int = 1
    embedding_model_name: str = os.getenv("CODE_SEARCH_MODEL", "thenlper/gte-large")
    remove_embedding_column: bool = True
    context_columns: list[str] = ["description", "reformulated_text", "key_topics", "relevant_content"]
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

            missing = [c for c in self.config.context_columns if c not in self.repo_data.columns]
            if missing:
                raise ValueError(
                    f"Missing columns in {self.config.data_file}: {missing}. "
                    f"Available columns: {list(self.repo_data.columns)}",
                )

            if self.config.embedder_type == "sentence-transformers":
                self.embedder = Embedder(self.config.embedding_model_name)
            elif self.config.embedder_type == "openai":
                self.embedder = OpenAIEmbedder(model_name=self.config.embedding_model_name)

            if self.config.embeddings_column not in self.repo_data.columns:
                logger.warning(
                    f"No embeddings found in column '{self.config.embeddings_column}'. Generating them now...",
                )
                self.generate_embeddings(
                    force_regenerate=False,
                    batch_size=32,
                )

            # Parse embeddings if they are in string format
            if self.debug:
                logger.debug(f"Embeddings column dtype: {self.repo_data[self.config.embeddings_column].dtype}")
            if self.repo_data[self.config.embeddings_column].dtype == "object":
                self.repo_data[self.config.embeddings_column] = self.repo_data[self.config.embeddings_column].apply(
                    self.embedder._parse_embedding,
                )

            # Stack all embeddings into a matrix
            if self.debug:
                logger.debug(f"Embeddings column: {self.repo_data[self.config.embeddings_column].head()}")
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

    def _stringify_columns(self, columns: list[str]) -> list[str]:
        """
        Concatenate the given columns row-wise and return a list of embedding texts
        """

        # Convert selected columns to strings with empty strings for NaN
        df_str = (
            self.repo_data[columns]
            .applymap(
                lambda v: " ".join(map(str, v)) if isinstance(v, (list, tuple)) else ("" if v is None else str(v)),
            )
            .fillna("")
        )

        # Concatenate across columns for each row
        texts = df_str.apply(
            lambda row: " ".join(part for part in row if part),
            axis=1,
        ).tolist()

        if self.debug:
            logger.debug(f"Built {len(texts)} embedding texts; sample[0:2]={texts[:2]}")

        return texts

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
        if self.config.embeddings_column in self.repo_data.columns and not force_regenerate:
            logger.info(f"Embeddings column '{self.config.embeddings_column}' already exists. Skipping generation.")
            return

        logger.info(
            f"Generating embeddings for {len(self.repo_data)} texts in batches of {batch_size} using {self.config.embedder_type}...",
        )

        # Get texts to embed
        texts = self._stringify_columns(self.config.context_columns)

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
        save_data[self.config.embeddings_column] = save_data[self.config.embeddings_column].apply(
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

    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """
        Runs the in-memory code search for a list of queries.
        """

        all_results_data = []
        if self.debug:
            logger.debug(
                f"Searching for query: '{query}' with top_k={max_results}",
            )

        try:
            results = self.find_repo(
                query=query,
                top_k=max_results,
                remove_embedding_column=self.config.remove_embedding_column,
            )
            if results:
                for result in results:
                    result["query"] = query
                all_results_data.extend(results)
        except Exception as e:
            logger.error(f"Error during search for query '{query}': {e}")

        formatted_results = [
            SearchResultItem(
                title=str(result.pop("name", "")),
                url=HttpUrlAdapter.validate_python(result.pop("URL", "")),
                content=result.pop("text", ""),
                query=result.pop("query", ""),
                extra=result,
            )  # type: ignore
            for result in all_results_data
        ]

        sorted_results = self._sort_results(
            formatted_results,
            sort_by="score",
        )

        return self.output_schema(results=sorted_results)


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
        config.description = "Tool for performing targeted searches on GitHub for code, repositories, and issues."

        super().__init__(config, debug)

    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """
        Fetch search results for a single query from GitHub via SearxNG.

        This implements the abstract method from SearchTool base class.
        It forces category to 'technology', delegates to SearxNG, and applies post-processing.

        Args:
            query: The search query string.
            max_results: Maximum number of results to fetch for this query.
            **kwargs: Additional parameters.

        Returns:
            CodeSearchToolOutputSchema with deduplicated and sorted results.
        """
        # Force category to 'technology' for GitHub searches
        kwargs["category"] = "technology"

        if self.debug:
            logger.debug(
                f"GitHubSearchTool: Searching for '{query}' with category=technology",
            )

        # Call SearxNGSearchTool's _arun_single_query
        output = await SearxNGSearchTool._arun_single_query(self, query, max_results, **kwargs)

        sorted_results = self._sort_results(output.results, sort_by="score")

        # Convert output to CodeSearchToolOutputSchema
        return self.output_schema(results=sorted_results, extra=output.extra or {})


class SDECodeSearchToolConfig(CodeSearchToolConfig):
    """
    Configuration for the SDE code search tool.
    """

    base_url: str = os.getenv("SDE_BASE_URL", "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search")
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
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"


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
            "page": page,
            "pageSize": self.page_size,
            "search_term": query,
            "search_type": self.search_mode,
        }
        if self.debug:
            logger.debug(f"Payload: {payload}")
        response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
        if self.debug:
            logger.debug(f"Response: {response.json()}")
        return response.json()["documents"]

    async def _arun_single_query(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> CodeSearchToolOutputSchema:
        """
        Run the SDE code search tool.
        """

        all_results_data = []
        query_results = []
        if self.debug:
            logger.debug(f"Searching for query: '{query}' with top_k={max_results}")

        try:
            for page in range(1, self.config.max_pages + 1):
                try:
                    results = self.sde_search(page=page, query=query)
                    if results:
                        for result in results:
                            result["query"] = query
                        query_results.extend(results)
                    else:
                        break
                except Exception as e:
                    logger.error(f"Error during search for query '{query}' on page {page}: {e}")
                    continue  # continue to the next page
            all_results_data.extend(query_results[:max_results])
        except Exception as e:
            logger.error(f"Error during search for query '{query}': {e}")

        formatted_results = [
            SearchResultItem(
                title=str(result.get("url", "")).split("/")[-1],
                url=HttpUrlAdapter.validate_python(result.pop("url", "")),
                content=result.pop("full_text", ""),
                query=result.pop("query", ""),
                extra=result,
            )  # type: ignore
            for result in all_results_data
        ]

        sorted_results = self._sort_results(
            formatted_results,
            sort_by="score",
        )
        return self.output_schema(results=sorted_results)
