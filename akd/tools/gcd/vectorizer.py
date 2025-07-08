from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import cdist
from tqdm import tqdm

from akd.tools.gcd.embeddings import Embedder


class RepoFinder:
    """RepoFinder for generating and caching embeddings with similarity search."""  # noqa

    def __init__(
        self,
        embedder: Embedder,
        data: pd.DataFrame | None = None,
        text_column: str = "text",
        embeddings_column: str = "embeddings",
        area_column: str = "area",
        parse_embeddings_from_string: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Initialize RepoFinder.

        Args:
            embedder: EmbeddingScorer instance
            data: DataFrame with text data
            text_column: Name of column containing text to embed
            embeddings_column: Name of column to store embeddings
            parse_embeddings_from_string: Whether to parse string embeddings to numpy arrays # noqa
            debug: Enable debug logging
        """
        self.embedder = embedder
        self.text_column = text_column
        self.embeddings_column = embeddings_column
        self.area_column = area_column.strip()
        self.debug = bool(debug)
        self.data = None

        if data is not None:
            self.load_data(data, parse_embeddings_from_string)

    def load_data(
        self,
        data: pd.DataFrame,
        parse_embeddings_from_string: bool = True,
    ) -> None:
        """
        Load data and validate required columns.

        Args:
            data: DataFrame with text data
            parse_embeddings_from_string:
                Whether to parse string embeddings to numpy arrays
        """
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"  # noqa
        assert self.text_column in data.columns, (
            f"Text column '{self.text_column}' not found"
        )
        data = data.copy()
        self.data = data[data[self.area_column] != "Not a NASA Division"].reset_index(
            drop=True,
        )  # noqa

        if self.embeddings_column not in self.data.columns:
            logger.warning(
                f"Embeddings column '{self.embeddings_column}' not found. "
                "Use generate_embeddings() to create it.",
            )

        # Parse embeddings if they exist and are in string format
        if (
            parse_embeddings_from_string
            and self.embeddings_column in self.data.columns
            and self.data[self.embeddings_column].dtype == "object"
        ):
            self.data[self.embeddings_column] = self.data[
                self.embeddings_column
            ].apply(self._parse_embedding)
            logger.info("Parsed embeddings from string format")

        logger.info(f"Loaded data with {len(self.data)} rows")

    def _parse_embedding(self, emb_str) -> np.ndarray:
        """Parse embedding from string format to numpy array."""
        if pd.isna(emb_str):
            return np.zeros(self.embedding_dim)
        try:
            return np.array([float(x) for x in str(emb_str).split(",")])
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse embedding: {emb_str}")
            return np.zeros(self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensions from the embedder."""
        return self.embedder.get_embedding_dimensions()

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        embedder: Embedder,
        text_column: str = "text",
        embeddings_column: str = "embeddings",
        parse_embeddings_from_string: bool = True,
        debug: bool = False,
    ) -> RepoFinder:
        """
        Create RepoFinder instance from CSV file.

        Args:
            filepath: Path to CSV file
            embedder: EmbeddingScorer instance
            text_column: Name of column containing text to embed
            embeddings_column: Name of column to store embeddings
            parse_embeddings_from_string: Whether to parse string embeddings to numpy arrays # noqa
            debug: Enable debug logging

        Returns:
            Vectorizer instance with loaded data
        """
        filepath = Path(filepath)
        data = pd.read_csv(filepath)

        return cls(
            embedder=embedder,
            data=data,
            text_column=text_column,
            embeddings_column=embeddings_column,
            parse_embeddings_from_string=parse_embeddings_from_string,
            debug=debug,
        )

    def generate_embeddings(
        self,
        force_regenerate: bool = False,
        batch_size: int = 32,
    ) -> None:
        """
        Generate embeddings for text column if not already present.

        Args:
            force_regenerate: If True, regenerate embeddings even if column exists # noqa
            batch_size: Size of batches for embedding generation
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Check if embeddings already exist
        if self.embeddings_column in self.data.columns and not force_regenerate:  # noqa
            logger.info(
                f"Embeddings column '{self.embeddings_column}' "
                "already exists. Skipping generation.",
            )
            return

        logger.info(
            f"Generating embeddings for {len(self.data)} "
            f"texts in batches of {batch_size}...",
        )

        # Get texts to embed
        texts = self.data[self.text_column].fillna("").astype(str).tolist()  # noqa

        # Process in batches to avoid memory issues
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i : i + batch_size]  # noqa
            logger.debug(
                f"Processing batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}",
            )

            # Generate embeddings for batch
            batch_embeddings = self.embedder.embed_texts(
                batch_texts,
                batch_size=batch_size,
            )
            embeddings.extend(batch_embeddings)

        # Store embeddings
        self.data[self.embeddings_column] = embeddings
        logger.info("Embeddings generation completed")

    def save_to_csv(
        self,
        filepath: str | Path,
        include_embeddings: bool = True,
    ) -> None:
        """
        Save data to CSV file.

        Args:
            filepath: Path to save CSV file
            include_embeddings: Whether to include embeddings column in CSV
        """
        if self.data is None:
            raise ValueError("No data to save. Load data first.")

        filepath = Path(filepath)
        save_data = self.data.copy()

        if include_embeddings and self.embeddings_column in save_data.columns:
            # Convert numpy arrays to string representation for CSV storage
            save_data[self.embeddings_column] = save_data[
                self.embeddings_column
            ].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, np.ndarray) else x,  # noqa
            )
        elif (not include_embeddings) and (
            self.embeddings_column in save_data.columns
        ):
            save_data = save_data.drop(columns=[self.embeddings_column])

        save_data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

    def find_repo(
        self,
        query: str,
        top_k: int = 25,
    ) -> list[dict]:
        """
        Perform similarity search against cached embeddings using vectorized computation. # noqa

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of dictionaries with top results and similarity scores
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.embeddings_column not in self.data.columns:
            raise ValueError(
                "No embeddings found. Run generate_embeddings() first.",
            )

        # Get query embedding
        query_embedding = self.embedder.embed_texts([query])
        if self.debug:
            logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Stack all embeddings into a matrix
        embeddings_matrix = np.vstack(
            self.data[self.embeddings_column].tolist(),
        )
        if self.debug:
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

        results = self.data.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]

        # return results.reset_index(drop=True)
        return results.reset_index(drop=True).to_dict("records")

    def get_stats(self) -> dict:
        """Get statistics about the vectorizer state."""
        if self.data is None:
            return {"status": "No data loaded"}

        stats = {
            "num_rows": len(self.data),
            "text_column": self.text_column,
            "has_embeddings": self.embeddings_column in self.data.columns,
        }

        if stats["has_embeddings"]:
            # Check embedding dimensions
            sample_emb = self.data[self.embeddings_column].iloc[0]
            if isinstance(sample_emb, np.ndarray):
                stats["embedding_dim"] = sample_emb.shape[0]

        return stats

    def print_results(
        self,
        results: list[dict],
        show_text: bool = True,
        max_text_length: int = 200,
    ) -> None:
        """
        Print search results in a formatted way.

        Args:
            results: List of dictionaries from find_repo() method
            show_text: Whether to display the text content
            max_text_length: Maximum length of text to display (truncated if longer)
        """
        if not results:
            print("No matching repositories found.")
            return

        print(f"\nTop {len(results)} matching repositories:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Repository Match")

            # Display score
            score = result.get("score", 0)
            print(f" Similarity Score: {score:.3f}")

            # Display area if available
            if self.area_column in result:
                area = result.get(self.area_column, "N/A")
                print(f" Area: {area}")

            # Display other relevant fields (excluding text, embeddings, and score)
            excluded_fields = {
                self.text_column,
                self.embeddings_column,
                "score",
                self.area_column,
            }
            for key, value in result.items():
                if key not in excluded_fields and value is not None:
                    # Handle different data types appropriately
                    if isinstance(value, (int, float, str, bool)):  # noqa
                        print(f" {key.replace('_', ' ').title()}: {value}")

            # Display text content if requested
            if show_text and self.text_column in result:
                text_content = result.get(self.text_column, "")
                if text_content:
                    # Truncate long text
                    if len(text_content) > max_text_length:
                        text_preview = text_content[:max_text_length] + "..."
                    else:
                        text_preview = text_content

                    print(" Text Content:")
                    # Indent the text for better readability
                    text_lines = text_preview.split("\n")
                    for line in text_lines:
                        print(f"   {line}")

            print("-" * 80)
