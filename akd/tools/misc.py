from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from pathlib import Path
from pydantic import HttpUrl, TypeAdapter

HttpUrlAdapter = TypeAdapter(HttpUrl)

class Embedder:
    """Base class for embedding models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        trust_remote_code: bool = False,
        model_max_seq_length: int | None = None,
        debug: bool = False,
    ):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model_original_max_seq_length = self.model.max_seq_length
        logger.info(
            f"Model original max sequence length: {self.model.max_seq_length} tokens",
        )
        if model_max_seq_length is not None:
            logger.info(
                f"Setting model max sequence length to {model_max_seq_length} tokens ",
            )
            self.model.max_seq_length = model_max_seq_length
        logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        self.debug = bool(debug)

    @property
    def model_max_seq_length(self) -> int:
        """Get the maximum sequence length of the model."""
        return self.model.max_seq_length

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get embeddings using sentence-transformers."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            convert_to_tensor=False,
            batch_size=batch_size,
        )

    def get_embedding_dimensions(self) -> int:
        """Get the dimensions of the embeddings."""
        assert self.model is not None, "Model is not loaded"
        if not hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.embed_texts(
                ["test"],
            ).shape[1]
        # Use the model's method to get embedding dimensions
        return self.model.get_sentence_embedding_dimension()

    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensions of the embeddings."""
        return self.get_embedding_dimensions()

    def compute_similarity(self, query: str, text: str) -> float:
        """Compute similarity between query and text."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _parse_embedding(self, emb_str) -> np.ndarray:
        """Parse embedding from string format to numpy array."""
        if pd.isna(emb_str):
            return np.zeros(self.embedding_dimensions)
        try:
            return np.array([float(x) for x in str(emb_str).split(",")])
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse embedding: {emb_str}")
            return np.zeros(self.embedding_dim)
    
