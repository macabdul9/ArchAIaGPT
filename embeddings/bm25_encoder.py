import numpy as np
from typing import List, Optional
from fastembed import SparseTextEmbedding
from .base_encoder import BaseEncoder

class BM25Encoder(BaseEncoder):
    """Encoder for BM25 sparse embeddings using FastEmbed."""

    def __init__(self, model_name: str = "Qdrant/bm25", device: str = None):
        super().__init__(device)
        self.model = SparseTextEmbedding(model_name=model_name)
        # BM25 embeddings are sparse, so they don't have a fixed dimension like dense ones.
        self.dim = None 

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> List:
        """Returns a list of SparseEmbedding objects."""
        return list(self.model.embed(texts, batch_size=batch_size))

    def encode_query(self, query: str) -> object:
        """Returns a single SparseEmbedding object."""
        return list(self.model.embed([query]))[0]

    def encode_images(self, images: List, batch_size: int = 32) -> np.ndarray:
        """BM25 is text-only."""
        return None

    def encode_image_query(self, image: object) -> np.ndarray:
        """BM25 is text-only."""
        return None
