import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from .base_encoder import BaseEncoder

class GemmaEncoder(BaseEncoder):
    """Google EmbeddingGemma-300M encoder."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = None):
        super().__init__(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = 768

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode documents for retrieval."""
        # Use encode_document if available (according to plan) or fallback to encode
        if hasattr(self.model, "encode_document"):
            return self.model.encode_document(texts, batch_size=batch_size)
        return self.model.encode(texts, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query for retrieval."""
        if hasattr(self.model, "encode_query"):
            return self.model.encode_query(query)
        return self.model.encode([query])[0]

    def encode_images(self, images: List, batch_size: int = 32) -> np.ndarray:
        return None

    def encode_image_query(self, image: object) -> np.ndarray:
        return None
