import numpy as np
from typing import List, Optional
from PIL import Image

class BaseEncoder:
    """Base class for all encoders."""
    def __init__(self, device: str = None):
        self.device = device
        self.dim = None

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError
