from typing import List, Optional, Dict, Any

class BaseGenerator:
    """Base class for all generation models."""
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device
        self.model = None

    def generate(self, query: str, context: str, images: List[Any] = None) -> str:
        raise NotImplementedError

    def generate_stream(self, query: str, context: str, images: List[Any] = None):
        raise NotImplementedError
