"""
Factory module for selecting and initialising various embedding models.
Supported encoders include multimodal (CLIP, Qwen3-VL, VLM2Vec, e5-omni)
and text-only models (Gemma, BM25).
"""

from .clip_encoder import CLIPEncoder
from .bm25_encoder import BM25Encoder
from .gemma_encoder import GemmaEncoder
from .qwen3_vl_encoder import Qwen3VLEncoder
from .e5_omni_encoder import E5OmniEncoder
from .vlm2vec_encoder import VLM2VecEncoder
from config import CLIP_MODEL_ID

def get_encoder(model_type: str, model_id: str = None, device: str = None):
    """
    Returns an instance of an artifact encoder based on the model_type.
    Arguments allow overriding the default model_id and compute device.
    """
    model_type = model_type.lower()
    
    if model_type == "clip":
        return CLIPEncoder(model_id=model_id or CLIP_MODEL_ID, device=device)
    elif model_type == "bm25":
        return BM25Encoder(model_name=model_id or "Qdrant/bm25", device=device)
    elif model_type == "gemma":
        return GemmaEncoder(model_name=model_id or "google/embeddinggemma-300m", device=device)
    elif model_type == "qwen3-vl":
        return Qwen3VLEncoder(model_path=model_id or "Qwen/Qwen3-VL-Embedding-2B", device=device)
    elif model_type == "e5-omni":
        return E5OmniEncoder(model_name=model_id or "Haon-Chen/e5-omni-3B", device=device)
    elif model_type == "vlm2vec":
        return VLM2VecEncoder(model_name=model_id or "TIGER-Lab/VLM2Vec-Qwen2VL-2B", device=device)
    else:
        raise ValueError(f"Unknown encoder type specified: {model_type}")
