"""
Retrieval engine for the ArchAIaGPT project.
Integrates dual-index searching (FAISS for dense vectors, pickle-cached sparse vectors) 
to provide multimodal artifact discovery with weighted score fusion.
"""

import os
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch

from embeddings.factory import get_encoder
from config import (
    CLIP_MODEL_ID, TEXT_INDEX, IMAGE_INDEX, META_STORE,
    TOP_K, TEXT_WEIGHT, INDEX_DIR
)

@dataclass
class ArtifactResult:
    """
    Represents a single retrieved artifact with its similarity scores and metadata.
    Used for display in the UI and context building for the LLM.
    """
    idx: int
    artifact_id: str
    label: str
    fused_score: float
    text_score: float
    image_score: float
    description: str
    level_1: str = ""
    level_2: str = ""
    level_3: str = ""
    level_4: str = ""
    level_5: str = ""
    period: str = ""
    project: str = ""
    metadata_json: str = "{}"
    num_images: int = 0
    image_cols: list = field(default_factory=list)

class Retriever:
    """
    The main retrieval class that manages FAISS indexes and the artifact metadata store.
    Supports on-the-fly encoder switching and GPU acceleration if available.
    """

    def __init__(
        self,
        text_index_path: str = None,
        image_index_path: str = None,
        meta_path: str = None,
        model_type: str = "clip",
        model_id: str = None,
        device: str = None,
    ):
        self.model_type = model_type.lower()
        self.model_id = model_id or (CLIP_MODEL_ID if self.model_type == "clip" else None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine paths for FAISS indexes based on the selected embedding model
        if text_index_path is None:
            if self.model_type == "clip":
                text_index_path = str(TEXT_INDEX)
            else:
                text_index_path = str(INDEX_DIR / f"text_{self.model_type}.faiss")
        
        if image_index_path is None:
            if self.model_type == "clip":
                image_index_path = str(IMAGE_INDEX)
            else:
                image_index_path = str(INDEX_DIR / f"image_{self.model_type}.faiss")

        meta_path = meta_path or str(META_STORE)

        # Load FAISS indexes (Dense vectors)
        print(f"Loading {self.model_type} text index from {text_index_path}")
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
        else:
            print(f"Warning: Text index not found at {text_index_path}")
            self.text_index = None

        print(f"Loading {self.model_type} image index from {image_index_path}")
        if os.path.exists(image_index_path):
            self.image_index = faiss.read_index(image_index_path)
        else:
            print(f"Warning: Image index not found at {image_index_path}")
            self.image_index = None
            
        # Load Sparse indexes (if .pkl caches exist)
        self.sparse_text_embs = None
        self.sparse_image_embs = None
        
        text_pkl = text_index_path.replace(".faiss", ".pkl")
        if os.path.exists(text_pkl):
            print(f"Loading sparse text cache: {text_pkl}")
            with open(text_pkl, "rb") as f:
                self.sparse_text_embs = pickle.load(f)
        
        image_pkl = image_index_path.replace(".faiss", ".pkl")
        if os.path.exists(image_pkl):
            print(f"Loading sparse image cache: {image_pkl}")
            with open(image_pkl, "rb") as f:
                self.sparse_image_embs = pickle.load(f)

        # Determine total artifact count
        if self.text_index:
            self.n = self.text_index.ntotal
        elif self.sparse_text_embs:
            self.n = len(self.sparse_text_embs)
        else:
            self.n = 0

        print(f"Retriever initialised with {self.n} artifacts.")

        # Attempt to move FAISS indexes to GPU for faster similarity search
        if self.device.startswith("cuda") and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                print(f"Moving FAISS indexes to {self.device}...")
                self.gpu_resource = faiss.StandardGpuResources()
                if self.text_index:
                    self.text_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.text_index)
                if self.image_index:
                    self.image_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.image_index)
            except Exception as e:
                print(f"Failed to move FAISS to GPU ({e}). Using CPU only.")

        # Load metadata mapping (artifact descriptions, etc.)
        self.metadata: List[dict] = []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                for line in f:
                    content = line.strip()
                    if content:
                        self.metadata.append(json.loads(content))
            print(f"Loaded metadata for {len(self.metadata)} artifacts.")
        
        # Data integrity check
        if self.text_index and len(self.metadata) != self.n:
             print(f"Warning: Metadata row count ({len(self.metadata)}) mismatches index size ({self.n}).")

        # Initialise the encoder model for query processing
        self.encoder = get_encoder(self.model_type, model_id=self.model_id, device=self.device)

    def retrieve(
        self,
        query: Optional[str] = None,
        image_query: Optional[object] = None,
        top_k: int = None,
        text_weight: float = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[ArtifactResult]:
        """
        Main retrieval method for processing multimodal search requests.
        Encodes inputs and performs similarity search across loaded indexes.
        """
        top_k = top_k or TOP_K
        text_weight = text_weight if text_weight is not None else TEXT_WEIGHT
        img_weight = 1.0 - text_weight

        # Step 1: Encode user inputs into the vector space
        q_text = None
        if query and query.strip():
            q_text = self.encoder.encode_query(query.strip())

        q_img = None
        if image_query is not None:
            from PIL import Image as PILImage
            if isinstance(image_query, PILImage.Image):
                q_img = self.encoder.encode_image_query(image_query)
            else:
                q_img = image_query # Handle pre-encoded vectors

        # Step 2: Search dense or sparse indexes
        candidate_count = min(top_k * 5, self.n)

        def perform_search(q_vec, index, sparse_embs):
            if q_vec is None or self.n == 0:
                return np.array([]), np.array([])
            
            # Use dense FAISS index if available
            if hasattr(q_vec, "reshape"):
                q_dense = q_vec.reshape(1, -1).astype(np.float32)
                if index is not None:
                    scores, indices = index.search(q_dense, candidate_count)
                    return scores[0], indices[0]
                return np.array([]), np.array([])
            
            # Fallback to manual sparse similarity if cached
            elif sparse_embs is not None:
                q_inds = q_vec.indices
                q_vals = q_vec.values
                
                similarity_scores = []
                for doc_idx, doc_emb in enumerate(sparse_embs):
                    if doc_emb is None:
                         similarity_scores.append((-1e9, doc_idx))
                         continue
                    
                    doc_map = dict(zip(doc_emb.indices, doc_emb.values))
                    dot_product = sum(v * doc_map[i] for i, v in zip(q_inds, q_vals) if i in doc_map)
                    similarity_scores.append((dot_product, doc_idx))
                
                similarity_scores.sort(key=lambda x: x[0], reverse=True)
                top_results = similarity_scores[:candidate_count]
                return np.array([x[0] for x in top_results]), np.array([x[1] for x in top_results])
            
            return np.array([]), np.array([])

        text_scores, text_ids = perform_search(q_text, self.text_index, self.sparse_text_embs)
        image_scores, image_ids = perform_search(q_img, self.image_index, self.sparse_image_embs)

        # Step 3: Accumulate and fuse scores from both modalities
        score_cache: Dict[int, List[float]] = {} # Maps artifact_idx -> [text_score, image_score]

        for idx, score in zip(text_ids, text_scores):
            if idx >= 0:
                score_cache.setdefault(int(idx), [0.0, 0.0])[0] = float(score)

        for idx, score in zip(image_ids, image_scores):
            if idx >= 0:
                score_cache.setdefault(int(idx), [0.0, 0.0])[1] = float(score)

        # Step 4: Missing score reconstruction for cross-modal consistency
        if self.text_index and self.image_index:
            # Create a unified query representation for reconstruction
            if q_text is not None and q_img is not None and hasattr(q_text, "flatten") and hasattr(q_img, "flatten"):
                q_unified = (q_text + q_img) / 2.0
                norm_val = np.linalg.norm(q_unified)
                if norm_val > 0: q_unified /= norm_val
            elif q_text is not None and hasattr(q_text, "flatten"):
                q_unified = q_text
            elif q_img is not None and hasattr(q_img, "flatten"):
                q_unified = q_img
            else:
                q_unified = None

            if q_unified is not None:
                q_flat = q_unified.flatten()
                for idx, scores in score_cache.items():
                    # If an item was found in one index but not the other, try to reconstruct its vector
                    if scores[0] == 0.0 and text_weight > 0:
                        try:
                            vec = self.text_index.reconstruct(int(idx))
                            scores[0] = float(np.dot(q_flat, vec))
                        except Exception: pass
                    if scores[1] == 0.0 and img_weight > 0:
                        try:
                            vec = self.image_index.reconstruct(int(idx))
                            scores[1] = float(np.dot(q_flat, vec))
                        except Exception: pass

        # Step 5: Final weighted sorting
        candidates = []
        for idx, (t_s, i_s) in score_cache.items():
            fused = text_weight * t_s + img_weight * i_s
            candidates.append((idx, fused, t_s, i_s))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Metadata Filtering (Pre-filtering at retrieval bottleneck)
        if filters:
            refined_list = []
            for idx, fused, t_s, i_s in candidates:
                meta = self.metadata[idx]
                is_match = True
                for key, val in filters.items():
                    if meta.get(key, "").lower() != val.lower():
                        is_match = False
                        break
                if is_match:
                    refined_list.append((idx, fused, t_s, i_s))
            candidates = refined_list

        # Step 7: Object instantiation
        final_results = []
        for idx, fused, t_s, i_s in candidates[:top_k]:
            meta = self.metadata[idx]
            final_results.append(ArtifactResult(
                idx=idx,
                artifact_id=meta.get("artifact_id", ""),
                label=meta.get("label", ""),
                fused_score=round(fused, 4),
                text_score=round(t_s, 4),
                image_score=round(i_s, 4),
                description=meta.get("description", ""),
                level_1=meta.get("level_1", ""),
                level_2=meta.get("level_2", ""),
                level_3=meta.get("level_3", ""),
                level_4=meta.get("level_4", ""),
                level_5=meta.get("level_5", ""),
                period=meta.get("period", ""),
                project=meta.get("project", ""),
                metadata_json=meta.get("metadata_json", "{}"),
                num_images=meta.get("num_images", 0),
                image_cols=meta.get("image_cols", []),
            ))

        return final_results
