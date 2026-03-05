"""
Dual-index retriever: query → Top-K artifacts via text + image FAISS search
with weighted score fusion.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from embeddings.clip_encoder import CLIPEncoder
from config import (
    CLIP_MODEL_ID, TEXT_INDEX, IMAGE_INDEX, META_STORE,
    TOP_K, TEXT_WEIGHT, IMAGE_WEIGHT,
)


@dataclass
class ArtifactResult:
    """One retrieved artifact with scores and display data."""
    idx:           int
    artifact_id:   str
    label:         str
    fused_score:   float
    text_score:    float
    image_score:   float
    description:   str            # original catalog description
    level_1:       str = ""
    level_2:       str = ""
    level_3:       str = ""
    level_4:       str = ""
    level_5:       str = ""
    period:        str = ""
    project:       str = ""
    metadata_json: str = "{}"
    num_images:    int = 0
    image_cols:    list = field(default_factory=list)


class Retriever:
    """
    Dual-index retriever.

    At init: loads FAISS text + image indexes and metadata sidecar.
    At query: encodes query with CLIP, searches both indexes, fuses scores.
    """

    def __init__(
        self,
        text_index_path:  str = None,
        image_index_path: str = None,
        meta_path:        str = None,
        clip_model:       str = None,
        device:           str = None,
    ):
        text_index_path  = text_index_path  or str(TEXT_INDEX)
        image_index_path = image_index_path or str(IMAGE_INDEX)
        meta_path        = meta_path        or str(META_STORE)
        clip_model       = clip_model       or CLIP_MODEL_ID

        # ── Load FAISS indexes ────────────────────────────────────────────────
        print(f"[Retriever] Loading text index:  {text_index_path}")
        self.text_index  = faiss.read_index(text_index_path)
        print(f"[Retriever] Loading image index: {image_index_path}")
        self.image_index = faiss.read_index(image_index_path)
        self.n = self.text_index.ntotal
        print(f"[Retriever] Index size: {self.n} artifacts")

        # ── Move to GPU if available ──────────────────────────────────────────
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                print(f"[Retriever] Moving FAISS indexes to {self.device} ...")
                self.res = faiss.StandardGpuResources()
                self.text_index  = faiss.index_cpu_to_gpu(self.res, 0, self.text_index)
                self.image_index = faiss.read_index(image_index_path) # reload for safety then move
                self.image_index = faiss.index_cpu_to_gpu(self.res, 0, self.image_index)
            except Exception as e:
                print(f"[Retriever] Warning: Could not move FAISS to GPU: {e}. Falling back to CPU.")

        # ── Load metadata sidecar ─────────────────────────────────────────────
        print(f"[Retriever] Loading metadata:    {meta_path}")
        self.metadata: List[dict] = []
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        assert len(self.metadata) == self.n, (
            f"Metadata rows ({len(self.metadata)}) ≠ index size ({self.n})"
        )

        # ── CLIP encoder (shared for all queries) ─────────────────────────────
        import torch
        self.encoder = CLIPEncoder(model_id=clip_model, device=self.device)

        print("[Retriever] Ready.\n")

    def retrieve(
        self,
        query:       Optional[str] = None,
        image_query: Optional[np.ndarray] = None,  # PIL image or pre-encoded vec
        top_k:       int   = None,
        text_weight: float = None,
        filters:     Optional[Dict[str, str]] = None,
    ) -> List[ArtifactResult]:
        """
        Retrieve top-K artifacts for a text and/or image query.

        Parameters
        ----------
        query        : optional search text
        image_query  : optional PIL image for visual search
        top_k        : number of results to return
        text_weight  : weight for text vs image index similarity [0-1]
        filters      : optional metadata pre-filter
        """
        top_k       = top_k       or TOP_K
        text_weight = text_weight if text_weight is not None else TEXT_WEIGHT
        img_weight  = 1.0 - text_weight

        # ── Encode query ──────────────────────────────────────────────────────
        q_text = None
        if query and query.strip():
            q_text = self.encoder.encode_query(query.strip())

        q_img = None
        if image_query is not None:
            from PIL import Image as PILImage
            if isinstance(image_query, PILImage.Image):
                q_img = self.encoder.encode_image_query(image_query)
            else:
                q_img = image_query  # assume already encoded

        # Combine embeddings if both provided
        if q_text is not None and q_img is not None:
            q = (q_text + q_img) / 2.0
            norm = np.linalg.norm(q)
            if norm > 0: q /= norm
        elif q_text is not None:
            q = q_text
        elif q_img is not None:
            q = q_img
        else:
            return []  # No query provided

        q = q.reshape(1, -1).astype(np.float32)

        # ── Search both indexes ───────────────────────────────────────────────
        # Retrieve more than top_k to allow for filtering
        search_k = min(top_k * 3, self.n)

        text_scores,  text_ids  = self.text_index.search(q, search_k)   # (1, search_k)
        image_scores, image_ids = self.image_index.search(q, search_k)  # (1, search_k)

        text_scores  = text_scores[0]     # (search_k,)
        text_ids     = text_ids[0]
        image_scores = image_scores[0]
        image_ids    = image_ids[0]

        # ── Build score map: artifact_idx → (text_score, image_score) ─────────
        score_map: Dict[int, List[float]] = {}   # idx → [text_s, image_s]

        for idx, s in zip(text_ids, text_scores):
            if idx < 0:
                continue
            score_map.setdefault(int(idx), [0.0, 0.0])
            score_map[int(idx)][0] = float(s)

        for idx, s in zip(image_ids, image_scores):
            if idx < 0:
                continue
            score_map.setdefault(int(idx), [0.0, 0.0])
            score_map[int(idx)][1] = float(s)

        # ── Fill missing cross-modal scores ───────────────────────────────────
        # If an artifact was found by text but not image (or vice versa),
        # compute the missing score via direct dot product.
        q_flat = q.flatten()
        for idx, scores in score_map.items():
            if scores[0] == 0.0 and text_weight > 0:
                vec = self.text_index.reconstruct(idx)
                scores[0] = float(np.dot(q_flat, vec))
            if scores[1] == 0.0 and img_weight > 0:
                vec = self.image_index.reconstruct(idx)
                scores[1] = float(np.dot(q_flat, vec))

        # ── Fuse scores ──────────────────────────────────────────────────────
        candidates = []
        for idx, (t_s, i_s) in score_map.items():
            fused = text_weight * t_s + img_weight * i_s
            candidates.append((idx, fused, t_s, i_s))

        # Sort by fused score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # ── Apply metadata filters ────────────────────────────────────────────
        if filters:
            filtered = []
            for idx, fused, t_s, i_s in candidates:
                meta = self.metadata[idx]
                match = True
                for key, val in filters.items():
                    if meta.get(key, "").lower() != val.lower():
                        match = False
                        break
                if match:
                    filtered.append((idx, fused, t_s, i_s))
            candidates = filtered

        # ── Build results ─────────────────────────────────────────────────────
        results = []
        for idx, fused, t_s, i_s in candidates[:top_k]:
            meta = self.metadata[idx]
            results.append(ArtifactResult(
                idx           = idx,
                artifact_id   = meta.get("artifact_id", ""),
                label         = meta.get("label", ""),
                fused_score   = round(fused, 4),
                text_score    = round(t_s, 4),
                image_score   = round(i_s, 4),
                description   = meta.get("description", ""),
                level_1       = meta.get("level_1", ""),
                level_2       = meta.get("level_2", ""),
                level_3       = meta.get("level_3", ""),
                level_4       = meta.get("level_4", ""),
                level_5       = meta.get("level_5", ""),
                period        = meta.get("period", ""),
                project       = meta.get("project", ""),
                metadata_json = meta.get("metadata_json", "{}"),
                num_images    = meta.get("num_images", 0),
                image_cols    = meta.get("image_cols", []),
            ))

        return results
