"""
ArchAIaGPT/embeddings/clip_encoder.py
──────────────────────────────────────
CLIP encoder wrapper — text and image encoding with L2-normalisation.
Uses openai/clip-vit-base-patch32 via HuggingFace transformers.
"""

import torch
import numpy as np
from typing import List
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """Thin wrapper around HF CLIPModel for batched text/image encoding."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIPEncoder] Loading {model_id} on {self.device} …")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.dim = self.model.config.projection_dim  # 512 for ViT-B/32
        print(f"[CLIPEncoder] Ready. Embedding dim = {self.dim}")

    def _to_tensor(self, output):
        """Extract a plain tensor from model output (handles both tensor and dataclass)."""
        if isinstance(output, torch.Tensor):
            return output
        # Some transformers versions return BaseModelOutputWithPooling
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :]
        raise TypeError(f"Unexpected output type: {type(output)}")

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of text strings → (N, dim) L2-normalised float32 array.
        Empty/None strings are encoded as zero vectors.
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace None/empty with a placeholder (will be zero'd out)
            clean = [t if (t and isinstance(t, str) and t.strip()) else " " for t in batch]

            # Process text only — extract just the text-related keys
            inputs = self.processor(text=clean, return_tensors="pt", padding=True, truncation=True, max_length=77)
            text_inputs = {
                "input_ids":      inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
            }

            # get_text_features returns the projected embedding tensor
            embs = self.model.get_text_features(**text_inputs)
            embs = self._to_tensor(embs)
            embs = embs / embs.norm(dim=-1, keepdim=True)   # L2-normalise

            # Zero out placeholders
            for j, t in enumerate(batch):
                if not (t and isinstance(t, str) and t.strip()):
                    embs[j] = 0.0

            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of PIL images → (N, dim) L2-normalised float32 array.
        None entries are encoded as zero vectors.
        """
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            valid_idxs = [j for j, img in enumerate(batch) if img is not None]
            valid_imgs = [batch[j] for j in valid_idxs]

            chunk_embs = np.zeros((len(batch), self.dim), dtype=np.float32)

            if valid_imgs:
                # Ensure all images are RGB
                valid_imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in valid_imgs]

                # Process images only — extract just pixel_values
                inputs = self.processor(images=valid_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                embs = self.model.get_image_features(pixel_values=pixel_values)
                embs = self._to_tensor(embs)
                embs = embs / embs.norm(dim=-1, keepdim=True)   # L2-normalise

                embs_np = embs.cpu().numpy()
                for out_j, orig_j in enumerate(valid_idxs):
                    chunk_embs[orig_j] = embs_np[out_j]

            all_embs.append(chunk_embs)
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a single query string → (dim,) L2-normalised vector."""
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_inputs = {
            "input_ids":      inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }
        emb = self.model.get_text_features(**text_inputs)
        emb = self._to_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image → (dim,) L2-normalised vector."""
        image = image.convert("RGB") if image.mode != "RGB" else image
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        emb = self.model.get_image_features(pixel_values=pixel_values)
        emb = self._to_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)
