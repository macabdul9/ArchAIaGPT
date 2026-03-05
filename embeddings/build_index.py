"""
Build FAISS indexes from HF dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from datasets import load_from_disk

from config import CLIP_MODEL_ID, TEXT_FIELDS, CLIP_DIM
from embeddings.clip_encoder import CLIPEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS indexes from HF dataset")
    p.add_argument("--dataset",     required=True,  help="Path to HF dataset (save_to_disk format)")
    p.add_argument("--out_dir",     required=True,  help="Output directory for indexes")
    p.add_argument("--images_root", default="/data/group_data/dei-group/archaia",
                   help="Root dir for resolving relative image paths")
    p.add_argument("--text_fields", nargs="+",
                   default=["level_5_description"],
                   help="Which text columns to encode (default: level_5_description only)")
    p.add_argument("--max_images",  type=int, default=1,
                   help="Max images to encode per artifact (default: 1 = first image only)")
    p.add_argument("--clip_model",  default=CLIP_MODEL_ID)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default=None, help="cuda or cpu (auto-detect if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    ds = load_from_disk(args.dataset)
    n = len(ds)
    print(f"  Artifacts: {n}")

    # ── Detect image handling mode ────────────────────────────────────────────
    # Check for actual HF Image columns (image_0, image_1, …) vs. paths-only
    from datasets import Image as HFImage
    hf_image_cols = [
        c for c in ds.column_names
        if c.startswith("image_") and c != "image_paths"
        and isinstance(ds.features.get(c), HFImage)
    ]
    has_image_paths = "image_paths" in ds.column_names
    images_root = Path(args.images_root)

    if hf_image_cols:
        print(f"  Image mode: HF Image columns {hf_image_cols}")
    elif has_image_paths:
        print(f"  Image mode: loading from paths (images_root={images_root})")
    else:
        print(f"  Image mode: NO images available")
    text_fields = args.text_fields
    max_img = args.max_images
    print(f"  Text fields:  {text_fields}")
    print(f"  Max images:   {max_img} per artifact")

    # ── Init CLIP encoder ─────────────────────────────────────────────────────
    encoder = CLIPEncoder(model_id=args.clip_model, device=args.device)
    dim = encoder.dim

    # ── Encode texts ──────────────────────────────────────────────────────────
    print("\n═══ Phase 1: Encoding text fields ═══")
    text_embeddings = np.zeros((n, dim), dtype=np.float32)

    for idx in tqdm(range(n), desc="  Encoding text", unit="art", dynamic_ncols=True):
        row = ds[idx]
        field_texts = []
        for field in text_fields:
            val = row.get(field, "")
            if val and isinstance(val, str) and val.strip():
                field_texts.append(val.strip())

        if field_texts:
            # Encode non-empty text fields → mean pool
            embs = encoder.encode_texts(field_texts)    # (k, dim)
            mean_emb = embs.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb /= norm
            text_embeddings[idx] = mean_emb

    # ── Encode images ─────────────────────────────────────────────────────────
    print("\n═══ Phase 2: Encoding images ═══")
    image_embeddings = np.zeros((n, dim), dtype=np.float32)

    for idx in tqdm(range(n), desc="  Encoding images", unit="art", dynamic_ncols=True):
        row = ds[idx]
        images = []

        if hf_image_cols:
            for col in hf_image_cols[:max_img]:
                img = row.get(col)
                if img is not None:
                    images.append(img)
        elif has_image_paths:
            raw = row.get("image_paths", "[]")
            try:
                rel_paths = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                rel_paths = []
            for rp in rel_paths[:max_img]:
                abs_path = images_root / rp
                if abs_path.exists():
                    try:
                        img = PILImage.open(abs_path).convert("RGB")
                        images.append(img)
                    except Exception:
                        pass

        if images:
            embs = encoder.encode_images(images)
            mean_emb = embs.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb /= norm
            image_embeddings[idx] = mean_emb

    # ── Build FAISS indexes (Inner Product on L2-normalised = cosine) ─────────
    print("\n═══ Phase 3: Building FAISS indexes ═══")

    text_index = faiss.IndexFlatIP(dim)
    text_index.add(text_embeddings)
    text_path = out_dir / "text.faiss"
    faiss.write_index(text_index, str(text_path))
    print(f"  ✓ Text index:  {text_path}  ({text_index.ntotal} vectors)")

    image_index = faiss.IndexFlatIP(dim)
    image_index.add(image_embeddings)
    image_path = out_dir / "image.faiss"
    faiss.write_index(image_index, str(image_path))
    print(f"  ✓ Image index: {image_path}  ({image_index.ntotal} vectors)")

    # ── Write metadata sidecar ────────────────────────────────────────────────
    print("\n═══ Phase 4: Writing metadata sidecar ═══")
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for idx in range(n):
            row = ds[idx]
            meta = {
                "idx":          idx,
                "artifact_id":  row.get("artifact_id", ""),
                "label":        row.get("label", ""),
                "description":  (row.get("description", "") or "")[:300],
                "level_1":      row.get("level_1_description", ""),
                "level_2":      row.get("level_2_description", ""),
                "level_3":      row.get("level_3_description", ""),
                "level_4":      row.get("level_4_description", ""),
                "level_5":      row.get("level_5_description", ""),
                "period":       row.get("period", ""),
                "project":      row.get("project", ""),
                "metadata_json": row.get("metadata", "{}"),
                "num_images":   row.get("num_images", 0),
            }
            # Store image paths
            raw = row.get("image_paths", "[]")
            try:
                meta["image_paths"] = json.loads(raw) if isinstance(raw, str) else []
            except (json.JSONDecodeError, TypeError):
                meta["image_paths"] = []
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"  ✓ Metadata:    {meta_path}  ({n} rows)")

    print(f"\n{'═'*60}")
    print(f"Done! All indexes saved to {out_dir.resolve()}")
    print(f"  text.faiss     — {text_embeddings.shape}")
    print(f"  image.faiss    — {image_embeddings.shape}")
    print(f"  metadata.jsonl — {n} artifact records")


if __name__ == "__main__":
    main()
