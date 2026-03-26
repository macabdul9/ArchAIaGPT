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

from config import CLIP_MODEL_ID, TEXT_FIELDS, INDEX_DIR
from embeddings.factory import get_encoder


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS indexes from HF dataset")
    p.add_argument("--dataset",     required=True,  help="Path to HF dataset (save_to_disk format)")
    p.add_argument("--out_dir",     default=str(INDEX_DIR),  help="Output directory for indexes")
    p.add_argument("--images_root", default="/data/group_data/dei-group/archaia",
                   help="Root dir for resolving relative image paths")
    p.add_argument("--text_fields", nargs="+",
                   default=TEXT_FIELDS,
                   help="Which text columns to encode")
    p.add_argument("--max_images",  type=int, default=1,
                   help="Max images to encode per artifact")
    p.add_argument("--model_type",  default="clip", help="clip, bm25, gemma, qwen3-vl, e5-omni, vlm2vec")
    p.add_argument("--model_id",    default=None)
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
    from datasets import Image as HFImage
    hf_image_cols = [
        c for c in ds.column_names
        if c.startswith("image_") and c != "image_paths"
        and isinstance(ds.features.get(c), HFImage)
    ]
    has_image_paths = "image_paths" in ds.column_names
    images_root = Path(args.images_root)

    print(f"  Model: {args.model_type} ({args.model_id or 'default'})")
    text_fields = args.text_fields
    max_img = args.max_images
    print(f"  Text fields:  {text_fields}")
    print(f"  Max images:   {max_img} per artifact")

    # ── Init encoder ──────────────────────────────────────────────────────────
    encoder = get_encoder(args.model_type, model_id=args.model_id, device=args.device)
    dim = encoder.dim

    # ── Encode texts ──────────────────────────────────────────────────────────
    print("\n═══ Phase 1: Encoding text fields ═══")
    
    # Handle BM25 (Sparse) differently if needed, but for FAISS we want dense if possible
    # BM25Encoder returns Sparse objects. FAISS IndexFlatIP requires dense vectors.
    # If using BM25, we might need a different index type or transform it.
    # For now, let's assume dense encoders for FAISS.
    
    if args.model_type == "bm25":
        print("[Warning] BM25 returns sparse embeddings. IndexFlatIP might not be suitable.")
        # We can implement a sparse index but FAISS IndexFlatIP is dense.
        # Skip for now or implement as list of dicts.
    
    text_embeddings = []
    
    for idx in tqdm(range(n), desc="  Encoding text", unit="art", dynamic_ncols=True):
        row = ds[idx]
        field_texts = []
        for field in text_fields:
            val = row.get(field, "")
            if val and isinstance(val, str) and val.strip():
                field_texts.append(val.strip())

        if field_texts:
            full_text = " ".join(field_texts)
            emb = encoder.encode_texts([full_text])[0]
            if isinstance(emb, np.ndarray):
                norm = np.linalg.norm(emb)
                if norm > 0: emb /= norm
                text_embeddings.append(emb)
            else:
                # Handle sparse or other
                text_embeddings.append(emb)
        else:
            text_embeddings.append(np.zeros(dim or 1))

    text_embeddings = np.array(text_embeddings).astype(np.float32)

    # ── Encode images ─────────────────────────────────────────────────────────
    print("\n═══ Phase 2: Encoding images ═══")
    image_embeddings = []

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
            if embs is not None:
                mean_emb = embs.mean(axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0: mean_emb /= norm
                image_embeddings.append(mean_emb)
            else:
                image_embeddings.append(np.zeros(dim or 1))
        else:
            image_embeddings.append(np.zeros(dim or 1))

    image_embeddings = np.array(image_embeddings).astype(np.float32)

    # ── Build FAISS indexes ───────────────────────────────────────────────────
    print("\n═══ Phase 3: Building FAISS indexes ═══")

    # Use model-specific file names
    suffix = f"_{args.model_type}" if args.model_type != "clip" else ""
    
    if dim:
        text_index = faiss.IndexFlatIP(dim)
        text_index.add(text_embeddings)
        text_path = out_dir / f"text{suffix}.faiss"
        faiss.write_index(text_index, str(text_path))
        print(f"  ✓ Text index:  {text_path}  ({text_index.ntotal} vectors)")

        image_index = faiss.IndexFlatIP(dim)
        image_index.add(image_embeddings)
        image_path = out_dir / f"image{suffix}.faiss"
        faiss.write_index(image_index, str(image_path))
        print(f"  ✓ Image index: {image_path}  ({image_index.ntotal} vectors)")
    else:
        # Sparse model — save as pickle
        print(f"  Saving sparse embeddings for {args.model_type} ...")
        import pickle
        text_path = out_dir / f"text_{args.model_type}.pkl"
        with open(text_path, "wb") as f:
            pickle.dump(text_embeddings, f)
        print(f"  ✓ Sparse Text Embs: {text_path}")
        
        # Images usually not supported for sparse, but save empty if needed
        image_path = out_dir / f"image_{args.model_type}.pkl"
        with open(image_path, "wb") as f:
            pickle.dump(image_embeddings, f)
        print(f"  ✓ Sparse Image Embs: {image_path}")

    # ── Write metadata sidecar ────────────────────────────────────────────────
    print("\n═══ Phase 4: Writing metadata sidecar ═══")
    meta_path = out_dir / "metadata.jsonl"
    if not meta_path.exists():
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
    else:
        print(f"  Metadata sidecar already exists: {meta_path}")

    print(f"\nDone! Indexes saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
