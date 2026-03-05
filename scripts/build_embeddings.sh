#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# ArchAIaGPT — Build Embedding Indexes
# ──────────────────────────────────────────────────────────────────────────────
# Run this once after generating the HF dataset.
# Encodes all artifacts with CLIP ViT-B/32 and saves FAISS indexes.
#
# Usage:
#   bash ArchAIaGPT/scripts/build_embeddings.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET="outputs/hf_dataset"
INDEX_DIR="ArchAIaGPT/indexes"
CLIP_MODEL="openai/clip-vit-base-patch32"
BATCH_SIZE=32
DEVICE="cuda"    # set to "cpu" if no GPU

cd "$(dirname "$0")/../.."   # cd to project root (Meta2Text/)

echo "═══════════════════════════════════════════════"
echo " ArchAIaGPT — Building Embedding Indexes"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Dataset:    ${DATASET}"
echo "  Index dir:  ${INDEX_DIR}"
echo "  CLIP model: ${CLIP_MODEL}"
echo "  Device:     ${DEVICE}"
echo ""

python ArchAIaGPT/embeddings/build_index.py \
    --dataset    "${DATASET}" \
    --out_dir    "${INDEX_DIR}" \
    --clip_model "${CLIP_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --device     "${DEVICE}"

echo ""
echo "═══════════════════════════════════════════════"
echo " Done! Indexes saved to ${INDEX_DIR}"
echo "═══════════════════════════════════════════════"
