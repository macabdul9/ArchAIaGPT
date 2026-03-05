#!/bin/bash
# Script to build (or rebuild) FAISS indexes from HF dataset

set -e

DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
INDEX_DIR="indexes"
CLIP_MODEL="openai/clip-vit-base-patch32"
BATCH_SIZE=32
DEVICE="cuda"

cd "$(dirname "$0")/.."

echo "--- ArchAIaGPT: Building Indexes ---"

python embeddings/build_index.py \
    --dataset     "${DATASET}" \
    --out_dir     "${INDEX_DIR}" \
    --images_root "${IMAGES_ROOT}" \
    --clip_model  "${CLIP_MODEL}" \
    --batch_size  "${BATCH_SIZE}" \
    --device      "${DEVICE}"

echo "Done. Indexes saved to ${INDEX_DIR}"
