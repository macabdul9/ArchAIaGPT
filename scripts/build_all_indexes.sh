#!/bin/bash
# Builds FAISS indexes for all supported models.

set -e

# Configuration
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
INDEX_DIR="indexes"
MODELS=("clip" "bm25" "gemma" "vlm2vec" "e5-omni") # qwen3-vl requires pooling runner (vLLM)
DEVICE="cuda"

cd "$(dirname "$0")/.."

echo "--- ArchAIaGPT: Building ALL Indexes ---"

export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=$(pwd)/.hf_cache
mkdir -p "$HF_HOME"

for MODEL in "${MODELS[@]}"; do
    echo "Encoding model: ${MODEL} ..."
    python embeddings/build_index.py \
        --dataset     "${DATASET}" \
        --out_dir     "${INDEX_DIR}" \
        --images_root "${IMAGES_ROOT}" \
        --model_type  "${MODEL}" \
        --device      "${DEVICE}"
done

echo "Done! All indexes saved to ${INDEX_DIR}."
