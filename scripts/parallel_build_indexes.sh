#!/bin/bash
# Parallel Index Building for ArchAIaGPT
# Distributes embedding tasks across all available GPUs to speed up pre-computation.

set -e

# Configuration
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
INDEX_DIR="indexes"
MODELS=("clip" "bm25" "gemma" "vlm2vec" "e5-omni")

# Move to project root
cd "$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=/data/hf_cache
mkdir -p "$HF_HOME"

# Detection of available hardware
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=0
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Warning: No GPUs detected. Falling back to sequential CPU execution."
    bash scripts/build_all_indexes.sh
    exit 0
fi

echo "Found $NUM_GPUS GPUs. Parallelising ${#MODELS[@]} embedding tasks..."

# Iterate through models and assign to GPUs in round-robin fashion
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU_ID=$(( i % NUM_GPUS ))
    
    echo "[GPU $GPU_ID] Launching index build for: $MODEL"
    
    # Run in background with isolated GPU visibility
    CUDA_VISIBLE_DEVICES=$GPU_ID python embeddings/build_index.py \
        --dataset     "${DATASET}" \
        --out_dir     "${INDEX_DIR}" \
        --images_root "${IMAGES_ROOT}" \
        --model_type  "${MODEL}" \
        --device      "cuda" &
done

# Wait for all background tasks to complete
echo "Waiting for all processes to finish..."
wait

echo "Successfully completed parallel embedding builds for all models."
