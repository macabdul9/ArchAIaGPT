#!/bin/bash
# Builds and launches the ArchAIaGPT app with multimodal features.

set -e

# Configuration
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
INDEX_DIR="indexes"
MODEL_TYPE="clip"
DEVICE="cuda"
PORT=7861
SHARE="--share"

cd "$(dirname "$0")/.."

# Fix for PermissionError in shared /tmp/gradio
export GRADIO_TEMP_DIR="$(pwd)/.gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

export PYTHONPATH=$PYTHONPATH:.

echo "--- ArchAIaGPT: Full Build & Launch ---"

# 1. Build Index if missing for the default model
if [ ! -f "${INDEX_DIR}/text.faiss" ] || [ ! -f "${INDEX_DIR}/image.faiss" ]; then
    echo "Default indexes missing. Building CLIP embeddings..."
    python embeddings/build_index.py \
        --dataset     "${DATASET}" \
        --out_dir     "${INDEX_DIR}" \
        --images_root "${IMAGES_ROOT}" \
        --model_type  "${MODEL_TYPE}" \
        --device      "${DEVICE}"
else
    echo "Indexes found in ${INDEX_DIR}. Skipping build."
fi

# 2. Launch Gradio App
echo "Launching Gradio app on port ${PORT}..."
python app.py \
    --dataset "${DATASET}" \
    --images_root "${IMAGES_ROOT}" \
    --port ${PORT} \
    ${SHARE}
