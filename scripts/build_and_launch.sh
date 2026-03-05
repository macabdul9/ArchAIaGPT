#!/bin/bash
# Simpler launch script for ArchAIaGPT

set -e

# Default Config
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
INDEX_DIR="indexes"
CLIP_MODEL="openai/clip-vit-base-patch32"
BATCH_SIZE=32
DEVICE="cuda"
TEXT_FIELDS="level_5_description"
MAX_IMAGES=1
GEN_BACKEND="openai"
PORT=7860
SHARE="--share"

# Ensure we are in the repo root
cd "$(dirname "$0")/.."

echo "--- ArchAIaGPT: Build & Launch ---"
echo "Project root: $(pwd)"

# 1. Build Index if missing
if [ ! -f "${INDEX_DIR}/text.faiss" ] || [ ! -f "${INDEX_DIR}/image.faiss" ]; then
    echo "Indexes missing. Building CLIP embeddings..."
    python embeddings/build_index.py \
        --dataset     "${DATASET}" \
        --out_dir     "${INDEX_DIR}" \
        --images_root "${IMAGES_ROOT}" \
        --text_fields ${TEXT_FIELDS} \
        --max_images  "${MAX_IMAGES}" \
        --clip_model  "${CLIP_MODEL}" \
        --batch_size  "${BATCH_SIZE}" \
        --device      "${DEVICE}"
else
    echo "Indexes found in ${INDEX_DIR}. Skipping build."
fi

# 2. Launch App
echo "Launching Gradio app on port ${PORT}..."
CMD="python app.py --dataset ${DATASET} --port ${PORT} --gen_backend ${GEN_BACKEND}"

if [ "${SHARE}" = "--share" ]; then
    CMD="${CMD} --share"
fi

exec ${CMD}
