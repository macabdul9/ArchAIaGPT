#!/bin/bash
# Simple launch script (skips build)

set -e

# Configuration
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
PORT=7861
SHARE="--share"

cd "$(dirname "$0")/.."

# Fix for PermissionError in shared /tmp/gradio
export GRADIO_TEMP_DIR="$(pwd)/.gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=$(pwd)/.hf_cache
mkdir -p "$HF_HOME"

echo "--- ArchAIaGPT: Launch Mode ---"

python app.py \
    --dataset "${DATASET}" \
    --images_root "${IMAGES_ROOT}" \
    --port ${PORT} \
    ${SHARE}
