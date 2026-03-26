#!/bin/bash
# Simple launch script (skips build)

set -e

DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
GEN_BACKEND="openai"
PORT=7861
SHARE="--share"

cd "$(dirname "$0")/.."

# Fix for PermissionError in shared /tmp/gradio
export GRADIO_TEMP_DIR="$(pwd)/.gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

echo "--- ArchAIaGPT: Launch Mode ---"

python app.py \
    --dataset "${DATASET}" \
    --port ${PORT} \
    --gen_backend "${GEN_BACKEND}" \
    ${SHARE}
