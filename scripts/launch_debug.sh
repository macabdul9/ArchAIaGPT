#!/bin/bash
# Launch ArchAIaGPT in Debug Mode (Hot Reloading + Local Only)

set -e

# Configuration
DATASET="/data/group_data/dei-group/archaia/archaia_hf_final"
IMAGES_ROOT="/data/group_data/dei-group/archaia"
PORT=7861

cd "$(dirname "$0")/.."

# Fix for PermissionError in shared /tmp/gradio
export GRADIO_TEMP_DIR="$(pwd)/.gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=$(pwd)/.hf_cache
mkdir -p "$HF_HOME"

echo "--- ArchAIaGPT: DEBUG MODE (Hot Reloading) ---"

# Using 'gradio' cli for hot-reloading if installed, else normal python with --debug
if command -v gradio &> /dev/null
then
    gradio app.py
else
    echo "Gradio CLI not found. Falling back to python -u for unbuffered logs."
    python -u app.py \
        --dataset "${DATASET}" \
        --images_root "${IMAGES_ROOT}" \
        --port ${PORT}
fi
