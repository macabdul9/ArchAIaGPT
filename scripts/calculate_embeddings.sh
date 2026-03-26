#!/bin/bash
# scripts/calculate_embeddings.sh
# Joint Embedding Calculation (Default: Gemma for Text, CLIP for Image)

set -e

# Project Root
ROOT_DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# 1. Default Configuration
TEXT_MODEL="gemma"
IMAGE_MODEL="clip"
TEXT_QUERY="The image shows an old, brown clay jug or pottery vessel."
IMAGE_SAMPLE="assets/dummy_artifact.jpg"
OUTPUT_DIR="embeddings_output"

# 2. Execution (Accepts overrides from CLI)
python embeddings/calculate_single.py \
  --text_model "${TEXT_MODEL}" \
  --image_model "${IMAGE_MODEL}" \
  --text "${TEXT_QUERY}" \
  --image_path "${IMAGE_SAMPLE}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
