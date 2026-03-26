#!/bin/bash
# Scripts to serve Qwen3-VL models via vLLM.

echo "--- Serving Qwen3-VL models via vLLM ---"

# Start Pooling Service (for embedding)
echo "Starting Pooling Engine (Qwen3-VL-Embedding-2B) on port 8001 ..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --task pooling \
    --port 8001 \
    --dtype bfloat16 \
    --trust-remote-code &

# Start Instruct Service (for generation)
echo "Starting Instruct Engine (Qwen3-VL-2B-Instruct) on port 8000 ..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code &

echo "Services started in background."
