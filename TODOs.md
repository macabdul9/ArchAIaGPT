# ArchAIaGPT Roadmap & TODOs 🚀

This document tracks planned architectural upgrades and feature enhancements for the ArchAIaGPT system.

---

## 🏗️ Future Implementation: Qwen3-VL Integration

We aim to move towards a unified, fully local, and multimodal architecture using the **Qwen3-VL** model family.

### 1. Unified Multimodal Embeddings (Qwen3-VL-Embedding-2B)
**Goal:** Replace CLIP with a model that can encode combined Text+Image inputs into a single semantic vector.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
*   **Backend:** [vLLM](https://github.com/vllm-project/vllm) with `runner="pooling"`
*   **Sample Implementation Logic:**

```python
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image

# 1. Load with pooling runner
engine_args = EngineArgs(
    model="models/Qwen3-VL-Embedding-2B",
    runner="pooling",
    dtype="bfloat16",
    trust_remote_code=True
)
llm = LLM(**vars(engine_args))

# 2. Format multimodal conversation
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {"role": "user", "content": [
        {"type": "image", "image": "file:///path/to/artifact.jpg"},
        {"type": "text", "text": "Detailed description of the find."}
    ]}
]

# 3. Extract pooling output
outputs = llm.embed([{"prompt": prompt, "multi_modal_data": {"image": img}}])
embedding = outputs[0].outputs.embedding
```

---

### 2. Local Multimodal Generation (Qwen3-VL-2B-Instruct)
**Goal:** Replace GPT-4o-mini with a local VLM that can "see" the retrieved images to support its reasoning.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
*   **Tasks:**
    *   Update `Generator` to support `vllm` backend for vision tasks.
    *   Modify `context_builder.py` to return multimodal content blocks (image paths + text) rather than just a flat string.
    *   Update `app.py` to pass the local file paths of the Top-K retrieved images back into the LLM prompt.

---

### 3. General Enhancements
- [ ] **Unified Mode Switch**: Add a `--local_vlm` flag to `scripts/launch.sh` to toggle between CLIP/OpenAI and Qwen3/Local.
- [ ] **Vector Database Comparison**: Benchmark FAISS performance between CLIP (512-dim) and Qwen3 (higher dim).
- [ ] **Metadata Filtering Expansion**: Add more granular filters (trench, material, dimensions) to the Gradio UI.
- [ ] **Export Feature**: Allow users to download the retrieved artifact list and the LLM's analytical report as a PDF.

---

## 🛠️ Performance Optimization
- [ ] Implement **FAISS IVFFlat** or **HNSW** for faster search once the dataset grows beyond 10k artifacts.
- [ ] Add **Request Batching** in the Gradio UI for high-concurrency usage on server deployments.
