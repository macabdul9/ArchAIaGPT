# ArchAIaGPT Roadmap & TODOs 🚀

This document tracks planned architectural upgrades, professional orchestration tasks, and feature enhancements for the ArchAIaGPT system. Each item below includes a proposed **GitHub Issue Title** for easy tracking.

---

## 🏗️ Future Implementation: Qwen3-VL Integration

### 1. Unified Multimodal Embeddings
**Issue Title:** `feat: Integrate Qwen3-VL-Embedding-2B for multimodal retrieval`

**Goal:** Replace CLIP with a model that can encode combined Text+Image inputs into a single semantic vector, enabling "Instruction-Aware" search.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
*   **Backend:** [vLLM](https://github.com/vllm-project/vllm) with `runner="pooling"`
*   **Tasks:**
    *   [ ] Implement `embeddings/qwen_encoder.py` wrapping vLLM's pooling output.
    *   [ ] Use `tokenizer.apply_chat_template` for exact training distribution matching.
    *   [ ] Re-build FAISS indexes for 2048-dimensional vectors.
    *   [ ] Explore `fp8` or `awq` for the embedding model.

---

### 2. Local Multimodal Generation
**Issue Title:** `feat: Support Qwen3-VL-2B-Instruct as local VLM generator`

**Goal:** Transition from text-only RAG to Vision-Language RAG, where the LLM can "see" the artifacts it is discussing.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
*   **Tasks:**
    *   [ ] Update `generation/generator.py` to handle `image` content types.
    *   [ ] Refactor `context_builder.py` to return message blocks (Text + Image paths).
    *   [ ] Develop system prompts for Optical Archaeological Analysis.

---

## 🌐 Orchestration & Production Serving

### 3. High-Concurrency vLLM Serving
**Issue Title:** `orchestration: Move to standalone vLLM OpenAI-compatible API server`

**Goal:** Decouple the UI from the model and enable concurrent request handling.
*   [ ] Implement API separation using vLLM entrypoints.
*   [ ] Enable continuous batching for simultaneous user support.
*   [ ] Configure Gradio queuing and concurrency limits.

### 4. Semantic Caching & Vector Latency
**Issue Title:** `perf: Implement Redis semantic caching and HNSW vector index`

**Goal:** Sub-10ms retrieval and zero-cost repeated queries.
*   [ ] Integrate **Redis** with **GPTCache**.
*   [ ] Migrate FAISS from `IndexFlatIP` to `IndexIVFFlat` or `HNSW`.
*   [ ] Tune KV cache management for specific hardware.

### 5. Production Monitoring & Observability
**Issue Title:** `reliability: Add Prometheus monitoring and health-check endpoints`

**Goal:** Professional system visibility and automated recovery.
*   [ ] Implement `/health` endpoints for Docker/K8s probes.
*   * [ ] Integrate Prometheus exporters for RPS, TTFT, and GPU metrics.
*   [ ] Implement application-level Rate Limiting.

---

## 🛠️ General Feature Roadmap

### 6. Unified Model Launcher
**Issue Title:** `enhancement: Add unified mode switch for Local vs Cloud backends`
*   [ ] Add `--local_vlm` flag to `scripts/launch.sh`.

### 7. Advanced Metadata Filtering
**Issue Title:** `ui: Expand sidebar filters to include trench, material, and dimensions`
*   [ ] Add more granular filter dropdowns/sliders to the side-panel.

### 8. Analytical Report Export
**Issue Title:** `feat: Add PDF/JSON export for artifact analysis reports`
*   [ ] Implement "Generate Report" functionality for retrieved results.

### 9. Geospatial Visualization
**Issue Title:** `ui: Integrate interactive map for artifact discovery locations`
*   [ ] Visualize coordinates using Plotly or Leaflet integration.
