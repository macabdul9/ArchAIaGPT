# ArchAIaGPT Roadmap & TODOs 🚀

This document tracks planned architectural upgrades, professional orchestration tasks, and feature enhancements for the ArchAIaGPT system.

---

## 🏗️ Future Implementation: Qwen3-VL Integration

We aim to move towards a unified, fully local, and multimodal architecture using the **Qwen3-VL** model family.

### 1. Unified Multimodal Embeddings (Qwen3-VL-Embedding-2B)
**Goal:** Replace CLIP with a model that can encode combined Text+Image inputs into a single semantic vector, enabling "Instruction-Aware" search.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
*   **Backend:** [vLLM](https://github.com/vllm-project/vllm) with `runner="pooling"`
*   **Detailed Implementation Tasks:**
    *   [ ] **Custom Encoder**: Implement `embeddings/qwen_encoder.py` wrapping vLLM's pooling output.
    *   [ ] **Chat Template Logic**: Use `tokenizer.apply_chat_template` to ensure inputs match the training distribution exactly.
    *   [ ] **Feature Indexing**: Re-build FAISS indexes to support the 2048-dimensional vectors (or appropriate dim for the 2B model).
    *   [ ] **Quantization**: Explore `fp8` or `awq` for the embedding model to reduce VRAM footprint while maintaining retrieval precision.

---

### 2. Local Multimodal Generation (Qwen3-VL-2B-Instruct)
**Goal:** Transition from text-only RAG to Vision-Language RAG, where the LLM can "see" the artifacts it is discussing.

*   **Model URL:** [huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
*   **Tasks:**
    *   [ ] **Vision-Aware Generator**: Update `generation/generator.py` to handle `image` content types in the messages payload.
    *   [ ] **Multimodal Context Builder**: Refactor `context_builder.py` to return a list of message blocks (Text + File URLs) for the top-8 artifacts.
    *   [ ] **Prompt Engineering**: Develop specific system prompts for "Archaeological Visual Analysis" that instruct the model to reference specific optical features in the provided images.

---

## 🌐 Orchestration & Production Serving

To serve ArchAIaGPT to a wider audience with high reliability and performance, the following orchestration tasks are prioritized:

### 1. High-Concurrency Serving
*   [ ] **API Separation**: Decouple the Gradio UI from the model. Move from "Offline Inference" to an **OpenAI-compatible vLLM Server**.
    *   *Command:* `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-VL-2B-Instruct --limit-mm-per-prompt image=8`
*   [ ] **Continuous Batching**: Standardize on vLLM to leverage continuous batching, which allows multiple users to be served simultaneously with minimal latency overhead.
*   [ ] **Gradio Queuing**: Enable `.queue(default_concurrency_limit=10)` in `app.py` to prevent server crashes under high load.

### 2. Caching & Latency Optimization
*   [ ] **Semantic Caching**: Integrate **Redis** with **GPTCache** to store and retrieve identical or highly similar queries without re-running the embedding or generation stages.
*   [ ] **FAISS Optimization**: Move from `IndexFlatIP` (Brute Force) to `IndexIVFFlat` or `HNSW` to maintain sub-10ms retrieval as the artifact database scales.
*   [ ] **KV Cache Management**: Tune vLLM's `gpu_memory_utilization` and `max_num_seqs` to maximize throughput on the specific GPU hardware available.

### 3. Monitoring & Reliability
*   [ ] **Health Checks**: Implement `/health` endpoints for the API and UI to integration with Docker/Kubernetes health probes.
*   [ ] **Observability**: Integrate **Prometheus** exporters to track:
    *   Requests per second (RPS)
    *   Time to First Token (TTFT)
    *   GPU Utilization and Memory pressure.
*   [ ] **Rate Limiting**: Implement basic rate limiting at the Nginx or application level to prevent API abuse.

---

## 🛠️ General Feature Roadmap
- [ ] **Unified Mode Switch**: Add a `--local_vlm` flag to `scripts/launch.sh` for easy environment toggling.
- [ ] **Metadata Filtering Expansion**: Add more granular filters (trench, material, dimensions, Munsell color) to the Gradio UI side-panel.
- [ ] **Export Feature**: Add a "Generate Report" button to export the retrieved artifacts and the LLM analysis as a formatted PDF or JSON file.
- [ ] **Interactive Map**: Visualize artifact discovery locations using a Map component (e.g., Gradio's `gr.Plot` with Plotly or a custom Leaflet integration).
