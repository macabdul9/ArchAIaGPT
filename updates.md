# ArchAIaGPT Quick Updates

- **Multimodal Encoders**: Support for BM25, Gemma, Qwen3-VL, E5-Omni, and VLM2Vec (via `embeddings/factory.py`).
- **Generation Backends**: Unified factory for OpenAI, Gemini, vLLM, and local VLMs (InternVL3, Ovis2).
- **Battle Mode**: New side-by-side comparison tab for different generation models.
- **Model Selection**: On-the-fly switching for both embedding and generation models in UI.
- **Feedback Logging**: Automated persistent storage of interaction ratings and battle winners in `feedback.csv`.
- **vLLM Integration**: New scaling scripts for serving embedding and instruct models.
- **Local Caching**: Configured local `.hf_cache` to bypass shared directory permission issues.
