# ArchAIaGPT 🏺

ArchAIaGPT is a state-of-the-art **Retrieval-Augmented Generation (RAG)** system designed specifically for archaeological research. It enables researchers to query a vast database of artifacts using natural language and/or images, providing context-aware analytical responses.

The system leverages **OpenAI CLIP** for multimodal embedding and **FAISS** for efficient vector similarity search, combined with an LLM backend (OpenAI or vLLM) for response generation.

---

## 🚀 Key Features

*   **Multimodal Search**: Seamlessly query using text descriptions, uploaded images, or a combination of both.
*   **Dual-Index Retriever**: Utilizes separate FAISS indexes (accelerated with **FAISS-GPU**) for text and image embeddings, with a weighted fusion mechanism to rank results.
*   **Hierarchical Analytical Context**: Specifically optimized for the *Archaia* dataset, retrieving across five levels of description:
    *   **Level 1**: Basic Identification
    *   **Level 2**: Physical Classification
    *   **Level 3**: Detailed Visual Description
    *   **Level 4**: Full Analytical Data
    *   **Level 5**: Publication-Ready Synthesis
*   **Clean Gradio Interface**: A premium UI featuring a responsive gallery, detailed artifact inspection panels, and custom search filters (e.g., filter by Project).
*   **Flexible LLM Backend**: Supports both OpenAI (online) and vLLM (local) backends via a unified interface.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone git@github.com:macabdul9/ArchAIaGPT.git
cd ArchAIaGPT
```

### 2. Setup Environment
It is recommended to use a Conda environment:
```bash
conda create -n archatagpt python=3.10
conda activate archatagpt
pip install -r requirements.txt
```

### 3. API Configuration
Create a `.env` file in the root directory (or set environment variables):
```bash
OPENAI_API_KEY=your_sk_key_here
# Optional: HF_TOKEN=your_hf_token (if using protected models)
```

---

## 📂 Project Structure

```text
ArchAIaGPT/
├── app.py                # Gradio Web Interface
├── pipeline.py           # Main RAG Orchestration logic
├── config.py             # Centralized configuration (Paths, Hyperparameters)
├── requirements.txt      # Project dependencies
├── retrieval/            # Dual-index retrieval & score fusion
├── embeddings/           # CLIP encoders and index building logic
├── generation/           # LLM interaction and prompt engineering
├── scripts/              # Automation scripts for build and launch
└── indexes/              # (Ignored by Git) Local FAISS indexes & metadata
```

---

## 📖 Usage

### 📊 1. Building Indexes
If you are running the system for the first time or if the dataset has changed, you need to build the FAISS indexes:
```bash
bash scripts/build_embeddings.sh
```
*Note: This script encodes all text fields and images defined in `config.py` using CLIP.*

### 🖥️ 2. Launching the Application
To build (if missing) and launch the Gradio interface:
```bash
bash scripts/build_and_launch.sh
```

If the indexes are already prepared:
```bash
bash scripts/launch.sh
```

### 🔍 3. Multimodal Querying
*   **Text Search**: Enter any natural language query (e.g., *"Terracotta tiles from Murlo"*).
*   **Visual Search**: Upload an image to find visually similar artifacts.
*   **Combined**: Use both for maximum precision. The system will fuse the embeddings using the weights defined in `config.py`.

---

## ⚙️ Configuration

All major settings are located in `config.py`. You can adjust:
*   `TEXT_WEIGHT` / `IMAGE_WEIGHT`: Balance between text and visual similarity in search results.
*   `TOP_K`: Number of artifacts to retrieve.
*   `TEXT_FIELDS`: Which metadata columns to include in the text embedding.
*   `GEN_BACKEND`: Switch between `"openai"` and `"vllm"`.

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments
- **Archaia Project** for the comprehensive archaeological dataset.
- **OpenAI** for the CLIP model and GPT-4o-mini.
- **FAISS** for the high-performance vector search library.
