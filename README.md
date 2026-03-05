# ArchAIaGPT 🏺

ArchAIaGPT is a Retrieval-Augmented Generation (RAG) system for archaeological artifacts. It uses CLIP for multimodal (text+image) retrieval and OpenAI/vLLM for generating analytical responses based on retrieved context.

## Features
- **Multimodal Search**: Search by text, image, or both.
- **Hierarchical Context**: Retrieves artifacts across 5 levels of analytical descriptions.
- **Streaming UI**: Clean Gradio interface with gallery visualization.
- **Dual-Index Retriever**: Fuses text and visual similarity scores.

## Installation

```bash
# Clone the repository
git clone git@github.com:macabdul9/ArchAIaGPT.git
cd ArchAIaGPT

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Build & Launch (First Time)
This script builds CLIP embeddings and FAISS indexes if they don't exist, then launches the app.
```bash
bash scripts/build_and_launch.sh
```

### Launch Only
If indexes are already built:
```bash
bash scripts/launch.sh
```

## Configuration
Edit `config.py` to change default paths, model IDs, or similarity weights.

## Project Structure
- `app.py`: Gradio web interface.
- `pipeline.py`: Main RAG orchestration.
- `retrieval/`: Dual-index FAISS retriever.
- `embeddings/`: CLIP encoding logic & index builder.
- `generation/`: LLM prompting & context building.
- `scripts/`: Helper shells for indexing and launching.

## License
MIT
