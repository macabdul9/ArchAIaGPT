"""
Central configuration — all paths, model IDs, and hyperparameters.
"""

from pathlib import Path

# Paths
PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_PATH  = PROJECT_ROOT / "outputs" / "hf_dataset"

INDEX_DIR     = Path(__file__).resolve().parent / "indexes"
TEXT_INDEX     = INDEX_DIR / "text.faiss"
IMAGE_INDEX   = INDEX_DIR / "image.faiss"
META_STORE    = INDEX_DIR / "metadata.jsonl"

# ── CLIP Model ────────────────────────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM      = 512                               # output dim for ViT-B/32

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 5
TEXT_WEIGHT   = 0.5
IMAGE_WEIGHT  = 0.5

# Text fields to embed (averaged into one text vector per artifact)
TEXT_FIELDS = [
    "description",
    "level_1_description",
    "level_2_description",
    "level_3_description",
    "level_4_description",
    "level_5_description",
]

# ── Generation (OpenAI-compatible — works for vLLM + OpenAI) ──────────────────
VLLM_BASE_URL   = "http://localhost:8000/v1"
VLLM_MODEL      = "Qwen/Qwen2.5-VL-7B-Instruct"
OPENAI_MODEL    = "gpt-4o-mini"

MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.7

SYSTEM_PROMPT = (
    "You are ArchAIaGPT, an expert archaeological assistant. "
    "You are given a set of retrieved artifact records from an archaeological database. "
    "Answer the user's query using ONLY the evidence in the retrieved artifacts below. "
    "Be specific, cite artifact IDs and labels, and use formal archaeological language. "
    "If the retrieved artifacts do not contain enough information to answer, say so."
)
