"""
Formats retrieved artifacts into a structured text context block
for injection into the LLM prompt.
"""

import json
from typing import List
from retrieval.retriever import ArtifactResult


def build_context(results: List[ArtifactResult], max_level: int = 5) -> str:
    """
    Build a formatted context string from retrieved artifacts.

    Each artifact is rendered as a labelled block containing:
    - ID, label, relevance score
    - Key metadata (type, material, period, site)
    - The richest description available (level_5 preferred, falls back to lower)

    Parameters
    ----------
    results    : list of ArtifactResult from the retriever
    max_level  : highest description level to include (1-5)

    Returns
    -------
    Formatted multi-block context string ready for LLM injection.
    """
    if not results:
        return "No artifacts were retrieved for this query."

    blocks = []

    for i, art in enumerate(results, 1):
        # ── Header ────────────────────────────────────────────────────────
        header = (
            f"[Artifact {i} — {art.label or art.artifact_id}]  "
            f"(relevance: {art.fused_score:.3f})"
        )

        # ── Metadata summary ──────────────────────────────────────────────
        meta_lines = []
        try:
            meta = json.loads(art.metadata_json) if art.metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        obj_type  = meta.get("object_type", "")
        material  = meta.get("material", "")
        color     = meta.get("color_munsell", "")
        size      = meta.get("size", "")
        trench    = meta.get("trench", "")

        if obj_type:
            meta_lines.append(f"Type: {obj_type}")
        if material:
            meta_lines.append(f"Material: {material}")
        if color:
            meta_lines.append(f"Munsell: {color}")
        if size:
            meta_lines.append(f"Dimensions: {size}")
        if art.period:
            meta_lines.append(f"Period: {art.period}")
        if art.project:
            meta_lines.append(f"Site/Project: {art.project}")
        if trench:
            meta_lines.append(f"Trench: {trench}")

        meta_block = " | ".join(meta_lines) if meta_lines else "No structured metadata."

        # ── Best available description (prefer richest level) ─────────────
        levels = {
            5: art.level_5,
            4: art.level_4,
            3: art.level_3,
            2: art.level_2,
            1: art.level_1,
        }

        description_lines = []
        for lvl in range(1, max_level + 1):
            text = levels.get(lvl, "")
            if text and text.strip():
                description_lines.append(f"  Level {lvl}: {text.strip()}")

        if not description_lines and art.description:
            description_lines.append(f"  Original: {art.description.strip()}")

        desc_block = "\n".join(description_lines) if description_lines else "  No description available."

        # ── Assemble ──────────────────────────────────────────────────────
        block = f"{header}\n{meta_block}\nDescriptions:\n{desc_block}"
        blocks.append(block)

    separator = "\n" + "─" * 60 + "\n"
    return separator.join(blocks)


def build_context_compact(results: List[ArtifactResult]) -> str:
    """
    Compact context: only level_5 (or best available) for each artifact.
    Useful when context window is limited.
    """
    if not results:
        return "No artifacts were retrieved."

    lines = []
    for i, art in enumerate(results, 1):
        # Pick richest available description
        desc = (art.level_5 or art.level_4 or art.level_3 or
                art.level_2 or art.level_1 or art.description or "No description.")
        lines.append(
            f"[{i}] {art.label or art.artifact_id} "
            f"(score: {art.fused_score:.3f}, project: {art.project or '?'})\n"
            f"    {desc.strip()}"
        )
    return "\n\n".join(lines)
