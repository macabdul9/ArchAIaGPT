#!/usr/bin/env python3
"""
ArchAIaGPT/app.py
──────────────────
Gradio demo for ArchAIaGPT — Archaeological Retrieval-Augmented Assistant.

Usage:
    python ArchAIaGPT/app.py [--share] [--port 7860]
"""

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
from datasets import load_from_disk

from pipeline import ArchAIaGPT
from config import DATASET_PATH, TOP_K, TEXT_WEIGHT

# Imports
# Globals (initialised in main)
# Imports
pipe: ArchAIaGPT = None
dataset = None                  # full HF dataset (for loading images)
project_choices: list = []      # unique project names for filter dropdown


IMAGES_ROOT = None   # set in main()

def load_artifact_images(idx: int):
    """Load images for a given artifact index from the HF dataset."""
    if dataset is None:
        return []
    row = dataset[idx]
    images = []

    # Try HF Image columns first (image_0, image_1, ...)
    from datasets import Image as HFImage
    hf_img_cols = [
        c for c in dataset.column_names
        if c.startswith("image_") and c != "image_paths"
        and isinstance(dataset.features.get(c), HFImage)
    ]
    if hf_img_cols:
        for col in sorted(hf_img_cols):
            img = row.get(col)
            if img is not None:
                images.append(img)
    elif "image_paths" in dataset.column_names and IMAGES_ROOT:
        # Load from paths
        from PIL import Image as PILImage
        raw = row.get("image_paths", "[]")
        try:
            rel_paths = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            rel_paths = []
        for rp in rel_paths:
            abs_path = Path(IMAGES_ROOT) / rp
            if abs_path.exists():
                try:
                    images.append(PILImage.open(abs_path).convert("RGB"))
                except Exception:
                    pass
    return images


def search_fn(query, image_query, top_k, text_weight, project_filter, do_generate):
    """
    Main search function wired to the Gradio UI.
    """
    if not (query and query.strip()) and image_query is None:
        return "Please enter a query or upload an image.", [], ""

    filters = None
    if project_filter and project_filter != "All":
        filters = {"project": project_filter}

    out = pipe.search(
        query       = query.strip() if query else None,
        image_query = image_query,
        top_k       = int(top_k),
        text_weight = float(text_weight),
        filters     = filters,
        generate    = bool(do_generate),
    )

    results = out.results
    answer_md = out.answer

    # ── Gallery: collect images from top-K artifacts ─────────────────────────
    gallery_items = []
    for r in results:
        images = load_artifact_images(r.idx)
        for img in images:
            if img is not None:
                caption = f"{r.label} (score: {r.fused_score:.3f})"
                gallery_items.append((img, caption))

    # ── Details panel ────────────────────────────────────────────────────────
    details_parts = []
    for i, r in enumerate(results, 1):
        try:
            meta = json.loads(r.metadata_json) if r.metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        obj_type = meta.get("object_type", "—")
        material = meta.get("material", "—")
        color    = meta.get("color_munsell", "—")
        size     = meta.get("size", "—")
        trench   = meta.get("trench", "—")

        detail = f"""### Artifact {i} — {r.label or r.artifact_id}

| Field | Value |
|---|---|
| **Relevance Score** | {r.fused_score:.4f} (text: {r.text_score:.4f}, image: {r.image_score:.4f}) |
| **Project** | {r.project or '—'} |
| **Period** | {r.period or '—'} |
| **Type** | {obj_type} |
| **Material** | {material} |
| **Munsell Color** | {color} |
| **Dimensions** | {size} |
| **Trench** | {trench} |

**Level 1:** {r.level_1 or '—'}

**Level 2:** {r.level_2 or '—'}

**Level 3:** {r.level_3 or '—'}

<details>
<summary>📄 Level 4 (Full Analytical)</summary>

{r.level_4 or '—'}

</details>

<details>
<summary>📖 Level 5 (Publication-Ready)</summary>

{r.level_5 or '—'}

</details>

---
"""
        details_parts.append(detail)

    details_md = "\n".join(details_parts) if details_parts else "No artifacts found."
    answer_md = out.answer if out.answer else "*(Generation disabled — showing retrieval results only.)*"

    return answer_md, gallery_items, details_md


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""

    with gr.Blocks(
        title="ArchAIaGPT — Archaeological RAG Assistant",
    ) as app:

        gr.HTML("""
        <div class="main-title">🏺 ArchAIaGPT</div>
        <div class="subtitle">Retrieval-Augmented Archaeological Assistant</div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                query_box = gr.Textbox(
                    label="Query",
                    value="Fuel and Plant Use in Northern Mesopotamia",
                    placeholder="e.g. What types of terracotta tiles were found at Murlo?",
                    lines=3,
                    elem_id="query_input",
                )
                image_query = gr.Image(
                    label="Image Query (Optional)",
                    type="pil",
                    elem_id="image_input",
                )

                with gr.Accordion("⚙️ Search Settings", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, step=1, value=TOP_K,
                        label="Top-K Artifacts",
                    )
                    weight_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=TEXT_WEIGHT,
                        label="Text Weight (0 = image only, 1 = text only)",
                    )
                    project_dd = gr.Dropdown(
                        choices=["All"] + project_choices,
                        value="All",
                        label="Filter by Project",
                    )
                    gen_toggle = gr.Checkbox(
                        value=True,
                        label="Generate LLM Answer",
                    )

                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

            # ── Right panel: answer + gallery ─────────────────────────────────
            with gr.Column(scale=2):
                answer_box = gr.Markdown(
                    label="Answer",
                    value="*Enter a question and click Search.*",
                    elem_id="answer_panel",
                )

                gallery = gr.Gallery(
                    label="Retrieved Artifact Images",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                    elem_id="artifact_gallery",
                )

        # ── Full-width details ────────────────────────────────────────────────
        with gr.Accordion("📋 Artifact Details (all levels)", open=False):
            details_box = gr.Markdown(
                value="*Search results will appear here.*",
                elem_id="details_panel",
            )

        # ── Wire events ──────────────────────────────────────────────────────
        search_btn.click(
            fn=search_fn,
            inputs=[query_box, image_query, top_k_slider, weight_slider, project_dd, gen_toggle],
            outputs=[answer_box, gallery, details_box],
        )
        query_box.submit(
            fn=search_fn,
            inputs=[query_box, image_query, top_k_slider, weight_slider, project_dd, gen_toggle],
            outputs=[answer_box, gallery, details_box],
        )

    return app


def main():
    global pipe, dataset, project_choices, IMAGES_ROOT

    parser = argparse.ArgumentParser(description="ArchAIaGPT Gradio Demo")
    parser.add_argument("--share",       action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",        type=int, default=7860)
    parser.add_argument("--dataset",     default=str(DATASET_PATH),  help="Path to HF dataset")
    parser.add_argument("--images_root", default="/data/group_data/dei-group/archaia",
                       help="Root dir for resolving relative image paths")
    parser.add_argument("--device",      default=None,               help="cuda / cpu")
    parser.add_argument("--gen_backend", default="openai",           choices=["openai", "vllm"])
    parser.add_argument("--gen_model",   default=None)
    parser.add_argument("--gen_api_key", default=None)
    args = parser.parse_args()

    # ── Set images root for artifact image loading ────────────────────────────
    IMAGES_ROOT = args.images_root

    # ── Load dataset for image display ────────────────────────────────────────
    print(f"Loading HF dataset for images: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    print(f"  {len(dataset)} artifacts loaded")

    # Extract unique project names for the filter dropdown
    projects = set()
    for i in range(len(dataset)):
        p = dataset[i].get("project", "")
        if p and p.strip():
            projects.add(p.strip())
    project_choices = sorted(projects)
    print(f"  Projects: {project_choices}")

    # ── Init pipeline ─────────────────────────────────────────────────────────
    pipe = ArchAIaGPT(
        device       = args.device,
        gen_backend  = args.gen_backend,
        gen_model    = args.gen_model,
        gen_api_key  = args.gen_api_key,
    )

    # ── Launch ────────────────────────────────────────────────────────────────
    app = build_app()

    # Gradio 6.0: theme/css go in launch()
    app.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
        theme       = gr.themes.Soft(primary_hue="amber", secondary_hue="stone"),
    )


if __name__ == "__main__":
    main()
