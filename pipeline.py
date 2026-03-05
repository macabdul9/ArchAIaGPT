"""
End-to-end pipeline: query → retrieve → generate → answer + results.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from retrieval.retriever import Retriever, ArtifactResult
from generation.context_builder import build_context
from generation.generator import Generator
from config import TOP_K, TEXT_WEIGHT


@dataclass
class PipelineOutput:
    """Container for the full RAG output."""
    query:     str
    answer:    str
    results:   List[ArtifactResult]
    context:   str                      # the text context that was sent to the LLM


class ArchAIaGPT:
    """
    Full RAG pipeline over the Archaia dataset.

    Usage:
        pipe = ArchAIaGPT(gen_backend="openai")
        out  = pipe.search("terracotta tiles with cross marks")
        print(out.answer)
        for r in out.results:
            print(r.label, r.fused_score)
    """

    def __init__(
        self,
        # Retriever args
        text_index_path:  str = None,
        image_index_path: str = None,
        meta_path:        str = None,
        clip_model:       str = None,
        device:           str = None,
        # Generator args
        gen_backend:      str = "openai",
        gen_model:        str = None,
        gen_base_url:     str = None,
        gen_api_key:      str = None,
        gen_max_tokens:   int = None,
        gen_temperature:  float = None,
    ):
        print("ArchAIaGPT — Initialising ...")

        # ── Retriever ─────────────────────────────────────────────────────────
        self.retriever = Retriever(
            text_index_path  = text_index_path,
            image_index_path = image_index_path,
            meta_path        = meta_path,
            clip_model       = clip_model,
            device           = device,
        )

        # ── Generator ─────────────────────────────────────────────────────────
        self.generator = Generator(
            backend     = gen_backend,
            model       = gen_model,
            base_url    = gen_base_url,
            api_key     = gen_api_key,
            max_tokens  = gen_max_tokens,
            temperature = gen_temperature,
        )

        print("\n[ArchAIaGPT] Pipeline ready.\n")

    def search(
        self,
        query:       str = None,
        image_query: any = None,
        top_k:       int = None,
        text_weight: float = None,
        filters:     Optional[Dict[str, str]] = None,
        generate:    bool  = True,
    ) -> PipelineOutput:
        """
        Full RAG: retrieve artifacts + optionally generate an answer.
        Supports text search, image search, or multimodal search.
        """
        top_k       = top_k       or TOP_K
        text_weight = text_weight if text_weight is not None else TEXT_WEIGHT

        has_text = bool(query and query.strip())

        # ── Retrieve ──────────────────────────────────────────────────────────
        results = self.retriever.retrieve(
            query       = query,
            image_query = image_query,
            top_k       = top_k,
            text_weight = text_weight,
            filters     = filters,
        )

        # ── Build context ─────────────────────────────────────────────────────
        context = build_context(results)

        # ── Generate (optional) ───────────────────────────────────────────────
        answer = ""
        if generate and has_text and results:
            answer = self.generator.generate(query=query.strip(), context=context)
        elif not results:
            answer = "No relevant artifacts were found for your query."
        elif not has_text and image_query is not None:
            answer = "*(Visual search only — LLM generation disabled for image queries without text.)*"
        else:
            answer = ""    # retrieve-only mode

        return PipelineOutput(
            query   = query or "",
            answer  = answer,
            results = results,
            context = context,
        )
