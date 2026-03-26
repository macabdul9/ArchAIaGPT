"""
Orchestration layer for the ArchAIaGPT RAG pipeline.
Handles the end-to-end flow from user query inputs to retrieval results and LLM generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from retrieval.retriever import Retriever, ArtifactResult
from generation.context_builder import build_context
from generation.factory import get_generator
from config import TOP_K, TEXT_WEIGHT

@dataclass
class PipelineOutput:
    """
    Standardised container for RAG execution results.
    Includes the original query, generated answer, retrieved documents, and the LLM context.
    """
    query: str
    answer: str
    results: List[ArtifactResult]
    context: str

class ArchAIaGPT:
    """
    Multimodal RAG pipeline designed for archaeological artifact discovery.
    Coordinates between vector-based retrieval and generative model inference.
    """

    def __init__(
        self,
        text_index_path: str = None,
        image_index_path: str = None,
        meta_path: str = None,
        model_type: str = "clip",
        model_id: str = None,
        device: str = None,
        gen_backend: str = "openai",
        gen_model: str = None,
        gen_base_url: str = None,
        gen_api_key: str = None,
    ):
        print("Initialising ArchAIaGPT pipeline...")
        self.device = device

        # Initialise the retrieval engine (FAISS/CLIP/BM25)
        self.retriever = Retriever(
            text_index_path=text_index_path,
            image_index_path=image_index_path,
            meta_path=meta_path,
            model_type=model_type,
            model_id=model_id,
            device=device,
        )

        # Initialise the default generation backend
        self.generator = get_generator(
            backend=gen_backend,
            model_name=gen_model,
            base_url=gen_base_url,
            api_key=gen_api_key,
            device=device
        )

        print("Pipeline successfully loaded.")

    def search(
        self,
        query: str = None,
        image_query: any = None,
        top_k: int = None,
        text_weight: float = None,
        filters: Optional[Dict[str, str]] = None,
        generate: bool = True,
        generator_override = None,
    ) -> PipelineOutput:
        """
        Executes a multimodal search. Optionally triggers LLM reasoning over the results.
        
        Args:
            query: The user's text inquiry.
            image_query: Optional PIL image for visual similarity search.
            top_k: Number of artifacts to retrieve.
            text_weight: Relative weight for text vs image similarity (0.0 to 1.0).
            filters: Metadata filters for constraining the search space.
            generate: Whether to perform LLM generation.
            generator_override: Temporary generator instance to use for this specific call.
        """
        top_k = top_k or TOP_K
        text_weight = text_weight if text_weight is not None else TEXT_WEIGHT

        # Step 1: Document Retrieval
        results = self.retriever.retrieve(
            query=query,
            image_query=image_query,
            top_k=top_k,
            text_weight=text_weight,
            filters=filters,
        )

        # Step 2: Context Preparation
        # Format retrieved artifact metadata into a prompt-friendly string.
        context = build_context(results)

        # Step 3: Generative Reasoning
        answer = ""
        gen_engine = generator_override if generator_override else self.generator
        
        if generate and results:
            prompt_text = query.strip() if query else "Provide a summary of the following artifacts."
            answer = gen_engine.generate(query=prompt_text, context=context)
            
            if not answer:
                print("Warning: Generative model returned an empty response.")
        elif not results:
            answer = "No matching artifacts were found in the database for the given query."
        else:
            # Execution without generation (retrieval-only mode)
            answer = ""

        return PipelineOutput(
            query=query or "",
            answer=answer,
            results=results,
            context=context,
        )
