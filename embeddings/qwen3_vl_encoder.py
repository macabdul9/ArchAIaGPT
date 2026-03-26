import os
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from .base_encoder import BaseEncoder

class Qwen3VLEncoder(BaseEncoder):
    """Qwen3-VL Multimodal Embedding encoder using vLLM."""

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-Embedding-2B", device: str = None, dtype: str = "bfloat16"):
        super().__init__(device)
        from vllm import LLM, EngineArgs
        
        # vLLM handles device_map internally
        engine_args = EngineArgs(
            model=model_path,
            runner="pooling",
            dtype=dtype,
            trust_remote_code=True,
        )
        self.llm = LLM(**vars(engine_args))
        self.tokenizer = self.llm.llm_engine.tokenizer
        self.dim = 2048 # Default for 2B model

    def _format_input(self, text: str = None, image: Image.Image = None, instruction: str = "Represent the user's input.") -> List[Dict]:
        content = []
        if image:
            content.append({'type': 'image', 'image': image})
        if text:
            content.append({'type': 'text', 'text': text})
        if not content:
            content.append({'type': 'text', 'text': ""})
            
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]

    def _prepare_vllm_input(self, text: str = None, image: Image.Image = None, instruction: str = "Represent the user's input.") -> Dict[str, Any]:
        conversation = self._format_input(text, image, instruction)
        prompt_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image} if image else None
        }

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vllm_inputs = [self._prepare_vllm_input(text=t) for t in texts]
        outputs = self.llm.embed(vllm_inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        return np.array(embeddings).astype(np.float32)

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        vllm_inputs = [self._prepare_vllm_input(image=img) for img in images]
        outputs = self.llm.embed(vllm_inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        return np.array(embeddings).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vllm_input = self._prepare_vllm_input(text=query)
        output = self.llm.embed([vllm_input])[0]
        return np.array(output.outputs.embedding).astype(np.float32)

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        vllm_input = self._prepare_vllm_input(image=image)
        output = self.llm.embed([vllm_input])[0]
        return np.array(output.outputs.embedding).astype(np.float32)

    def encode_multimodal_query(self, text: str, image: Image.Image) -> np.ndarray:
        """Specific for Qwen3-VL multimodal query (Instruction-aware)."""
        vllm_input = self._prepare_vllm_input(text=text, image=image)
        output = self.llm.embed([vllm_input])[0]
        return np.array(output.outputs.embedding).astype(np.float32)
