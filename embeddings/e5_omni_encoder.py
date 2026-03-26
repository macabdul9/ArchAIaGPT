import torch
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from .base_encoder import BaseEncoder

class E5OmniEncoder(BaseEncoder):
    """E5-Omni 3B encoder."""

    def __init__(self, model_name: str = "Haon-Chen/e5-omni-3B", device: str = None, dtype: str = "bfloat16"):
        super().__init__(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16
        ).to(self.device).eval()
        
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        self.dim = 3584 # Assuming Qwen2.5-Omni-3B hidden dimension

    def _encode_message(self, message):
        texts = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)[0] + "<|endoftext|>"
        # Need to handle process_mm_info or equivalent
        # For simplicity, if it's text-only or single-image, we can format manually
        
        # Simplified for now (only text and image supported as per requirements)
        audio_inputs, image_inputs, video_inputs = None, None, None
        
        # message format: [{"role": "user", "content": [{"type": "text", "text": ...}, {"type": "image", "image": ...}]}]
        for msg in message:
            for content in msg['content']:
                if content['type'] == 'image':
                    image_inputs = content['image']
                
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding="longest",
        )
        for k in inputs:
            if inputs[k] is not None:
                inputs[k] = inputs[k].to(self.device)

        cache_position = torch.arange(0, inputs["input_ids"].shape[1], device=self.device)
        inputs = self.model.prepare_inputs_for_generation(**inputs, use_cache=True, cache_position=cache_position)
        with torch.no_grad():
            model_outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)

        last_hidden_state = model_outputs.hidden_states[-1]
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps.cpu().numpy()

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_reps = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for t in batch:
                message = [{"role": "user", "content": [{"type": "text", "text": f"Query: {t}"}]}]
                all_reps.append(self._encode_message(message))
        return np.concatenate(all_reps, axis=0)

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_reps = []
        for img in images:
            message = [{"role": "user", "content": [{"type": "image", "image": img}]}]
            all_reps.append(self._encode_message(message))
        return np.concatenate(all_reps, axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        message = [{"role": "user", "content": [{"type": "text", "text": f"Query: {query}"}]}]
        return self._encode_message(message).flatten()

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        message = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        return self._encode_message(message).flatten()
