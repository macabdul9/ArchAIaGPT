import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Any
from .base_encoder import BaseEncoder

class VLM2VecEncoder(BaseEncoder):
    """VLM2Vec Multimodal Embedding encoder."""

    def __init__(self, model_name: str = "TIGER-Lab/VLM2Vec-Qwen2VL-2B", device: str = None, dtype: str = "bfloat16"):
        super().__init__(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vlm2vec_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VLM2Vec")
        if vlm2vec_path not in sys.path:
            sys.path.append(vlm2vec_path)
        
        from src.model import MMEBModel
        from src.arguments import ModelArguments
        from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
        
        self.QWEN2_VL = QWEN2_VL
        self.vlm_image_tokens = vlm_image_tokens
        
        model_args = ModelArguments(
            model_name='Qwen/Qwen2-VL-2B-Instruct',
            checkpoint_path=model_name,
            pooling='last',
            normalize=True,
            model_backbone='qwen2_vl',
            lora=True
        )
        
        self.processor = load_processor(model_args)
        self.model = MMEBModel.load(model_args)
        self.model = self.model.to(self.device, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16)
        self.model.eval()
        self.dim = 3584 # Based on Qwen2-VL-2B

    def _encode_single(self, text: str = None, image: Image.Image = None, is_query: bool = True) -> np.ndarray:
        if image:
            text_prompt = f'{self.vlm_image_tokens[self.QWEN2_VL]} Represent the given image with the following question: {text if text else "What is in the image?"}'
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            # Unsqueeze if needed
            if len(inputs['pixel_values'].shape) == 3:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
            if len(inputs['image_grid_thw'].shape) == 2:
                inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
        else:
            inputs = self.processor(text=text, images=None, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
        with torch.no_grad():
            if is_query:
                output = self.model(qry=inputs)["qry_reps"]
            else:
                output = self.model(tgt=inputs)["tgt_reps"]
        
        return output.cpu().numpy()

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_reps = []
        for t in texts:
            all_reps.append(self._encode_single(text=t, is_query=False))
        return np.concatenate(all_reps, axis=0)

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_reps = []
        for img in images:
            all_reps.append(self._encode_single(image=img, is_query=False))
        return np.concatenate(all_reps, axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode_single(text=query, is_query=True).flatten()

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        return self._encode_single(image=image, is_query=True).flatten()
