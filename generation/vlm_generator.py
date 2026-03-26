import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
from PIL import Image
from typing import List, Optional, Any
from .base_generator import BaseGenerator
from config import SYSTEM_PROMPT

class VLMGenerator(BaseGenerator):
    """Local VLM generator using transformers (Qwen3-VL, InternVL3, Ovis2)."""

    def __init__(self, model_name: str, model_type: str, device: str = None, dtype: str = "bfloat16"):
        super().__init__(model_name, device)
        self.model_type = model_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        
        print(f"[VLMGenerator] Loading {model_name} ({model_type}) on {self.device}...")
        
        if self.model_type == "internvl3":
             self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device).eval()
             self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
             
        elif self.model_type == "qwen3-vl":
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            
        elif self.model_type == "ovis2":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                multimodal_max_length=32768,
                trust_remote_code=True
            ).to(self.device).eval()
            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
            
        else:
            # Generic loader
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, query: str, context: str, images: List[Image.Image] = None) -> str:
        prompt_text = (
            f"System: {SYSTEM_PROMPT}\n\n"
            "## Retrieved Archaeological Artifacts\n\n"
            f"{context}\n\n"
            "────────────────────────────────────────\n\n"
            f"## User Query\n\n{query}\n\n"
            "Please answer the query using ONLY the artifact evidence above. "
            "Cite artifact labels/IDs when referencing specific items."
        )

        if self.model_type == "qwen3-vl":
            messages = [{"role": "user", "content": []}]
            if images:
                for img in images:
                    messages[0]["content"].append({"type": "image", "image": img})
            messages[0]["content"].append({"type": "text", "text": prompt_text})
            
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return output_text[0]

        elif self.model_type == "internvl3":
            # Simplified chat for InternVL3
            # InternVL3 usually has a .chat() method if loaded via trust_remote_code
            if hasattr(self.model, "chat"):
                 pixel_values = None
                 if images:
                     # Just take the first image for now for InternVL3 .chat()
                     # In a real scenario, we'd handle multi-image
                     pass 
                 response = self.model.chat(self.processor, None, prompt_text, {"max_new_tokens": 512})
                 return response
            return "InternVL3 chat implementation pending"

        elif self.model_type == "ovis2":
             query = prompt_text
             if images:
                 query = "\n".join(["<image>"] * len(images)) + "\n" + query
             
             prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images if images else [])
             attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
             input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
             attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
             if pixel_values is not None:
                 pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]
             
             output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=512)[0]
             return self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return "Generation logic for this model type not fully implemented in transformer mode."
