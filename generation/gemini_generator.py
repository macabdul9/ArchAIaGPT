import os
import google.generativeai as genai
from typing import List, Optional, Any
from .base_generator import BaseGenerator
from config import SYSTEM_PROMPT

class GeminiGenerator(BaseGenerator):
    """Gemini model generator using Google API."""

    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )

    def generate(self, query: str, context: str, images: List[Any] = None) -> str:
        prompt = (
            "## Retrieved Archaeological Artifacts\n\n"
            f"{context}\n\n"
            "────────────────────────────────────────\n\n"
            f"## User Query\n\n{query}\n\n"
            "Please answer the query using ONLY the artifact evidence above. "
            "Cite artifact labels/IDs when referencing specific items."
        )
        # Handle images if provided for multimodal reasoning
        content = [prompt]
        if images:
            for img in images:
                content.append(img)
        
        response = self.model.generate_content(content)
        return response.text.strip()

    def generate_stream(self, query: str, context: str, images: List[Any] = None):
        prompt = (
            "## Retrieved Archaeological Artifacts\n\n"
            f"{context}\n\n"
            "────────────────────────────────────────\n\n"
            f"## User Query\n\n{query}\n\n"
            "Please answer the query using ONLY the artifact evidence above. "
            "Cite artifact labels/IDs when referencing specific items."
        )
        content = [prompt]
        if images:
            for img in images:
                content.append(img)
                
        response = self.model.generate_content(content, stream=True)
        accumulated = ""
        for chunk in response:
            if chunk.text:
                accumulated += chunk.text
                yield accumulated
