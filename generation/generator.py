"""
Answer generation module for ArchAIaGPT.
Supports both local inference servers (vLLM) and commercial APIs (OpenAI/Gemini)
via a unified OpenAI-compatible client interface.
"""

import os
from typing import Optional, Generator as PyGenerator
from openai import OpenAI

from config import (
    VLLM_BASE_URL, VLLM_MODEL, OPENAI_MODEL,
    MAX_NEW_TOKENS, TEMPERATURE, SYSTEM_PROMPT,
)

class Generator:
    """
    Unified LLM interface for RAG task completion.
    Connects to specified backends (vLLM, OpenAI, etc.) and handles
    parameter adjustments for specific model variants (e.g., GPT-5).
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ):
        self.backend = backend
        self.max_tokens = max_tokens or MAX_NEW_TOKENS
        self.temperature = temperature if temperature is not None else TEMPERATURE

        if backend == "vllm":
            self.model = model or VLLM_MODEL
            self.base_url = base_url or VLLM_BASE_URL
            self.api_key = api_key or "EMPTY"
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            print(f"Generator initialised using vLLM backend: {self.base_url} ({self.model})")

        elif backend == "openai":
            self.model = model or OPENAI_MODEL
            # Look for API keys in the environment using several standard naming conventions
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY_CODEREASONING") or os.getenv("OPENAI_KEY_ZW")
            
            # Local fallback attempt for .env files if the environment isn't fully saturated
            if not self.api_key:
                try:
                    from dotenv import load_dotenv
                    dotenv_path = os.path.join(os.getcwd(), "..", ".env")
                    load_dotenv(dotenv_path)
                    self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY_CODEREASONING") or os.getenv("OPENAI_KEY_ZW")
                except ImportError:
                    pass

            if not self.api_key:
                raise ValueError("Could not locate an OpenAI API key in the environment.")
            
            self.client = OpenAI(api_key=self.api_key)
            print(f"Generator initialised using OpenAI backend: {self.model}")

        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'openai'.")

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generates a grounded answer based on archaeological evidence.
        """
        active_sys_prompt = system_prompt or SYSTEM_PROMPT

        # Construct the grounded prompt with the retrieval context
        user_message_content = (
            "## Archaeological Context\n\n"
            f"{context}\n\n"
            "## User Query\n\n"
            f"{query}\n\n"
            "Instructions: Provide a detailed answer based strictly on the provided artifact metadata. "
            "Internal citations for specific artifacts are mandatory."
        )

        messages = [
            {"role": "system", "content": active_sys_prompt},
            {"role": "user", "content": user_message_content},
        ]

        inference_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Specific model handling: newer variants like GPT-5-Nano may reject temperature
        if "gpt-5" not in self.model.lower():
            inference_params["temperature"] = self.temperature
            inference_params["max_tokens"] = self.max_tokens

        try:
            completion = self.client.chat.completions.create(**inference_params)
            return (completion.choices[0].message.content or "").strip()

        except Exception as e:
            # Automatic retry logic for models that do not support temperature parameters
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                print(f"Model {self.model} does not support temperature. Retrying with defaults...")
                inference_params.pop("temperature", None)
                try:
                    completion = self.client.chat.completions.create(**inference_params)
                    return (completion.choices[0].message.content or "").strip()
                except Exception as retry_error:
                    return f"Generator error (retry failed): {retry_error}"
            
            return f"Generator error: {e}"

    def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> PyGenerator[str, None, None]:
        """
        Streaming variant for token-by-token response display in UIs.
        """
        active_sys_prompt = system_prompt or SYSTEM_PROMPT
        user_message_content = (
            "## Archaeological Context\n\n"
            f"{context}\n\n"
            "## User Query\n\n"
            f"{query}\n\n"
            "Instructions: Answer using exactly the provided evidence."
        )

        messages = [
            {"role": "system", "content": active_sys_prompt},
            {"role": "user", "content": user_message_content},
        ]

        stream_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature
        }
        
        # Adjust token limit parameters for different model generations
        if "gpt-5" in self.model.lower():
            stream_params["max_completion_tokens"] = self.max_tokens
        else:
            stream_params["max_tokens"] = self.max_tokens

        try:
            try:
                inference_stream = self.client.chat.completions.create(**stream_params)
            except Exception as e:
                if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                    stream_params.pop("temperature", None)
                    inference_stream = self.client.chat.completions.create(**stream_params)
                else:
                    raise e

            accumulated_text = ""
            for chunk in inference_stream:
                token = chunk.choices[0].delta.content
                if token:
                    accumulated_text += token
                    yield accumulated_text

        except Exception as e:
            yield f"Stream generator error: {e}"
