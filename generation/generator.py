"""
LLM answer generator — supports both vLLM (local) and OpenAI API
through the same OpenAI-compatible interface.
"""

import os
from typing import Optional

from openai import OpenAI

from config import (
    VLLM_BASE_URL, VLLM_MODEL, OPENAI_MODEL,
    MAX_NEW_TOKENS, TEMPERATURE, SYSTEM_PROMPT,
)


class Generator:
    """
    Unified LLM generator.

    backend="vllm"   → connects to a local vLLM server (OpenAI-compatible API)
    backend="openai" → connects to OpenAI API
    """

    def __init__(
        self,
        backend:    str  = "openai",
        model:      str  = None,
        base_url:   str  = None,
        api_key:    str  = None,
        max_tokens: int  = None,
        temperature: float = None,
    ):
        self.backend     = backend
        self.max_tokens  = max_tokens  or MAX_NEW_TOKENS
        self.temperature = temperature if temperature is not None else TEMPERATURE

        if backend == "vllm":
            self.model   = model   or VLLM_MODEL
            self.base_url = base_url or VLLM_BASE_URL
            self.api_key = api_key or "EMPTY"
            self.client  = OpenAI(base_url=self.base_url, api_key=self.api_key)
            print(f"[Generator] vLLM backend → {self.base_url} / {self.model}")

        elif backend == "openai":
            self.model   = model or OPENAI_MODEL
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set. Pass api_key= or set env var.")
            self.client = OpenAI(api_key=self.api_key)
            print(f"[Generator] OpenAI backend → {self.model}")

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'openai'.")

    def generate(
        self,
        query:   str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a grounded answer.

        Parameters
        ----------
        query          : the user's natural-language question
        context        : formatted artifact context block (from context_builder)
        system_prompt  : override the default system prompt

        Returns
        -------
        Answer string from the LLM.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT

        user_message = (
            "## Retrieved Archaeological Artifacts\n\n"
            f"{context}\n\n"
            "────────────────────────────────────────\n\n"
            f"## User Query\n\n{query}\n\n"
            "Please answer the query using ONLY the artifact evidence above. "
            "Cite artifact labels/IDs when referencing specific items."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"[Generator ERROR] {type(e).__name__}: {e}"
            print(error_msg)
            return error_msg

    def generate_stream(
        self,
        query:   str,
        context: str,
        system_prompt: Optional[str] = None,
    ):
        """
        Streaming version of generate(). Yields partial answer strings
        as tokens arrive from the LLM.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT

        user_message = (
            "## Retrieved Archaeological Artifacts\n\n"
            f"{context}\n\n"
            "────────────────────────────────────────\n\n"
            f"## User Query\n\n{query}\n\n"
            "Please answer the query using ONLY the artifact evidence above. "
            "Cite artifact labels/IDs when referencing specific items."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_message},
        ]

        try:
            stream = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                stream      = True,
            )
            accumulated = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta
                    yield accumulated

        except Exception as e:
            error_msg = f"[Generator ERROR] {type(e).__name__}: {e}"
            print(error_msg)
            yield error_msg
