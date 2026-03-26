from .generator import Generator
from .gemini_generator import GeminiGenerator
from .vlm_generator import VLMGenerator

def get_generator(backend: str, model_name: str = None, api_key: str = None, base_url: str = None, device: str = None):
    backend = backend.lower()
    
    if backend == "openai":
        return Generator(backend="openai", model=model_name, api_key=api_key)
    elif backend == "vllm":
        return Generator(backend="vllm", model=model_name, base_url=base_url)
    elif backend == "gemini":
        return GeminiGenerator(model_name=model_name or "gemini-3-flash-preview", api_key=api_key)
    elif backend in ["qwen3-vl", "internvl3", "ovis2"]:
        # Use transformers for local loading
        return VLMGenerator(model_name=model_name, model_type=backend, device=device)
    else:
        # Fallback to general VLMGenerator if it's a model name
        return VLMGenerator(model_name=model_name, model_type="generic", device=device)
