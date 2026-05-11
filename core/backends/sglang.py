from transformers import AutoProcessor
from typing import List, Optional, Union, Tuple, Dict, Any
from sglang import Engine

_cached_sgllm_instance = None
_cached_sgllm_processor = None

def get_sgllm_instance(model_path: str, engine_kwargs: Optional[Dict[str, Any]] = None):
    global _cached_sgllm_instance

    if _cached_sgllm_instance is None:
        runtime_kwargs = engine_kwargs or {}
        _cached_sgllm_instance = Engine(
            model_path=model_path,
            enable_multimodal=True,
            trust_remote_code=True,
            **runtime_kwargs
        )

    return _cached_sgllm_instance

def get_sgllm_processor(model_path: str):
    global _cached_sgllm_processor
    if _cached_sgllm_processor is None:
        _cached_sgllm_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return _cached_sgllm_processor

def get_sgllm(model_path: str, engine_kwargs: Optional[Dict[str, Any]] = None):
    inst = get_sgllm_instance(model_path, engine_kwargs)
    proc = get_sgllm_processor(model_path)
    return inst, proc