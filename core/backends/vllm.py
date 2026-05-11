import uuid
import atexit
import asyncio
import threading
from vllm import AsyncLLMEngine, AsyncEngineArgs, LLM
from transformers import AutoProcessor
from typing import List, Optional, Union, Tuple, Dict, Any

_cached_vllm_instance = None
_cached_vllm_processor = None
_vllm_loop = None
_vllm_loop_thread = None

def shutdown_vllm():
    global _vllm_loop, _cached_vllm_instance
    if _vllm_loop is not None and _vllm_loop.is_running():
        print("Shutting down vLLM AsyncEngine event loop...")
        _vllm_loop.call_soon_threadsafe(_vllm_loop.stop)

def _get_or_create_loop():
    global _vllm_loop, _vllm_loop_thread
    if _vllm_loop is None:
        _vllm_loop = asyncio.new_event_loop()
        _vllm_loop_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(_vllm_loop), _vllm_loop.run_forever()),
            daemon=True
        )
        _vllm_loop_thread.start()
        atexit.register(shutdown_vllm)
    return _vllm_loop

def get_vllm_instance(model_path: str, tp: int, engine_kwargs: Optional[Dict[str, Any]] = None, disable_async: bool = False) -> Any:
    global _cached_vllm_instance
    if _cached_vllm_instance is None:
        runtime_kwargs = engine_kwargs or {}
        if not disable_async:
            loop = _get_or_create_loop()
            def _init():
                engine_args = AsyncEngineArgs(
                    model=model_path,
                    tensor_parallel_size=tp,
                    trust_remote_code=True,
                    **runtime_kwargs
                )
                return AsyncLLMEngine.from_engine_args(engine_args)
            future = asyncio.run_coroutine_threadsafe(asyncio.to_thread(_init), loop) # init in a separate thread
            _cached_vllm_instance = future.result()
        else:
            _cached_vllm_instance = LLM(
                model=model_path,
                tensor_parallel_size=tp,
                trust_remote_code=True,
                **runtime_kwargs
            )
    return _cached_vllm_instance

def get_vllm_processor(model_path: str) -> Any:
    global _cached_vllm_processor
    if _cached_vllm_processor is None:
        _cached_vllm_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return _cached_vllm_processor

def get_vllm(model_path: str, tp: int, engine_kwargs: Optional[Dict[str, Any]] = None, disable_async: bool = False) -> Tuple[Any, Any]:
    """
    Return a cached vLLM instance. Load model only once.
    """
    global _cached_vllm_instance, _cached_vllm_processor
    if _cached_vllm_instance is None or _cached_vllm_processor is None:
        _cached_vllm_instance = get_vllm_instance(model_path, tp, engine_kwargs, disable_async=disable_async)
        _cached_vllm_processor = get_vllm_processor(model_path)
    return _cached_vllm_instance, _cached_vllm_processor

async def _submit_and_wait(engine, prompt: str, mm_data: Dict, mm_kwargs: Dict, sampling_params) -> str:
    request_id = str(uuid.uuid4())
    vllm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    if mm_kwargs:
        vllm_inputs["mm_processor_kwargs"] = mm_kwargs
    results_generator = engine.generate(
        vllm_inputs,
        sampling_params, 
        request_id 
    )
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    text = final_output.outputs[0].text
    if '</think>' in text:
        answer = text.split('</think>')[1].strip()
    else:
        answer = text.strip()
    return answer

def run_batch_async(engine, inputs: List[Dict]):
    loop = _get_or_create_loop()
    async def batch_task():
        tasks = [
            _submit_and_wait(engine, item["prompt"], item["mm_data"], item["mm_kwargs"], item["sampling_params"])
            for item in inputs
        ]
        return await asyncio.gather(*tasks)
    future = asyncio.run_coroutine_threadsafe(batch_task(), loop)
    return future.result()



# exports
from vllm import SamplingParams