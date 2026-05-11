import logging
import asyncio
import threading
import atexit
import random
from typing import List, Dict, Any, Optional
from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig
from lmdeploy.serve.vl_async_engine import VLAsyncEngine


# legacy version
_cached_lmdeploy_pipeline = None
_cached_lmdeploy_instance = None
_lmdeploy_loop = None
_lmdeploy_loop_thread = None

def get_lmdeploy_pipeline(model_path: str, tp: int = 1, dp: int = 1, use_pytorch: bool = False, model_len: int = 32768) -> Any:
    global _cached_lmdeploy_pipeline, _cached_lmdeploy_instance
    assert _cached_lmdeploy_instance is None, (
        "You are using the legacy lmdeploy pipeline while lmdeploy AsyncEngine instance"
        " has been created (via run_batch_async_lmdeploy). Please choose one backend only."
    )
    if _cached_lmdeploy_pipeline is None:
        if use_pytorch:
            backend_config = PytorchEngineConfig(session_len=model_len, tp=tp, dp=dp)
        else:
            backend_config = TurbomindEngineConfig(session_len=model_len, tp=tp, dp=dp)
        logging.info(f"Initializing LMDeploy pipeline for '{model_path}' with TP={tp}...")
        _cached_lmdeploy_pipeline = pipeline(
            model_path,
            backend_config=backend_config,
        )
        logging.info("LMDeploy pipeline initialized successfully.")
    return _cached_lmdeploy_pipeline



# new AsyncEngine version
def get_lmdeploy_instance(model_path: str, tp: int, use_pytorch: bool = False, other_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    global _cached_lmdeploy_instance, _cached_lmdeploy_pipeline
    assert _cached_lmdeploy_pipeline is None, (
        "You are using the lmdeploy AsyncEngine instance while legacy lmdeploy pipeline"
        " has been created (via get_lmdeploy_pipeline). Please choose one backend only."
    )
    if _cached_lmdeploy_instance is None:
        if other_kwargs is None: other_kwargs = {}
        loop = _get_or_create_lmdeploy_loop()
        def _init():
            if use_pytorch:
                engine_config = PytorchEngineConfig(
                    tp=tp,
                    **other_kwargs
                )
            else:
                engine_config = TurbomindEngineConfig(
                    tp=tp,
                    **other_kwargs
                )
            return VLAsyncEngine(model_path=model_path, backend_config=engine_config)
        future = asyncio.run_coroutine_threadsafe(asyncio.to_thread(_init), loop)
        _cached_lmdeploy_instance = future.result()
    return _cached_lmdeploy_instance

def shutdown_lmdeploy():
    global _lmdeploy_loop
    if _lmdeploy_loop is not None and _lmdeploy_loop.is_running():
        print("Shutting down lmdeploy AsyncEngine event loop...")
        _lmdeploy_loop.call_soon_threadsafe(_lmdeploy_loop.stop)

def _get_or_create_lmdeploy_loop():
    global _lmdeploy_loop, _lmdeploy_loop_thread
    if _lmdeploy_loop is None:
        _lmdeploy_loop = asyncio.new_event_loop()
        _lmdeploy_loop_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(_lmdeploy_loop), _lmdeploy_loop.run_forever()),
            daemon=True
        )
        _lmdeploy_loop_thread.start()
        atexit.register(shutdown_lmdeploy)
    return _lmdeploy_loop

async def _submit_and_wait_lmdeploy(engine, prompt: str, session_id: int, gen_config: GenerationConfig, input_kwargs: Dict) -> str:
    """use input_kwargs to include multi-model input data, like images=..."""
    async for outputs in engine.generate(prompt, session_id=session_id, gen_config=gen_config, **input_kwargs):
        final_output = outputs
    print(f"LMDeploy final output: {final_output}")
    text = final_output.response
    assert text, "LMDeploy returned empty response!"
    if '</think>' in text:
        answer = text.split('</think>')[1].strip()
    else:
        answer = text.strip()
    return answer

def run_batch_async_lmdeploy(engine, inputs: List[Dict]) -> List[str]:
    """
    inputs : [
        {"prompt": "...", "gen_config": GenerationConfig(...), "input_kwargs": {"images": [...]}}
    ]
    """
    loop = _get_or_create_lmdeploy_loop()
    async def batch_task():
        tasks = [
            _submit_and_wait_lmdeploy(
                engine=engine, 
                prompt=item["prompt"], 
                session_id=random.randint(1, 2**31 - 1),
                gen_config=item.get("gen_config", GenerationConfig()), 
                input_kwargs=item.get("input_kwargs", {})
            )
            for item in inputs
        ]
        return await asyncio.gather(*tasks)
    future = asyncio.run_coroutine_threadsafe(batch_task(), loop)
    return future.result()




# exports
from lmdeploy.vl.utils import encode_image_base64
from lmdeploy.vl.constants import IMAGE_TOKEN
