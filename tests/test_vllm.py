"""
python grounding/tests/test_vllm.py
"""
import ray
import os
from dotenv import load_dotenv
from PIL import Image
from core.models.qwenvl import batch_frame_list_generate_with_vllm, get_vllm

if __name__ == "__main__":
    load_dotenv()
    # warm up
    TENSOR_PARALLEL_SIZE = 8
    VLM_PATH = os.getenv("VLM_PATH_QWEN_INFER")
    qwenvl_kwargs = {
        "quantization": "fp8",
        "gpu_memory_utilization": 0.85,
        # "enforce_eager": True,
        "enable_expert_parallel": True,
        # "disable_cuda_graph": True,
        # "dtype": "bfloat16",
        "max_model_len": 131072,
        "max_num_seqs": 192,
        # "max_num_batched_tokens": 1024,
        # "seed": 0,
        # "cpu_offload_gb": 0,
        # "kv_cache_memory_bytes": 6817551975,
        "distributed_executor_backend": "mp",
        "mm_encoder_tp_mode": "data",
        "disable_custom_all_reduce": True,
    }
    # llm, processor = get_vllm(VLM_PATH, TENSOR_PARALLEL_SIZE)
    dummy_img1 = Image.new('RGB', (24, 24), color='cyan')
    dummy_img2 = Image.new('RGB', (24, 24), color='pink')
    res = batch_frame_list_generate_with_vllm(
        [[(1, dummy_img1), (2, dummy_img2)]], ["How many pictures are there? What colors are they respectively?"], 
        tp=TENSOR_PARALLEL_SIZE, model_path=VLM_PATH, engine_kwargs=qwenvl_kwargs
    )
    answer = res[0]
    print("Answer:", answer)
    print("VLLM test completed.")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)