"""
python grounding/tests/test_sgllm.py
"""
import os
from dotenv import load_dotenv
from PIL import Image
from core.backends.sglang import get_sgllm
from core.models.qwenvl import batch_frame_list_generate_with_sgllm

qwenvl_kwargs = {
    "tp_size": 8,
    "ep_size": 2,
    "mem_fraction_static": 0.85,
    "context_length": 131072,
    "attention_backend": "fa3",
    "cuda_graph_max_bs": 16,
    "disable_custom_all_reduce": True,
    "chunked_prefill_size": 2048,
}

if __name__ == "__main__":
    load_dotenv()

    # warm up
    VLM_PATH = os.getenv("VLM_PATH_QWEN_INFER")
    llm, processor = get_sgllm(VLM_PATH, engine_kwargs=qwenvl_kwargs)

    dummy_img = Image.new("RGB", (24, 24), color="black")
    res = batch_frame_list_generate_with_sgllm(
        [[(1, dummy_img)]],
        ["you are a helpful assistant."],
        model_path=VLM_PATH,
    )
    
    answer = res[0]
    print("Answer:", answer)
    print("SGLang test completed.")
