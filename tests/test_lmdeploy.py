"""
python grounding/tests/test_lmdeploy.py
"""
import ray
import os
from dotenv import load_dotenv
from PIL import Image
from core.models.internvl import batch_generate_with_lmd

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, include_dashboard=False)

if __name__ == "__main__":
    load_dotenv()
    # warm up
    TENSOR_PARALLEL_SIZE = 2
    VLM_PATH = os.getenv("VLM_PATH_INTERN_INFER")
    dummy_img = Image.new('RGB', (24, 24), color='cyan')
    res = batch_generate_with_lmd([dummy_img], ["what is the color of this image"], tp=TENSOR_PARALLEL_SIZE, model_path=VLM_PATH, use_pytorch=True)
    answer = res[0]
    print("Answer:", answer)
    print("LMDeploy test completed.")