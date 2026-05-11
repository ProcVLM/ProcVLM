"""
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_sub_task.py --world_size 2 --rank 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_sub_task.py --world_size 2 --rank 1
"""

import logging
import argparse
import ray
import os
import time
from dotenv import load_dotenv
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from ray.util.queue import Queue
from ray.experimental.tqdm_ray import tqdm
from ecot.utils.sub_task_utils import producer_task, consumer_task
from core.models.qwenvl import process_batch_frame_list_vllm, run_batch_frame_list_vllm
from core.utils.runner_utils import auto_split_datasets, gpu_worker, process_task

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)
logging.getLogger("lmdeploy").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING) 
logging.getLogger("transformers_modules").setLevel(logging.WARNING)

args = argparse.ArgumentParser()
args.add_argument("--rank", type=int, default=0, help="rank id for current process")
args.add_argument("--world_size", type=int, default=1, help="total number of processes")
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args.add_argument("--device", type=int, default=None, help="cuda device id to use")
args.add_argument("--enable_auto_dp", action='store_true', help="whether to enable automatic data parallel splitting based on world size")
args = args.parse_args()
if args.device is not None:
    logging.warning(f"Setting CUDA_VISIBLE_DEVICES to {args.device} as per argument.")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
os.environ["RAY_TMPDIR"] = f"{os.getenv('CACHE_ROOT')}/ray_tmpdir_{args.rank}"
# ===== dataset config =====
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 12 # needs update #
TENSOR_PARALLEL_SIZE = 8
VLM_PATH = os.getenv('VLM_PATH_QWEN_INFER')
OUTPUT_SUB_TASK_JSONL_DIR = os.getenv('OUTPUT_SUB_TASK_JSONL_DIR')
REFRESH_FREQ = 1145 * 5 # needs update #
# --- auto config dp mode ---
if args.enable_auto_dp:
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
else:
    DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank


# ===== infer helpers =====
qwenvl_kwargs = {
    "quantization": "fp8",
    "gpu_memory_utilization": 0.85,
    # "enforce_eager": True,
    "enable_expert_parallel": True,
    # "disable_cuda_graph": True,
    # "dtype": "bfloat16",
    "max_model_len": 131072,
    "max_num_seqs": 128,
    # "max_num_batched_tokens": 1024,
    # "seed": 0,
    # "cpu_offload_gb": 0,
    # "kv_cache_memory_bytes": 6817551975,
    "distributed_executor_backend": "mp",
    "mm_encoder_tp_mode": "data",
    "disable_custom_all_reduce": True,
}

def infer_instance(not_used_image_list: List[Image.Image | None], llm_inputs_batch: List[Dict[str, Any]]):
    return run_batch_frame_list_vllm(
        llm_inputs_batch, VLM_PATH,
        tp=TENSOR_PARALLEL_SIZE, engine_kwargs=qwenvl_kwargs
    )

def process_instance(videos: List[List[Tuple[int, Image.Image]]], questions: List[str]):
    return process_batch_frame_list_vllm(videos, questions, VLM_PATH)


# ===== main =====
@ray.remote
def producer_task_remote(*args):
    return producer_task(*args)

@ray.remote
def process_task_remote(*args):
    return process_task(*args)

@ray.remote
def consumer_task_remote(*args):
    return consumer_task(*args)

def process_dataset(dataset_path: str):
    # sequential delay to avoid possible ray init conflict
    time.sleep(args.rank * 5)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    queue_max_size = max(20, BATCH_SIZE * 2)
    input_queue = Queue(maxsize=queue_max_size)
    mid_queue = Queue(maxsize=queue_max_size)
    output_queue = Queue(maxsize=queue_max_size)
    # submit tasks remotely
    producer_ref = producer_task_remote.remote(
        input_queue, dataset_path, 
        OUTPUT_SUB_TASK_JSONL_DIR,
        WORLD_SIZE, RANK,
        REFRESH_FREQ
    )
    processor_ref = process_task_remote.remote(
        input_queue, mid_queue, 
        process_instance, 
        BATCH_SIZE, REFRESH_FREQ
    )
    consumer_ref = consumer_task_remote.remote(
        output_queue, dataset_path, 
        OUTPUT_SUB_TASK_JSONL_DIR,
        REFRESH_FREQ,
        # f'{VLM_PATH.strip("/").split("/")[-1]}_v2_b2_wdone_wstart', # needs update #
    )
    # start gpu worker locally
    gpu_worker(mid_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    # wait for all to complete
    ray.get([producer_ref, processor_ref, consumer_ref])


if __name__ == "__main__":
    # warm up
    infer_instance([[]], [])

    for every_path in tqdm(DATASETS):
        # producer_task(None, every_path, OUTPUT_SUB_TASK_JSONL_DIR, WORLD_SIZE, RANK)
        process_dataset(every_path)