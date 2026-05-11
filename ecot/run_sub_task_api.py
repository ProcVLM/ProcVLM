"""
python ecot/run_sub_task_api.py --world_size 3 --rank 2
"""
import os
import logging
import argparse
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
from ecot.utils.sub_task_utils import producer_task, consumer_task
from core.backends.api import batch_generate
from core.utils.runner_utils import auto_split_datasets, gpu_worker

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
args = args.parse_args()
# ===== load datasets =====
_raw_datasets = args.dataset_paths


# ===== config =====
VLM_PATH = os.getenv('API_VLM_PATH')
BASE_URL = os.getenv('DASHSCOPE_API_URL')
OUTPUT_SUB_TASK_JSONL_DIR = os.getenv('OUTPUT_SUB_TASK_JSONL_DIR')
BATCH_SIZE = 1
REFRESH_FREQ = 1 # needs update #
# --- auto config dp mode ---
DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)


# ===== infer helpers =====
def infer_instance(videos: List[Tuple[str, int, int, int, int]], questions: List[str]):
    from core.backends.api import general_api_query
    results = []
    for video, question in zip(videos, questions):
        gt = general_api_query(
            video, question, 
            base_url=BASE_URL,
            model_name=VLM_PATH,
            api_key_namespace="DASHSCOPE_API_KEY",
        )
        results.append(gt)
    return results
# def infer_instance(images: List[str], questions: List[str]):
#     return batch_generate(
#         images, questions, url=VLM_PATH,
#         question_template="{task_desc}",
#     )


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = 10
    input_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path, OUTPUT_SUB_TASK_JSONL_DIR,
        WORLD_SIZE, RANK, REFRESH_FREQ,
    ))
    consumer = ctx.Process(target=consumer_task, args=(
        output_queue, dataset_path, OUTPUT_SUB_TASK_JSONL_DIR,
        REFRESH_FREQ,
        VLM_PATH.strip("/").split("/")[-1], # needs update #
    ))

    producer.start()
    consumer.start()
    gpu_worker(input_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    producer.join()
    consumer.join()


if __name__ == "__main__":
    for every_path in tqdm(DATASETS):
        # producer_task(None, every_path, OUTPUT_SUB_TASK_JSONL_DIR, WORLD_SIZE, RANK)
        process_dataset(every_path)