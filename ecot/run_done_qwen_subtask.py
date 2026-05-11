"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_done_qwen_subtask.py --world_size 2 --rank 0
"""

prompt_example = '''You are given two images of a robotic arm performing a task: The first image shows the initial state of the task (at the start). The second image shows the current state of the task (as it is now).
Output the completion criterion for the task based on the changes observed between the two images. Then use this criterion to determine whether the task has been completed successfully.
Output Format:
- A single sentence stating the functional and visual condition for task completion.
- A final line stating "YES" if the task is completed according to the criterion, or "NO" if it is not.
Now here are the two images and the task description: {instruction}'''

import logging
import os
import argparse
import pickle
import time
import numpy as np
import multiprocessing as mp
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import load_jsonlines, append_jsonlines, episode2num_token, draw_text_block, draw_bboxes, json_response_to_bboxes
from core.utils.runner_utils import gpu_worker, process_task

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)
logging.getLogger("lmdeploy").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING) 
logging.getLogger("transformers_modules").setLevel(logging.WARNING)

args = argparse.ArgumentParser()
args.add_argument("--rank", type=int, default=0)
args.add_argument("--world_size", type=int, default=1)
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = args.parse_args()
os.environ["RAY_TMPDIR"] = f"{os.getenv('CACHE_ROOT')}/ray_tmpdir_{args.rank}"
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 32 # needs update #
TENSOR_PARALLEL_SIZE = 8
VLM_PATH = os.getenv('VLM_PATH_QWEN_INFER')
DONE_JSONL_OUTPUT_DIR = os.getenv('OUTPUT_DONE_JSONL_DIR')
SUB_TASK_JSONL_DIR = os.getenv('ANNOTATION_SUB_TASK_DIR')
BBOX_JSONL_DIR = os.getenv('ANNOTATION_BBOX_DIR')
TASK_BORDER_DIR = os.getenv('OUTPUT_PLAN_JSONL_DIR')
REFRESH_FREQ = 514 # needs update #
# --- auto config dp ---
# DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
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
    "max_num_seqs": 64,
    # "max_num_batched_tokens": 1024,
    # "seed": 0,
    # "cpu_offload_gb": 0,
    # "kv_cache_memory_bytes": 6817551975,
    "distributed_executor_backend": "mp",
    "mm_encoder_tp_mode": "data",
}

def infer_instance(not_used_image_list: List[Image.Image | None], llm_inputs_batch: List[Dict[str, Any]]):
    from core.models.qwenvl import run_batch_frame_list_vllm
    return run_batch_frame_list_vllm(
        llm_inputs_batch, VLM_PATH,
        tp=TENSOR_PARALLEL_SIZE, engine_kwargs=qwenvl_kwargs
    )

def process_instance(videos: List[List[Tuple[int, Image.Image]]], questions: List[str]):
    from core.models.qwenvl import process_batch_frame_list_vllm
    return process_batch_frame_list_vllm(videos, questions, VLM_PATH)


# ===== pipeline helpers =====
def producer_task(input_queue: mp.Queue, dataset_path: str, rank: int, world_size: int, refresh_freq: int):
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            image_resize=(480, 480),
            return_record_meta=True,
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return
    logging.info(f"Dataset '{dataset.name}' loaded, producer started for rank {rank}/{world_size}.")

    # ----- start processing -----
    last_time = time.time()
    eps = dataset.all_episode_names
    total_frames = len(eps)
    for _tidx, ep_stem in enumerate(tqdm(eps, desc=f"[{dataset.name}] SD R{rank}/{world_size}", total=total_frames)):
        # if _tidx >= 8: # needs update #
        #     break
        
        if episode2num_token(ep_stem) % world_size != rank:
            continue
        
        # --- cache for .sub_task.jsonl ---
        pej = Path(SUB_TASK_JSONL_DIR) / dataset.name / f"{ep_stem}.sub_task.jsonl"
        if not pej.exists():
            logging.warning(f"Episode {ep_stem} in {dataset.name} does not have sub-task annotation file, skipping.")
            continue
        subtasks = load_jsonlines(pej)
        if len(subtasks) == 0:
            logging.warning(f"Episode {ep_stem} in {dataset.name} has empty sub-task annotation file, skipping.")
            continue

        # --- cache for .done.jsonl ---
        dej = Path(DONE_JSONL_OUTPUT_DIR) / dataset.name / f"{ep_stem}.done.jsonl"
        possible_ep_rid = set()
        if dej.exists():
            done_items = load_jsonlines(dej)
            for ditem in done_items:
                possible_ep_rid.add(ditem['frame_id'])

        subtask_borders = []
        last_subtask = ''
        for item in subtasks:
            if item['sub_task'] != last_subtask:
                subtask_borders.append((item['frame_id'], item['sub_task']))
                last_subtask = item['sub_task']
        epst, eped = dataset.get_episode_range(ep_stem)
        subtask_borders.append((eped - epst, '')) # end border

        for sti, (start_fid, sub_task) in enumerate(subtask_borders[:-1]):
            end_fid, _ = subtask_borders[sti + 1]

            slen = max(1, int(0.2 * (end_fid - start_fid)))
            sample_num = min(5, slen)
            sampled_fids = np.random.choice(range(end_fid - slen, end_fid), size=sample_num, replace=False)
            sampled_fids = sorted(sampled_fids.tolist())
            sampled_fids = [int(fid) for fid in sampled_fids]
            for fid in sampled_fids:
                # --- skip already done ---
                if fid in possible_ep_rid:
                    continue

                prompt = prompt_example.format(instruction=sub_task)

                image_front0 = dataset[epst + start_fid]['exo_image']
                image_front = dataset[epst + fid]['exo_image']
                image_list = [image_front0, image_front]

                # --- submit to input queue ---
                _supp = (start_fid, fid, sub_task, ep_stem)
                input_queue.put((image_list, prompt, _supp))

        # --- log progress ---
        if _tidx and _tidx % refresh_freq == 0:
            elapsed = time.time() - last_time
            speed = _tidx / elapsed
            rest_frames = total_frames - _tidx
            eta = rest_frames / speed if speed > 1e-5 else -1
            logging.info(f"Producer throughput: {speed:.2f} frames/s. Submitted {_tidx}/{total_frames} frames. ETA: {eta/3600:.2f} hours.")
    
    # end of input
    input_queue.put(None)
    return

# Post Process Model Output
def consumer_task(output_queue: mp.Queue, dataset_path: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    frame_count, success_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        images, question, generated_text, _supp = item
        start_fid, fid, sub_task, ep_stem = _supp

        generated_text = generated_text.lower().replace("\n", " ").strip()
        cot = generated_text.strip()
        if "yes" in generated_text:
            finished = True
            success_count += 1
        elif "no" in generated_text:
            finished = False
            success_count += 1
        else:
            finished = False

        done_item = {
            "start_frame_id": start_fid,
            "frame_id": fid,
            "finished": finished,
            "sub_task": sub_task,
            "cot": cot,
        }

        jsl_path = Path(DONE_JSONL_OUTPUT_DIR) / dataset_name / f"{ep_stem}.done.jsonl"
        jsl_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(done_item, jsl_path)

        frame_count += 1
        if frame_count % refresh_freq == 0:
            speed = refresh_freq / (time.time() - last_time)
            logging.info(
                f"Consumer throughput: {speed:.2f} frames/s. "
                f"Processed {frame_count} frames, {success_count} with bbox. "
                f"Failed rate: {(frame_count - success_count)/frame_count:.2%}."
            )
            last_time = time.time()

            # save sample image with bbox
            insp_path = Path(DONE_JSONL_OUTPUT_DIR) / dataset_name / 'inspect'
            insp_path.mkdir(parents=True, exist_ok=True)
            img0, img = images
            new_img = Image.new('RGB', (img0.width + img.width, max(img0.height, img.height)))
            new_img.paste(img0, (0, 0))
            new_img.paste(img, (img0.width, 0))
            # draw cot text on new_img
            new_img = draw_text_block(new_img, f"{sub_task} DONE: {finished}\nstart: {start_fid} end: {fid} cot: {cot}", bg_transparency=0.55)
            new_img.save(insp_path / f"{ep_stem}_{fid}.png")
    return


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = BATCH_SIZE * 20
    input_queue = ctx.Queue(maxsize=queue_max_size)
    mid_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path, 
        RANK, WORLD_SIZE,
        REFRESH_FREQ
    ))
    processor = ctx.Process(target=process_task, args=(
        input_queue, mid_queue, 
        process_instance, 
        BATCH_SIZE, REFRESH_FREQ
    ))
    consumer = ctx.Process(target=consumer_task, args=(
        output_queue, dataset_path, 
        REFRESH_FREQ
    ))

    producer.start()
    processor.start()
    consumer.start()
    gpu_worker(mid_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    producer.join()
    processor.join()
    consumer.join()

if __name__ == "__main__":
    # warm up
    # dummy_img = Image.new('RGB', (24, 24), color = 'black')
    # infer_instance([dummy_img], ["you are a helpful assistant."])
    
    for dataset_path in tqdm(DATASETS):
        process_dataset(dataset_path)
