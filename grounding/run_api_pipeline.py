"""
export http_proxy=http://httpproxy.glm.ai:3128
python grounding/run_api_pipeline.py
"""

import logging
import pickle
import argparse
import time
import multiprocessing as mp
import os
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from core.utils.common import (
    _pil_to_base64,
    json_response_to_bboxes,
    draw_bboxes,
)
from core.backends.api import (
    batch_generate,
    generated_texts_to_json_responses,
)
from core.data.reader import FastLerobotVLReader

# ===== config =====
BATCH_SIZE = 16
load_dotenv()
API_URL = os.getenv('API_URL')
OUTPUT_BBOX_JSONL_DIR = os.getenv('OUTPUT_BBOX_JSONL_DIR')
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
INFER_TYPE = 'single'  # 'single' or 'multi'
REFRESH_FREQ = min(5144, 114 * BATCH_SIZE + 514)  # show log and save pkl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)
logging.getLogger("lmdeploy").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING) 
logging.getLogger("transformers_modules").setLevel(logging.WARNING)


# ===== infer helpers =====
def infer_instance(images: List[str], task_descriptions: List[str]):
    return batch_generate(images, task_descriptions, API_URL)


# ===== pipeline helpers =====
from core.utils.runner_utils import gpu_worker

# Prepare Model Input
def producer_task(input_queue: mp.Queue, dataset_path: str, ego_mapper: Dict, third_mapper: Dict, output_bbox_dir: str, refresh_freq: int):
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            ego_name=ego_mapper.get(Path(dataset_path).name, None),
            exo_name=third_mapper.get(Path(dataset_path).name, None)
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return

    # -----  try to recover processed record -----
    output_bbox_dir = Path(output_bbox_dir) # pkl name format: '{dataset_name}_{rank}.pkl'
    possible_processed_records_paths = list(output_bbox_dir.glob(f"{dataset.name}_*.pkl"))
    processed_base64_set = set()
    for pkl_path in possible_processed_records_paths:
        base64_to_json_response = pickle.load(open(pkl_path, 'rb'))
        processed_base64_set.update(set(base64_to_json_response.keys()))

    # ----- start processing -----
    last_time = time.time()
    total_frames = len(dataset) * 2 if dataset.ego_camera_key else len(dataset)
    frame_count = 0
    for data in tqdm(dataset, desc=f"[{dataset.name}] Producer"):
        task_desc = data['task_description']

        third_img = data['exo_image']
        third_img_base64 = _pil_to_base64(third_img)
        if third_img_base64 not in processed_base64_set:
            # input 1
            input_queue.put((third_img, task_desc, None))
            frame_count += 1
        
        if 'ego_image' in data:
            ego_img = data['ego_image']
            ego_img_base64 = _pil_to_base64(ego_img)
            if ego_img_base64 not in processed_base64_set:
                # input 2
                input_queue.put((ego_img, task_desc, None))
                frame_count += 1
        
        if frame_count and frame_count % refresh_freq == 0:
            elapsed = time.time() - last_time
            speed = refresh_freq / elapsed
            rest_frames = total_frames - frame_count
            eta = rest_frames / speed if speed > 1e-5 else -1
            logging.info(f"Producer throughput: {speed:.2f} frames/s. Submitted {frame_count}/{total_frames} frames. ETA: {eta/60:.2f} mins.")
            last_time = time.time()
    # end of input
    input_queue.put(None)
    return

# Post Process Model Output
def consumer_task(output_queue: mp.Queue, dataset_path: str, output_bbox_dir: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    output_bbox_dir = Path(output_bbox_dir)
    pkl_path = output_bbox_dir / f"{dataset_name}_0.pkl"

    if pkl_path.exists():
        base64_to_json_response = pickle.load(open(pkl_path, 'rb'))
    else:
        base64_to_json_response = {}
    
    frame_count, bbox_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        pil_img, task, generated_text, _ = item
        json_responses, _failed_indices = generated_texts_to_json_responses([generated_text], [pil_img])
        json_response = json_responses[0]
        img_base64 = _pil_to_base64(pil_img)
        base64_to_json_response[img_base64] = json_response
        if len(_failed_indices) == 0: 
            bbox_count += 1
        elif frame_count < refresh_freq: # save some failed samples
            failed_image_save_path = pkl_path.parent / f"{dataset_name}/failed_{frame_count}.png"
            failed_image_save_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(failed_image_save_path)
            failed_text_save_path = pkl_path.parent / f"{dataset_name}/failed_tasks.txt"
            with open(failed_text_save_path, 'a') as f:
                f.write(f"Frame {frame_count}\nTask: {task}\nGenerated Text: {generated_text}\n\n")
        frame_count += 1
        if frame_count % refresh_freq == 0:
            with open(pkl_path, 'wb') as f:
                pickle.dump(base64_to_json_response, f)
            speed = refresh_freq / (time.time() - last_time)
            logging.info(
                f"Consumer throughput: {speed:.2f} frames/s. "
                f"Processed {frame_count} frames, {bbox_count} with bbox. "
                f"Failed rate: {(frame_count - bbox_count)/frame_count:.2%}."
            )
            last_time = time.time()
            # save sample image with bbox
            image_with_bbox = draw_bboxes(
                pil_img,
                json_response_to_bboxes(json_response),
                show_object_name=True
            )
            image_with_bbox_save_path = pkl_path.parent / f"{dataset_name}/0_{frame_count}.png"
            image_with_bbox_save_path.parent.mkdir(parents=True, exist_ok=True)
            image_with_bbox.save(image_with_bbox_save_path)

    with open(pkl_path, 'wb') as f:
        pickle.dump(base64_to_json_response, f)
    logging.info(f"Saved bbox for {dataset_name} to {pkl_path}. This file has {len(base64_to_json_response)} records.")
    return


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = BATCH_SIZE
    input_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path, 
        EGO_CAMERA_MAP, EXO_CAMERA_MAP, 
        OUTPUT_BBOX_JSONL_DIR,
        REFRESH_FREQ
    ))
    consumer = ctx.Process(target=consumer_task, args=(
        output_queue, dataset_path, 
        OUTPUT_BBOX_JSONL_DIR,
        REFRESH_FREQ
    ))

    producer.start()
    consumer.start()
    gpu_worker(input_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    producer.join()
    consumer.join()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = args.parse_args()
    DATASETS = args.dataset_paths
    
    for dataset_path in DATASETS:
        process_dataset(dataset_path)
