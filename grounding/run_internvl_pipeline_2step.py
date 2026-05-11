"""
CUDA_VISIBLE_DEVICES=3,5 python grounding/run_internvl_pipeline.py --world_size 1 --rank 0

export PYTHONPATH=.:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 \
    python grounding/run_internvl_pipeline.py --world_size 12 --rank 0

CUDA_VISIBLE_DEVICES=2,3 \
    python grounding/run_internvl_pipeline.py --world_size 12 --rank 1

CUDA_VISIBLE_DEVICES=4,5 \
    python grounding/run_internvl_pipeline.py --world_size 12 --rank 2

CUDA_VISIBLE_DEVICES=6,7 \
    python grounding/run_internvl_pipeline.py --world_size 12 --rank 3
"""

import argparse
import logging
import os
import time
import multiprocessing as mp
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from core.utils.common import (
    generated_texts_to_json_responses,
    json_response_to_bboxes,
    draw_bboxes,
    episode2num_token,
    load_jsonlines,
    append_jsonlines,
    check_data_sim,
)
from core.utils.runner_utils import auto_split_datasets
from core.data.reader import FastLerobotVLReader
from core.data.generals import (
    EGO_CAMERA_MAP, 
    EXO_CAMERA_MAP
)
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
# ===== dataset paths =====
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 40
TENSOR_PARALLEL_SIZE = 2
VLM_PATH = os.getenv("VLM_PATH")
OUTPUT_BBOX_JSONL_DIR = os.getenv("OUTPUT_BBOX_JSONL_DIR")
SUB_TASK_JSONL_DIR = os.getenv("ANNOTATION_SUB_TASK_DIR")
REFRESH_FREQ = min(5144, 114 * BATCH_SIZE + 514) # show log and save pkl
# --- auto config dp ---
# DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank


# ===== infer helpers =====
def infer_instance(images: List[str], task_descriptions: List[str], return_all_turns: bool = False):
    from core.models.internvl import batch_multiturn_generate_with_lmd
    return batch_multiturn_generate_with_lmd(
        images, task_descriptions, VLM_PATH,
        max_tokens=1024, temperature=0.0,
        return_all_turns=return_all_turns,
        tp=TENSOR_PARALLEL_SIZE,
        use_pytorch=True,
    )


# ===== pipeline helpers =====
# GPU Worker (int main process)
def gpu_worker(input_queue: mp.Queue, output_queue: mp.Queue, infer_func: callable, batch_size: int, refresh_freq: int):
    start_time = last_time = time.time()
    total_count = 0
    batch: List[Tuple[Image.Image, str, Any]] = []

    def flush_batch():
        nonlocal total_count, last_time
        if not batch:
            return
        imgs, qs, supps = zip(*batch)
        answers_turn1, generated_texts = infer_func(list(imgs), list(qs), return_all_turns=True)
        supps = [supp + [a1] for supp, a1 in zip(supps, answers_turn1)]
        for img, q, res, supp in zip(imgs, qs, generated_texts, supps):
            output_queue.put((img, q, res, supp))
            total_count += 1
            if total_count % refresh_freq == 0:
                elapsed = time.time() - last_time
                speed = refresh_freq / elapsed if elapsed > 0 else float('inf')
                speed_mtpd = speed * 86400 / 1e6
                logging.info(
                    f"GPU worker throughput: {speed:.2f} it/s ({speed_mtpd:.2f} mits/d). "
                    f"Processed {total_count} items."
                )
                last_time = time.time()
        batch.clear()

    while True:
        item = input_queue.get()
        if item is None:
            # Flush remaining items, signal end, log summary
            flush_batch()
            output_queue.put(None)
            elapsed = time.time() - start_time
            final_speed = total_count / elapsed if elapsed > 0 else float('inf')
            speed_mtpd = final_speed * 86400 / 1e6
            logging.info(f"GPU worker done: processed {total_count} items, avg throughput is {final_speed:.2f} it/s ({speed_mtpd:.2f} mits/d) over {elapsed:.2f}s.")
            break
        batch.append(item)
        if len(batch) >= batch_size:
            flush_batch()

# Prepare Model Input
def producer_task(input_queue: mp.Queue, dataset_path: str, ego_mapper: Dict, third_mapper: Dict, output_bbox_dir: str, rank: int, world_size: int, refresh_freq: int):
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            ego_name=ego_mapper.get(Path(dataset_path).name, None),
            exo_name=third_mapper.get(Path(dataset_path).name, None),
            image_resize=(480, 480),
            return_record_meta=True,
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return
    logging.info(f"Dataset '{dataset.name}' loaded, producer started for rank {rank}/{world_size}.")

    # ----- fast skip processed parquets -----
    skip_episodes = []
    for ep_stem in dataset.all_episode_names:
        if episode2num_token(ep_stem) % world_size != rank:
            skip_episodes.append(ep_stem)
            continue
        pej = Path(output_bbox_dir) / dataset.name / f"{ep_stem}.bbox.jsonl"
        if not pej.exists():
            break
        _jf = load_jsonlines(pej)
        if all('ref' in item.keys() for item in _jf):
            break
        skip_episodes.append(ep_stem)
    if len(skip_episodes) > 1:
        dataset.set_skip_episode(skip_episodes[:-1])

    # ----- start processing -----
    possible_ep_jsonl = ''
    possible_ep_third, possible_ep_ego = [], []
    last_third, last_ego = None, None # last image global id that was processed
    possible_subtask_jsonl = ''
    sub_task_map = {}

    last_time = time.time()
    total_frames = len(dataset)
    for g_id, data in tqdm(enumerate(dataset), desc=f"[{dataset.name}] BBOX R{rank}/{world_size}", total=total_frames):
        frame_id = int(data['frame_index'])
        ep_stem = data['episode_name']
        third_img, ego_img = data['exo_image'], data.get('ego_image', None)
        
        # --- skip episodes not assigned to this rank ---
        # test_eps = [
        #     'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
        #     'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
        #     'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
        #     'episode_006678', 'episode_004399'
        # ]
        # if ep_stem not in test_eps: # needs update #
        #     continue

        if episode2num_token(ep_stem) % world_size != rank:
            continue
        
        # --- cache for .bbox.jsonl ---
        pej = Path(output_bbox_dir) / dataset.name / f"{ep_stem}.bbox.jsonl"
        if pej != possible_ep_jsonl:
            possible_ep_jsonl = pej
            if pej.exists():
                _jf = load_jsonlines(pej)
                possible_ep_third = [item['frame_id'] for item in _jf if item['camera_key'] == dataset.exo_camera_key]
                possible_ep_ego = [item['frame_id'] for item in _jf if item['camera_key'] == dataset.ego_camera_key]
            else:
                possible_ep_third, possible_ep_ego = [], []

            last_third, last_ego = None, None # reset last ids for new episode

        # --- cache for .sub_task.jsonl ---
        stj = Path(SUB_TASK_JSONL_DIR) / dataset.name / f"{ep_stem}.sub_task.jsonl"
        if stj != possible_subtask_jsonl:
            possible_subtask_jsonl = stj
            sub_task_map = {}
            if stj.exists():
                st_items = load_jsonlines(stj)
                for st_item in st_items:
                    sub_task_map[st_item['frame_id']] = st_item['sub_task']

        # --- check if all processed ---
        t_not_in, e_not_in = (frame_id not in possible_ep_third), (frame_id not in possible_ep_ego) if ego_img else False
        if not t_not_in and not e_not_in:
            last_third = g_id
            if ego_img: last_ego = g_id
            continue
            
        # --- skip highly similar frames ---
        if not last_third or not check_data_sim(data, dataset[last_third], img_key='exo_image'):
            last_third = g_id # this frame is different enough and needs processing
        if ego_img and (not last_ego or not check_data_sim(data, dataset[last_ego], img_key='ego_image')):
            last_ego = g_id # this frame is different enough and needs processing

        if frame_id in sub_task_map:
            task_desc = sub_task_map[frame_id]
        else:
            task_desc = data['task_description']

        # --- submit to input queue ---
        if (last_third == g_id) and t_not_in:
            # input 1
            input_queue.put((third_img, task_desc, [frame_id, dataset.exo_camera_key, ep_stem]))
        else:
            if t_not_in: # output reference, but not input
                append_jsonlines({
                    "frame_id": frame_id,
                    "camera_key": dataset.exo_camera_key,
                    "ref": int(dataset[last_third]['frame_index'])
                }, pej)
        
        if ego_img and (last_ego == g_id) and e_not_in:
            # input 2
            input_queue.put((ego_img, task_desc, [frame_id, dataset.ego_camera_key, ep_stem]))
        else:
            if ego_img and e_not_in: # output reference, but not input
                append_jsonlines({
                    "frame_id": frame_id,
                    "camera_key": dataset.ego_camera_key,
                    "ref": int(dataset[last_ego]['frame_index'])
                }, pej)
        
        # --- log progress ---
        if g_id and g_id % refresh_freq == 0:
            elapsed = time.time() - last_time
            speed = g_id / elapsed
            rest_frames = total_frames - g_id
            eta = rest_frames / speed if speed > 1e-5 else -1
            logging.info(f"Producer throughput: {speed:.2f} frames/s. Submitted {g_id}/{total_frames} frames. ETA: {eta/3600:.2f} hours.")

    # end of input
    input_queue.put(None)
    return

# Post Process Model Output
def consumer_task(output_queue: mp.Queue, dataset_path: str, output_bbox_dir: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    output_bbox_dir = Path(output_bbox_dir)

    frame_count, bbox_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        pil_img, task, generated_text, _supps = item
        r_idx, cam_key, ep_stem, answer_turn1 = _supps

        json_responses, _failed_indices = generated_texts_to_json_responses([generated_text], [pil_img])
        json_response = json_responses[0]
        bbox_item = {
            "frame_id": r_idx,
            "camera_key": cam_key,
            "bboxes": json_response,
            "task_desc": task,
            "answer_turn1": answer_turn1,
            "generated_text": generated_text,
        }
        jsl_path = output_bbox_dir / dataset_name / f"{ep_stem}.bbox.jsonl"
        jsl_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(bbox_item, jsl_path)

        if len(_failed_indices) == 0: 
            bbox_count += 1
        elif frame_count < refresh_freq: # save some failed samples
            failed_image_save_path = jsl_path.parent / f"failed/{ep_stem}_{r_idx}_failed.png"
            failed_image_save_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(failed_image_save_path)
       
        frame_count += 1
        if frame_count % refresh_freq == 0:
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
            image_with_bbox_save_path = jsl_path.parent / f"{ep_stem}_{r_idx}.png"
            image_with_bbox.save(image_with_bbox_save_path)
    return


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = BATCH_SIZE * 20
    input_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path, 
        EGO_CAMERA_MAP, EXO_CAMERA_MAP, 
        OUTPUT_BBOX_JSONL_DIR,
        RANK, WORLD_SIZE,
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
    # warm up
    dummy_img = Image.new('RGB', (14, 14), color = 'black')
    infer_instance([dummy_img], ["you are a helpful assistant."])
    
    for dataset_path in tqdm(DATASETS):
        process_dataset(dataset_path)