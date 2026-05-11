"""
CUDA_VISIBLE_DEVICES=0,1 python grounding/run_internvl_pipeline.py --world_size 4 --rank 0 &
CUDA_VISIBLE_DEVICES=2,3 python grounding/run_internvl_pipeline.py --world_size 4 --rank 1 &
CUDA_VISIBLE_DEVICES=4,5 python grounding/run_internvl_pipeline.py --world_size 4 --rank 2 &
CUDA_VISIBLE_DEVICES=6,7 python grounding/run_internvl_pipeline.py --world_size 4 --rank 3 &
wait

CUDA_VISIBLE_DEVICES=0,1 python grounding/run_internvl_pipeline.py --world_size 4 --rank 0 --load_all_cameras &
CUDA_VISIBLE_DEVICES=2,3 python grounding/run_internvl_pipeline.py --world_size 4 --rank 1 --load_all_cameras &
CUDA_VISIBLE_DEVICES=4,5 python grounding/run_internvl_pipeline.py --world_size 4 --rank 2 --load_all_cameras &
CUDA_VISIBLE_DEVICES=6,7 python grounding/run_internvl_pipeline.py --world_size 4 --rank 3 --load_all_cameras &
wait
"""

import argparse
import logging
import os
import time
import ray
from ray.util.queue import Queue
from ray.experimental.tqdm_ray import tqdm
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from core.utils.common import (
    prepare_grounding_questions,
    generated_texts_to_json_responses,
    json_response_to_bboxes,
    draw_bboxes,
    episode2num_token,
    load_jsonlines,
    append_jsonlines,
    check_data_sim,
)
from core.models.internvl import batch_generate_with_lmd
from core.utils.runner_utils import auto_split_datasets, gpu_worker
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
args.add_argument("--split_factor", type=int, choices=[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], default=None, help="Modulo for dataset splitting, used as num_tokens_factor in episode2num_token(). Set to different prime numbers for different runs to get different splits. If not set, no modulo is applied.")
args.add_argument("--load_all_cameras", action='store_true', help="whether to load all camera views in the dataset", default=False)
args.add_argument("--test_run", action='store_true', help="whether to run in test mode with limited data for quick checking", default=False)
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args.add_argument("--device", type=str, default=None, help="CUDA visible devices string.")
args.add_argument("--enable_auto_dp", action='store_true', help="whether to enable automatic data parallel splitting based on world size")
args = args.parse_args()
# --- check spilit factor ---
if args.split_factor is not None:
    assert args.split_factor > args.world_size, f"Split factor {args.split_factor} must be larger than world size {args.world_size}, otherwise some splits will be empty."
if args.device is not None:
    logging.warning(f"Setting CUDA_VISIBLE_DEVICES to {args.device} as per argument.")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
os.environ["RAY_TMPDIR"] = f"{os.getenv('CACHE_ROOT')}/ray_tmpdir_{args.rank}"
# ===== dataset paths =====
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 64
TENSOR_PARALLEL_SIZE = 2
VLM_PATH = os.getenv("VLM_PATH_INTERN_INFER")
if not args.test_run:
    OUTPUT_BBOX_JSONL_DIR = os.getenv("OUTPUT_BBOX_JSONL_DIR")
else:
    OUTPUT_BBOX_JSONL_DIR = os.getenv("OUTPUT_BBOX_JSONL_DIR") + '/test'
SUB_TASK_JSONL_DIR = os.getenv("ANNOTATION_SUB_TASK_DIR")
REFRESH_FREQ = 11451 # show log
SPLIT_FACTOR = args.split_factor
# --- auto config dp ---
if args.enable_auto_dp:
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
else:
    DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank


# ===== infer helpers =====
def infer_instance(images: List[str], task_descriptions: List[str]):
    images, questions = prepare_grounding_questions(images, task_descriptions)
    return batch_generate_with_lmd(
        images, questions, VLM_PATH,
        max_tokens=1024, temperature=0.0,
        tp=TENSOR_PARALLEL_SIZE,
        use_pytorch=True,
    )


# ===== pipeline helpers =====
# Prepare Model Input
def producer_task(input_queue: Queue, dataset_path: str, ego_mapper: Dict, third_mapper: Dict, output_bbox_dir: str, rank: int, world_size: int, refresh_freq: int):
    # ----- load dataset -----
    try:
        if isinstance(ego_mapper, str) and ego_mapper == 'load_all':
            dataset = FastLerobotVLReader(
                root=dataset_path,
                load_all_camera_keys=True,
                image_resize=(504, 504),
                return_record_meta=True,
            )
        else:
            dataset = FastLerobotVLReader(
                root=dataset_path,
                ego_name=ego_mapper.get(Path(dataset_path).name, None),
                exo_name=third_mapper.get(Path(dataset_path).name, None),
                image_resize=(504, 504),
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
        if episode2num_token(ep_stem, num_tokens_factor=SPLIT_FACTOR) % world_size != rank:
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

    if args.test_run:
        test_eps = [ # needs update #
            'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
            'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
            'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
            'episode_006678', 'episode_004399'
        ]
        skip_episodes = [ep for ep in dataset.all_episode_names if ep not in test_eps]
        if len(skip_episodes) > 0:
            dataset.set_skip_episode(skip_episodes)

    # ----- start processing -----
    possible_ep_jsonl = ''
    possible_ep = {}
    last = {} # last image global id that was processed
    possible_subtask_jsonl = ''
    sub_task_map = {}

    last_time = time.time()
    total_frames = len(dataset)
    for g_id, data in tqdm(enumerate(dataset), desc=f"[{dataset.name}] BBOX R{rank}/{world_size}", total=total_frames):
        ep_stem = dataset.get_episode_name_by_global_index(g_id)
        
        # --- skip episodes not assigned to this rank ---
        if episode2num_token(ep_stem, num_tokens_factor=SPLIT_FACTOR) % world_size != rank:
            continue
        
        # --- cache for .bbox.jsonl ---
        pej = Path(output_bbox_dir) / dataset.name / f"{ep_stem}.bbox.jsonl"
        if pej != possible_ep_jsonl:
            possible_ep_jsonl = pej
            if pej.exists():
                _jf = load_jsonlines(pej)
                for cam_key in dataset.loaded_camera_keys:
                    possible_ep[cam_key] = [item['frame_id'] for item in _jf if item['camera_key'] == cam_key]
            else:
                possible_ep = {}
            last = {} # reset last ids for new episode

        # --- cache for .sub_task.jsonl ---
        stj = Path(SUB_TASK_JSONL_DIR) / dataset.name / f"{ep_stem}.sub_task.jsonl"
        if stj != possible_subtask_jsonl:
            possible_subtask_jsonl = stj
            sub_task_map = {}
            if stj.exists():
                st_items = load_jsonlines(stj)
                for st_item in st_items:
                    if st_item['sub_task']: # non-empty sub_task
                        sub_task_map[st_item['frame_id']] = st_item['sub_task']
        
        # --- prepare task description ---
        frame_id = int(data['frame_index'])
        if frame_id in sub_task_map:
            task_desc = sub_task_map[frame_id]
        else:
            task_desc = data['task_description']
        
        # --- finish frame do not need processing ---
        if task_desc in ['done', 'not done']:
            for cam_key in dataset.loaded_camera_keys:
                append_jsonlines({ # output empty directly
                    "frame_id": frame_id,
                    "camera_key": cam_key,
                    "bboxes": [],
                    "task_desc": task_desc,
                    "generated_text": "",
                }, pej)
            continue

        # --- process per camera ---
        for cam_key in dataset.loaded_camera_keys:
            # --- check if processed ---
            frame_not_in = frame_id not in possible_ep.get(cam_key, [])
            if not frame_not_in:
                last[cam_key] = g_id
                continue
                
            # --- skip highly similar frames ---
            if cam_key not in last or not check_data_sim(data, dataset[last[cam_key]], img_key=cam_key):
                last[cam_key] = g_id # this frame is different enough and needs processing

            # --- submit to input queue ---
            if cam_key in last and (last[cam_key] == g_id):
                input_queue.put((data[cam_key], task_desc, [frame_id, cam_key, ep_stem])) # input
            else:
                append_jsonlines({ # output reference, but not input
                    "frame_id": frame_id,
                    "camera_key": cam_key,
                    "ref": int(dataset[last[cam_key]]['frame_index'])
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
def consumer_task(output_queue: Queue, dataset_path: str, output_bbox_dir: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    output_bbox_dir = Path(output_bbox_dir)

    frame_count, bbox_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        pil_img, task, generated_text, _supps = item
        # r_idx, cam_key, ep_stem, answer_turn1 = _supps
        r_idx, cam_key, ep_stem = _supps

        json_responses, _failed_indices = generated_texts_to_json_responses([generated_text], [pil_img])
        json_response = json_responses[0]
        bbox_item = {
            "frame_id": r_idx,
            "camera_key": cam_key,
            "bboxes": json_response,
            "task_desc": task,
            # "answer_turn1": answer_turn1,
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
            # image_with_bbox = draw_bboxes(
            #     pil_img,
            #     json_response_to_bboxes(json_response),
            #     show_object_name=True
            # )
            # image_with_bbox_save_path = jsl_path.parent / f"{ep_stem}_{r_idx}.png"
            # image_with_bbox.save(image_with_bbox_save_path)
    return


# ===== main =====
@ray.remote
def producer_task_remote(*args):
    return producer_task(*args)

@ray.remote
def consumer_task_remote(*args):
    return consumer_task(*args)

def process_dataset(dataset_path: str):
    # sequential delay to avoid possible ray init conflict
    time.sleep(args.rank * 5)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    queue_max_size = BATCH_SIZE * 2
    input_queue = Queue(maxsize=queue_max_size)
    output_queue = Queue(maxsize=queue_max_size)
    
    if args.load_all_cameras:
        ego_mapper = third_mapper = 'load_all'
    else:
        ego_mapper = EGO_CAMERA_MAP
        third_mapper = EXO_CAMERA_MAP
    

    # submit tasks remotely
    producer_ref = producer_task_remote.remote(
        input_queue, dataset_path, 
        ego_mapper, third_mapper, 
        OUTPUT_BBOX_JSONL_DIR,
        RANK, WORLD_SIZE,
        REFRESH_FREQ
    )
    consumer_ref = consumer_task_remote.remote(
        output_queue, dataset_path, 
        OUTPUT_BBOX_JSONL_DIR,
        REFRESH_FREQ
    )
    
    # start gpu worker locally
    gpu_worker(input_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    # wait for all to complete
    ray.get([producer_ref, consumer_ref])

if __name__ == "__main__":
    # warm up
    # dummy_img = Image.new('RGB', (14, 14), color = 'black')
    # infer_instance([dummy_img], ["hello"])
    
    for dataset_path in tqdm(DATASETS):
        process_dataset(dataset_path)