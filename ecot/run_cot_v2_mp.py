"""
CUDA_VISIBLE_DEVICES=0,1 python ecot/run_cot_v2_mp.py --world_size 4 --rank 0 &
CUDA_VISIBLE_DEVICES=2,3 python ecot/run_cot_v2_mp.py --world_size 4 --rank 1 &
CUDA_VISIBLE_DEVICES=4,5 python ecot/run_cot_v2_mp.py --world_size 4 --rank 2 &
CUDA_VISIBLE_DEVICES=6,7 python ecot/run_cot_v2_mp.py --world_size 4 --rank 3 &
wait
"""

prompt_not_finished = """
The image shows a robot performing a task: '{task}', which may be incomplete.
Remaining subtasks: '{rest_sub_task}'.
Explain why it's unfinished and briefly describe the next steps based on image details.

Output (≤150 words, 3 sentences):
<analysis with image details>. This task is not finished <short reason>. <one-sentence summary of next steps>.

Example:
Task: 'put all the green objects on the pink plate.'
Image: a green apple in robot arm, a green pear on blue plate.
Output:
Image shows a green apple held by the robot and a green pear on the blue plate. This task is not finished because both green objects are not yet on the pink plate. The robot should place the green apple on the pink plate, then move the green pear from the blue plate to the pink plate.
"""

prompt_finished = """
The image shows a robot performing a task: '{task}', which is finished.
Explain briefly why it's completed based on image details.

Output (≤50 words, 2 sentences):
<analysis with image details>. This task is finished <short reason>.

Example:
Task: 'put all the green objects on the pink plate.'
Image: both green apple and pear on pink plate.
Output:
Image shows a green apple and pear on the pink plate. This task is finished because all green objects are placed correctly.
"""

prompt_giveup = """
The image shows a robot performing a task: '{task}', which is not finished.
Explain briefly why it's unfinished based on image details.

Output (≤50 words, 2 sentences):
<analysis with image details>. This task is not finished <short reason>.

Example:
Task: 'put all the green objects on the pink plate.'
Image: a green apple held by the robot, a green pear on blue plate.
Output:
Image shows a green apple in the robot arm and a green pear on the blue plate. This task is not finished because neither object has been placed on the pink plate.
"""


import logging
import os
import argparse
import pickle
import time
import multiprocessing as mp
from multiprocessing import Queue
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import load_jsonlines, append_jsonlines, episode2num_token, draw_text_block, check_data_sim
from core.utils.runner_utils import gpu_worker, auto_split_datasets
from core.models.internvl import batch_generate_with_lmd

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
# ===== dataset paths =====
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 14
TENSOR_PARALLEL_SIZE = 2
VLM_PATH = os.getenv("VLM_PATH_INTERN_INFER")
OUTPUT_COT_JSONL_DIR = os.getenv("OUTPUT_COT_JSONL_DIR")
SUB_TASK_JSONL_DIR = os.getenv("ANNOTATION_SUB_TASK_DIR")
TASK_BORDER_DIR = os.getenv("OUTPUT_PLAN_JSONL_DIR")
REFRESH_FREQ = 11451 # show log and save pkl
SPLIT_FACTOR = args.split_factor
# --- auto config dp ---
if args.enable_auto_dp:
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
else:
    DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank


# ===== infer helpers =====
def infer_instance(images: List[str], task_descriptions: List[str]):
    return batch_generate_with_lmd(
        images, task_descriptions, VLM_PATH,
        max_tokens=2048, temperature=0.0,
        tp=TENSOR_PARALLEL_SIZE,
        use_pytorch=True,
    )


# ===== pipeline helpers =====
def producer_task(input_queue: Queue, dataset_path: str, rank: int, world_size: int, refresh_freq: int):
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            image_resize=(448, 448),
            return_record_meta=True,
        )
        with open(Path(TASK_BORDER_DIR) / dataset.name / "borders.pkl", 'rb') as f:
            task_borders = pickle.load(f)        
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return
    logging.info(f"Dataset '{dataset.name}' loaded, producer started for rank {rank}/{world_size}.")

    # ----- fast skip processed parquets -----
    last_ep_stem = None
    for ep_stem in dataset.all_episode_names:
        if episode2num_token(ep_stem, num_tokens_factor=SPLIT_FACTOR) % world_size != rank:
            continue
        pej = Path(OUTPUT_COT_JSONL_DIR) / dataset.name / f"{ep_stem}.cot.jsonl"
        if not pej.exists():
            break
        last_ep_stem = ep_stem
    start_global_idx = 0
    if last_ep_stem is not None:
        start_global_idx, _ = dataset.get_episode_range(last_ep_stem)
        logging.info(f"Producer rank {rank}/{world_size} fast skipped to episode {last_ep_stem} at global index {start_global_idx}.")

    # ----- start processing -----
    possible_ep_jsonl = ''
    possible_ep_rid = set()
    ep_sub_task_map = {}

    last_time = time.time()
    total_frames = len(task_borders) - 1
    for _tidx, task_start in enumerate(tqdm(task_borders[:-1], desc=f"[{dataset.name}] CoT R{rank}/{world_size}", total=total_frames)):
        if task_start < start_global_idx:
            continue

        # if _tidx % (total_frames / 8 / world_size) != 0: # needs update #
        #     continue

        ep_stem = dataset.get_episode_name_by_global_index(task_start)
        if episode2num_token(ep_stem, num_tokens_factor=SPLIT_FACTOR) % world_size != rank:
            continue
        
        # --- cache for .cot.jsonl ---
        pej = Path(OUTPUT_COT_JSONL_DIR) / dataset.name / f"{ep_stem}.cot.jsonl"
        if pej != possible_ep_jsonl:
            possible_ep_jsonl = pej
            if pej.exists():
                _jf = load_jsonlines(pej)
                possible_ep_rid = set(int(item['frame_id']) for item in _jf)
            else:
                possible_ep_rid = set()
            # --- load sub_task annotations ---
            ep_sub_task_map = {}
            pstj = Path(SUB_TASK_JSONL_DIR) / dataset.name / f"{ep_stem}.sub_task.jsonl"
            if pstj.exists():
                _sub_jf = load_jsonlines(pstj)
                for item in _sub_jf:
                    ep_sub_task_map[int(item['frame_id'])] = item['sub_task']
            else:
                logging.warning(f"Sub-task annotation file not found: {pstj}, all frames in episode {ep_stem} will fall back to task only.")
                ep_sub_task_map = {}

        # --- get task info ---
        task_end = task_borders[_tidx + 1]
        data0 = dataset[task_start]
        # task_id = int(data0['task_index'])
        task = data0['task']
        data1 = dataset[task_end - 1]
        task_start_frame_id = int(data0['frame_index'])
        task_end_frame_id = int(data1['frame_index'])
        # ep_sub_task_map.values() but sort by frame_id
        rest_sub_tasks = [ep_sub_task_map[frame_id] for frame_id in sorted(ep_sub_task_map.keys()) if task_start_frame_id <= frame_id <= task_end_frame_id]
        # evict empty, 'done', 'not done' from rest_sub_tasks
        rest_sub_tasks = [st for st in rest_sub_tasks if st and st != 'done' and st != 'not done']
        # if not rest_sub_tasks: # if you want to skip tasks without sub-tasks, uncomment this
        #     continue
        
        # process each frame in the task segment
        last_processed_id = None
        for g_id, data in zip(range(task_start, task_end), dataset[task_start: task_end]):
            frame_id = int(data['frame_index'])
            if frame_id in possible_ep_rid:
                last_processed_id = g_id
                continue

            # --- prepare prompt ---
            if frame_id in ep_sub_task_map:
                sub_task = ep_sub_task_map[frame_id]
            else:
                # logging.warning(f"Record {frame_id} in {ep_stem} does not have sub-task annotation, falling back to task only.")
                # next_action = data['task'] if data['task'] != '' else 'No specified task'
                sub_task = ''

            # --- remove outdated ---
            if len(rest_sub_tasks) > 0 and sub_task in rest_sub_tasks:
                while sub_task != rest_sub_tasks[0]:
                    last_processed_id = None # reset last processed id when sub_task changes
                    rest_sub_tasks.pop(0)

            # --- skip highly similar frames ---
            if not last_processed_id or not check_data_sim(data, dataset[last_processed_id], img_key='exo_image', thershold=0.98):
                last_processed_id = g_id

            if last_processed_id == g_id: # this frame is different enough and needs processing
                if sub_task == 'done':
                    prompt = prompt_finished.format(task=task)
                elif sub_task == 'not done':
                    prompt = prompt_giveup.format(task=task)
                else:
                    if len(rest_sub_tasks) == 0 and sub_task == '':
                        rest_sub_task_str = data['task']
                    elif len(rest_sub_tasks) == 0:
                        rest_sub_task_str = sub_task
                    else:
                        rest_sub_task_str = '; '.join(rest_sub_tasks)
                    prompt = prompt_not_finished.format(task=task, rest_sub_task=rest_sub_task_str)

                image_front = data['exo_image']

                # --- submit to input queue ---
                _supp = (frame_id, task, sub_task, ep_stem)
                input_queue.put((image_front, prompt, _supp))
            else:
                append_jsonlines({ # output reference, but not input
                    "frame_id": frame_id,
                    "ref": int(dataset[last_processed_id]['frame_index'])
                }, pej)

        # --- log progress ---
        if _tidx and _tidx % refresh_freq == 0:
            elapsed = time.time() - last_time
            speed = _tidx / elapsed
            rest_frames = total_frames - _tidx
            eta = rest_frames / speed if speed > 1e-5 else -1
            logging.info(f"Producer throughput: {speed:.2f} episodes/s. Submitted {_tidx}/{total_frames} episodes. ETA: {eta/3600:.2f} hours.")
    # end of input
    input_queue.put(None)
    return

# Post Process Model Output
def consumer_task(output_queue: Queue, dataset_path: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    frame_count, success_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        images, question, generated_text, _supp = item
        frame_id, task, sub_task, ep_stem = _supp

        # keep at most 3 sentences
        cot_sentences = generated_text.strip().split('. ')
        if len(cot_sentences) > 3:
            cot = '. '.join(cot_sentences[:3])
            if not cot.endswith('.'):
                cot += '.'
        else:
            cot = generated_text.strip()
        if cot != '':
            success_count += 1

        cot_item = {
            "frame_id": frame_id,
            "task": task,
            "sub_task": sub_task,
            "cot": cot,
            "generated_text": generated_text,
        }

        jsl_path = Path(OUTPUT_COT_JSONL_DIR) / dataset_name / f"{ep_stem}.cot.jsonl"
        jsl_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(cot_item, jsl_path)
       
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
            # insp_path = Path(OUTPUT_COT_JSONL_DIR) / dataset_name / 'inspect'
            # insp_path.mkdir(parents=True, exist_ok=True)
            # # img0, img = images
            # # new_img = Image.new('RGB', (img0.width + img.width, max(img0.height, img.height)))
            # # new_img.paste(img0, (0, 0))
            # # new_img.paste(img, (img0.width, 0))
            # # draw cot text on new_img
            # images = draw_text_block(images, cot, bg_transparency=0.55)
            # images.save(insp_path / f"{ep_stem}_{frame_id}.png")
    return


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = BATCH_SIZE * 2
    input_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path,
        RANK, WORLD_SIZE,
        REFRESH_FREQ // 10
    ))
    consumer = ctx.Process(target=consumer_task, args=(
        output_queue, dataset_path, 
        REFRESH_FREQ
    ))

    producer.start()
    consumer.start()
    gpu_worker(input_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    producer.join()
    consumer.join()


if __name__ == "__main__":
    # warm up
    # dummy_img = Image.new('RGB', (24, 24), color = 'black')
    # infer_instance([dummy_img], ["you are a helpful assistant."])
    
    for dataset_path in tqdm(DATASETS):
        process_dataset(dataset_path)
