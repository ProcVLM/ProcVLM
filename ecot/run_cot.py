"""
CUDA_VISIBLE_DEVICES=0,1 python ecot/run_cot.py --world_size 4 --rank 0
CUDA_VISIBLE_DEVICES=2,3 python ecot/run_cot.py --world_size 4 --rank 1
CUDA_VISIBLE_DEVICES=4,5 python ecot/run_cot.py --world_size 4 --rank 2
CUDA_VISIBLE_DEVICES=6,7 python ecot/run_cot.py --world_size 4 --rank 3
"""

prompt_example = '''
Here is an example chain of thought:
current task: put all red items in the blue plate.
last action: pick up the red apple on the pink plate.
next action: place the red apple in the blue plate.
cot: There are 4 red items on the table not in the blue plate, a red apple on the pink plate, 2 red blocks on the green plate and a strawberry on the table.
To complete the task, the robot need to pick up these items and place them in the blue plate one by one. Now the red apple has been picked up. So the next action is to place the red apple in the blue plate.
'''

prompt_example = '''
Here is an example chain of thought:
current task: put all red items in the blue plate.
next action: place the red apple in the blue plate.
cot: There are 4 red items on the table not in the blue plate, a red apple on the pink plate, 2 red blocks on the green plate and a strawberry on the table.
To complete the task, the robot need to pick up these items and place them in the blue plate one by one. Now the red apple has been picked up. So the next action is to place the red apple in the blue plate.
'''

prompt_temp = '''
Here is your task: {instruction}
last action: {last_action}
the action should be done next: {next_action}
'''

prompt_temp = '''
Here is your task: {instruction}
the action should be done next: {next_action}
'''

prompt_temp_st = '''
Here is your task: {instruction}
the action should be done next: {next_action}
'''

prompt_example_st = '''
Here is an example chain of thought:
current task: put all red items in the blue plate.
next action: pick up the apple on the pink plate.
cot: There are 4 red items on the table not in the blue plate, a red apple on the pink plate, 2 red blocks on the green plate and a strawberry on the table.
To complete the task, the robot need to pick up these items and place them in the blue plate one by one. Now the next action is to pick up the apple on the pink plate.
'''

prompt_example_ed = '''
Here is an example chain of thought:
current task: put all red items in the blue plate.
next action: done
cot: All red items have been placed in the blue plate, including a red apple on the pink plate, 2 red blocks on the green plate and a strawberry on the table. The task is completed.
'''

prompt_temp_ed = '''
Here is your task: {instruction}
the action should be done next: {next_action}
'''


prompt_inst = '''
You are an intelligent agent. Given an image of the origin state and current state, an instruction and the next action, please carefully reason step-by-step: Why does the current state lead to this action? 
The first image is about the origin state and the second one is about current state. Don't mention the last action in your reasoning, just focus on the current state and the next action.
Based on your understanding of the images and the current state, interpretation of the command, write down your brief chain of thought in a simple paragraph with 2 or 3 short sentences.
'''

prompt_inst = '''
You are an intelligent agent. Given an image of the origin state and current state, an instruction and the next action, please carefully reason step-by-step: Why does the current state lead to this action? 
The first image is about the origin state and the second one is about current state. 
Based on your understanding of the images and the current state, interpretation of the command, write down your brief chain of thought in a simple paragraph with 2 or 3 short sentences.
'''

import logging
import os
import argparse
import pickle
import time
import multiprocessing as mp
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import load_jsonlines, append_jsonlines, episode2num_token, draw_text_block
from core.utils.runner_utils import gpu_worker

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
BATCH_SIZE = 40
TENSOR_PARALLEL_SIZE = 2
VLM_PATH = os.getenv("VLM_PATH_INTERN_INFER")
OUTPUT_COT_JSONL_DIR = os.getenv("OUTPUT_COT_JSONL_DIR")
SUB_TASK_JSONL_DIR = os.getenv("ANNOTATION_SUB_TASK_DIR")
TASK_BORDER_DIR = os.getenv("ANNOTATION_ROOT")
REFRESH_FREQ = min(5144, 114 * BATCH_SIZE + 514) # show log and save pkl
# --- auto config dp ---
# DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank


# ===== infer helpers =====
def infer_instance(images: List[str], task_descriptions: List[str]):
    from core.models.internvl import batch_generate_with_lmd
    return batch_generate_with_lmd(
        images, task_descriptions, VLM_PATH,
        max_tokens=8192, temperature=0.0,
        tp=TENSOR_PARALLEL_SIZE,
        use_pytorch=True,
    )


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

    with open(Path(TASK_BORDER_DIR) / dataset.name / "borders.pkl", 'rb') as f:
        task_borders = pickle.load(f)

    # ----- start processing -----
    possible_ep_jsonl = ''
    possible_ep_rid = set()
    ep_sub_task_map = {}

    last_time = time.time()
    total_frames = len(task_borders) - 1
    for _tidx, task_start in enumerate(tqdm(task_borders[:-1], desc=f"[{dataset.name}] CoT(v1) R{rank}/{world_size}", total=total_frames)):
        data0 = dataset[task_start]
        ep_stem = data0['episode_name']
        task_id = int(data0['task_index'])
        task_end = task_borders[_tidx + 1]

        # if _tidx % (total_frames / 8 / world_size) != 0: # needs update #
        #     continue

        if episode2num_token(ep_stem) % world_size != rank:
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
            
        instruction = data0['task']
        image_front0 = data0['exo_image']
        for data in dataset[task_start: task_end]:
            frame_id = int(data['frame_index'])

            if frame_id in possible_ep_rid:
                continue

            # --- prepare prompt ---
            if frame_id in ep_sub_task_map:
                next_action = ep_sub_task_map[frame_id]
            else:
                # logging.warning(f"Record {frame_id} in {ep_stem} does not have sub-task annotation, falling back to task only.")
                # next_action = data['task'] if data['task'] != '' else 'No specified task'
                next_action = ''

            if not next_action:
                prompt = prompt_temp_ed.format(instruction=instruction,next_action=next_action)
                prompt = prompt_inst + prompt_example_ed + prompt
            else:
                # prompt = prompt_temp.format(instruction=instruction,last_action=last_action,next_action=next_action)
                prompt = prompt_temp.format(instruction=instruction,next_action=next_action)
                prompt = prompt_inst + prompt_example + prompt

            image_front = data['exo_image']
            image_list = [image_front0, image_front]

            # --- submit to input queue ---
            _supp = (frame_id, instruction, next_action, ep_stem)
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
        frame_id, instruction, next_action, ep_stem = _supp

        if "cot:" in generated_text:
            cot = generated_text.split("cot:")[1].strip()
            success_count += 1
        else:
            cot = generated_text.strip()

        cot_item = {
            "frame_id": frame_id,
            "cot": cot,
            "instruction": instruction,
            "sub_task": next_action,
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
            insp_path = Path(OUTPUT_COT_JSONL_DIR) / dataset_name / 'inspect'
            insp_path.mkdir(parents=True, exist_ok=True)
            img0, img = images
            new_img = Image.new('RGB', (img0.width + img.width, max(img0.height, img.height)))
            new_img.paste(img0, (0, 0))
            new_img.paste(img, (img0.width, 0))
            # draw cot text on new_img
            new_img = draw_text_block(new_img, cot, bg_transparency=0.55)
            new_img.save(insp_path / f"{ep_stem}_{frame_id}.png")
    return


# ===== main =====
def process_dataset(dataset_path: str):
    ctx = mp.get_context("spawn")
    queue_max_size = BATCH_SIZE * 20
    input_queue = ctx.Queue(maxsize=queue_max_size)
    output_queue = ctx.Queue(maxsize=queue_max_size)
    
    producer = ctx.Process(target=producer_task, args=(
        input_queue, dataset_path,
        RANK, WORLD_SIZE,
        REFRESH_FREQ
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
