"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_plan_task.py --world_size 3 --rank 0 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_plan_task.py --world_size 3 --rank 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ecot/run_plan_task.py --world_size 3 --rank 2
"""

instruction = '''I will give you a robot task and a video showing the robot arm performing the task. You need to analyze actions of the robot arm from the video and decompose the task into a sequence of detailed sub-tasks.
The sequence of sub-tasks should lead to the completion of the overall task.

###
Grasp [specific object]
e.g. "Grasp the red block"
Explain: Use this pattern when the robot isn't holding anything and needs to pick up an object, before performing other actions like placing or lifting.
###
Place [specific object] onto / into [specific location]
e.g. "Place the cup onto the table" or "Place the screwdriver into the tool rack."
Explain: Use this pattern when the robot is holding an object and needs to put it down at a specific location.
###
Push [specific object] [forward / backward / left / right]
e.g., "Push the blue block forward"
Explain: Use this pattern when the robot needs to move an object in a specific direction by applying force to it, without lifting or grasping it. "Push" and ("Grasp", "Place") are mutually exclusive.
###
Tilt the gripper
e.g. "Tilt the gripper to pour the liquid" or "Tilt the gripper slightly to the left"
Explain: Use this pattern when the robot needs to adjust the angle of its gripper, either to pour liquid or position the gripper for some specific task.
###
Hang [specific object] on / above [specific location]
e.g. "Hang the coat on the hook" or "Hang the cup above the table"
Explain: Use this pattern when the robot needs to suspend an object from a specific location, such as hanging a coat on a hook or a cup above a table.
###
Press [specific object]
e.g., "Press the button" or "Press the power switch until it clicks."
###
Open [specific object] 
e.g. "Open the door slowly"
###
Close [specific object] 
e.g., "Close the lid securely"
###
Rotate [specific object] 
e.g., "Rotate the knob clockwise"
###

All sub-tasks must be in exactly one of the eight patterns above, and following the Explain for each pattern. There's no need to include robot arm itself as an object in the sub-tasks.
You should output sub-tasks in a numbered list format, starting from 1. Each line contains one sub-task with a leading number and a period. No extra text or explanation. Just like the examples below.

Example 1:
Task: Move all the fruit from the blue plate to the pink plate.
<Images showing a table and serveral fruits. The robot first moves the apple, then the banana, and finally the orange.>
Output:
1. Grasp the apple from the blue plate.
2. Place the apple onto the pink plate.
3. Grasp the banana from the blue plate.
4. Place the banana onto the pink plate.
5. Grasp the orange from the blue plate.
6. Place the orange onto the pink plate.
7. done.

Example 2:
Task: Push the blocks displaying 7 times 6 to the front.
<Images showing a table with several marked blocks. Block displaying 7 is the first to be pushed forward, followed by the block displaying the multiplication symbol, and finally the block displaying 6.>
Output:
1. Push the block displaying 7 forward.
2. Push the block displaying the multiplication symbol forward.
3. Push the block displaying 6 forward.
4. done.

Example 3:
Task: Press the button from top to bottom.
<Images showing a table with a button. The robot moves its arm to the button and presses it.>
Output:
1. Press the button.
2. done.

Example 4:
Task: water the plant
<Images showing a table with a plant and a watering can. The robot grasps the watering can, lifts it, tilts it to pour water into the plant pot, then places the watering can back down.>
Output:
1. Grasp the watering can.
2. Hang the watering can above the plant pot.
3. Tilt the gripper to pour the liquid.
4. Place the watering can onto the table.
5. done.

Task: {task}
Output: '''

import logging
import os
import argparse
import time
import pickle
import ray
from ray.util.queue import Queue
from ray.experimental.tqdm_ray import tqdm
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from core.data.reader import FastLerobotVLReader
from core.data.generals import EXO_CAMERA_MAP, EGO_CAMERA_MAP
from core.utils.common import load_jsonlines, append_jsonlines, episode2num_token, images_to_video
from core.utils.runner_utils import process_task, gpu_worker, auto_split_datasets
from ecot.utils.sub_task_utils import get_sample_image_indices_sampled
from core.models.qwenvl import run_batch_frame_list_sgllm, process_batch_frame_list_sgllm

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
args.add_argument("--device", type=str, default=None, help="CUDA visible devices string.")
args.add_argument("--enable_auto_dp", action='store_true', help="whether to enable automatic data parallel splitting based on world size")
args = args.parse_args()
if args.device is not None:
    logging.warning(f"Setting CUDA_VISIBLE_DEVICES to {args.device} as per argument.")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
os.environ["RAY_TMPDIR"] = f"{os.getenv('CACHE_ROOT')}/ray_tmpdir_{args.rank}"
_raw_datasets = args.dataset_paths


# ===== config =====
BATCH_SIZE = 16 # needs update #
VLM_PATH = os.getenv("VLM_PATH_QWEN_INFER")
OUTPUT_PLAN_JSONL_DIR = os.getenv("OUTPUT_PLAN_JSONL_DIR")
REFRESH_FREQ = 1145 # needs update #
# --- auto config dp mode ---
if args.enable_auto_dp:
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
else:
    DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank
                  

# ===== infer helpers =====
qwenvl_kwargs = {
    "tp_size": 8,
    "ep_size": 2,
    "mem_fraction_static": 0.77,
    "context_length": 131072,
    "attention_backend": "fa3",
    # "cuda_graph_max_bs": 16,
    "disable_custom_all_reduce": True,
    # "chunked_prefill_size": 8192,
}

def infer_instance(not_used_image_list: List[Image.Image | None], llm_inputs_batch: List[Dict[str, Any]]):
    return run_batch_frame_list_sgllm(
        llm_inputs_batch, VLM_PATH,
        engine_kwargs=qwenvl_kwargs
    )

def process_instance(videos: List[List[Tuple[int, Image.Image]]], questions: List[str]):
    return process_batch_frame_list_sgllm(videos, questions, VLM_PATH)


# ===== pipeline helpers =====
def producer_task(input_queue: Queue, dataset_path: str, output_dir: str, world_size: int, rank: int, refresh_freq: int):
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            # ego_name=EGO_CAMERA_MAP.get(Path(dataset_path).name, None),
            image_resize=(480, 480),
            return_record_meta=True,
        )
        dataset.ego_camera_key = None
        dataset.loaded_camera_keys = [dataset.exo_camera_key]
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return
    
    # ----- pre-iterate the dataset to get task borders -----
    # task_borders = []
    # last_task, last_ep = None, None
    # for idx, data in tqdm(enumerate(dataset), desc=f"[{dataset.name}] Pre-iterating", total=len(dataset)):
    #     ep_stem = data['episode_name']
    #     if ep_stem != last_ep:
    #         last_ep = ep_stem
    #         last_task = None
    #     task_id = int(data['task_index'])
    #     if task_id != last_task:
    #         task_borders.append(idx)
    #         last_task = task_id
    # task_borders.append(len(dataset))  # end border
    with open(Path(output_dir) / dataset.name / "borders.pkl", 'rb') as f:
        task_borders = pickle.load(f)
    
    # ----- fast skip processed parquets -----
    last_ep_stem = None
    for ep_stem in dataset.all_episode_names:
        if episode2num_token(ep_stem) % world_size != rank:
            continue
        s, t = dataset.get_episode_range(ep_stem)
        if t - s < 10:
            continue
        pej = Path(output_dir) / dataset.name / f"{ep_stem}.plan.jsonl"
        if not pej.exists():
            break
        last_ep_stem = ep_stem
    start_global_idx = 0
    if last_ep_stem is not None:
        start_global_idx, _ = dataset.get_episode_range(last_ep_stem)
        logging.info(f"Producer rank {rank}/{world_size} fast skipped to episode {last_ep_stem} at global index {start_global_idx}.")

    # ----- start processing -----
    possible_jsonl_path = ''
    possible_jsonl_rid = set()

    start_time = time.time()
    fps = int(dataset.meta.fps)
    total_frames = len(task_borders) - 1
    for _tidx, task_start in enumerate(tqdm(task_borders[:-1], desc=f"[{dataset.name}] Plan R{rank}/{world_size}", total=total_frames)):
        if task_start < start_global_idx:
            continue

        task_end = task_borders[_tidx + 1]
        if int(task_end - task_start) < 10: # skip too short tasks
            continue

        data = dataset[task_start]
        ep_stem = data['episode_name']
        frame_id = int(data['frame_index'])
        task_id = int(data['task_index'])

        # if _tidx % (total_frames / 8 / world_size) != 0: # needs update #
        #     continue

        if episode2num_token(ep_stem) % world_size != rank:
            continue

        # --- cache for .plan.jsonl ---
        pjp_sub = Path(output_dir) / dataset.name / f"{ep_stem}.plan.jsonl"
        if str(pjp_sub) != str(possible_jsonl_path):
            possible_jsonl_path = str(pjp_sub)
            if pjp_sub.exists():
                _jf = load_jsonlines(pjp_sub)
                possible_jsonl_rid = set(int(item['frame_id']) for item in _jf)
            else:
                possible_jsonl_rid = set()

        if frame_id in possible_jsonl_rid:
            continue
        
        task = data['task'] if data['task'] != '' else 'No specified task'

        sampled_frame_indices = get_sample_image_indices_sampled(task_start, task_end, fps, max_frames=128)
        # if dataset.ego_camera_key is not None:
        #     frames = []
        #     for i in sampled_frame_indices:
        #         # combine third and ego views side by side (both Image.Image)
        #         F_third = dataset[i]['exo_image']
        #         F_ego = dataset[i]['ego_image']
        #         W, H = F_third.size
        #         F_combined = Image.new('RGB', (W * 2, H))
        #         F_combined.paste(F_third, (0, 0))
        #         F_combined.paste(F_ego, (W, 0))
        #         frames.append(F_combined)
        # else:
        frames = [dataset[i]['exo_image'] for i in sampled_frame_indices]
        frame_ids = [int(dataset[i]['frame_index']) for i in sampled_frame_indices]
        video = [(fid, f) for fid, f in zip(frame_ids, frames)]
        instr = instruction.format(
            task=task,
        )
        supps = (frame_id, ep_stem, task, task_id, task_start, task_end)
        input_queue.put((video, instr, supps))
        
        if _tidx % refresh_freq == 0 and _tidx > 0:
            elapsed = time.time() - start_time
            speed = refresh_freq / elapsed
            remaining = (total_frames - _tidx) / speed
            logging.info(
                f"Producer rank {rank}/{world_size} processed {_tidx}/{total_frames} tasks, "
                f"speed: {speed:.2f} tasks/s, "
                f"elapsed: {elapsed/3600:.2f} h, "
                f"remaining: {remaining/3600:.2f} h."
            )
            start_time = time.time()
    # end of input
    input_queue.put(None)
    return

def consumer_task(output_queue: Queue, dataset_path: str, output_dir: str, refresh_freq: int):
    dataset_name = Path(dataset_path).name
    output_dir = Path(output_dir)

    frame_count, success_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        pil_imgs, question, generated_text, _supp = item
        frame_id, ep_stem, task, task_index, task_start, task_end = _supp

        # try to get structured output
        plan_list = generated_text.strip().split('\n')
        cleaned_plan_list = []
        for i in range(len(plan_list)):
            if plan_list[i].strip() == '':
                continue
            # remove leading numbering if any
            if plan_list[i].strip()[0].isdigit():
                dot_pos = plan_list[i].find('.')
                if dot_pos != -1:
                    plan_list[i] = plan_list[i][dot_pos+1:].strip()
            cleaned_plan_list.append(plan_list[i].strip())
        
        if len(cleaned_plan_list) > 0:
            success_count += 1

        plan_item = {
            "frame_id": frame_id,
            "task_index": task_index,
            "plan": cleaned_plan_list,
            "task_start": task_start,
            "task_end": task_end,
            "task": task,
        }
        jsl_path = output_dir / dataset_name / f"{ep_stem}.plan.jsonl"
        jsl_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(plan_item, jsl_path)

        # if int(frame_id) == 0:
        #     pil_imgs[0].save(
        #         output_dir / dataset_name / f"{ep_stem}.plan.jpg",
        #         format="JPEG", quality=95
        #     )

        frame_count += 1
        if frame_count % refresh_freq == 0:
            speed = refresh_freq / (time.time() - last_time)
            logging.info(
                f"Consumer throughput: {speed:.2f} frames/s. "
                f"Processed {frame_count} frames, {success_count} successful. "
                f"Failed rate: {(frame_count - success_count)/frame_count:.2%}."
            )
            last_time = time.time()

            # video_path = output_dir / dataset_name / f"inspect_plan/{ep_stem}.mp4"
            # images_to_video(frames, video_path, fps=10)
            # shutil.copy(jsl_path, output_dir / dataset_name / f"inspect_plan/{ep_stem}.plan.jsonl")

    return


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
        ray.init(ignore_reinit_error=True)

    queue_max_size = max(20, BATCH_SIZE * 2)
    input_queue = Queue(maxsize=queue_max_size)
    mid_queue = Queue(maxsize=queue_max_size)
    output_queue = Queue(maxsize=queue_max_size)
    # submit tasks remotely
    producer_ref = producer_task_remote.remote(
        input_queue, dataset_path, 
        OUTPUT_PLAN_JSONL_DIR,
        WORLD_SIZE, RANK,
        REFRESH_FREQ
    )
    processor_ref = process_task_remote.remote(
        input_queue, mid_queue,
        process_instance,
        1, REFRESH_FREQ
    )
    consumer_ref = consumer_task_remote.remote(
        output_queue, dataset_path,
        OUTPUT_PLAN_JSONL_DIR,
        REFRESH_FREQ
    )
    # start gpu worker locally
    gpu_worker(mid_queue, output_queue, infer_instance, BATCH_SIZE, REFRESH_FREQ)
    # wait for all to complete
    ray.get([producer_ref, processor_ref, consumer_ref])


if __name__ == "__main__":
    # warm up
    # infer_instance([[]], [])

    for every_path in tqdm(DATASETS):
        process_dataset(every_path)