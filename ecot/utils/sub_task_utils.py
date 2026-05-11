direct_instruction = '''You will be shown a VIDEO of a robot doing a task. Decompose it into sub-tasks.

Each sub-task should be a simple atomic action, described by one of these patterns:
- Grasp [object] (pick up before placing/lifting)
- Place [object] onto/into [location] (when holding object, put down)
- Push [object] [direction] (move without grasp; exclusive with grasp/place)
- Tilt the gripper (adjust angle, e.g. pour/position)
- Hang [object] on/above [location] (suspend, e.g. coat on hook)
- Press [object] (e.g. button, switch)
- Open [object] 
- Close [object] 
- Rotate [object] (e.g. knob)

Examples:
Task: Move all the fruit from the blue plate to the pink plate.
Subtasks:
1. Grasp the apple from the blue plate.
2. Place the apple onto the pink plate.
3. Grasp the banana from the blue plate.
4. Place the banana onto the pink plate.
5. Grasp the orange from the blue plate.
6. Place the orange onto the pink plate.
Task: water the plant
Subtasks:
1. Grasp the watering can.
2. Hang the watering can above the plant pot.
3. Tilt the gripper to pour the liquid.
4. Place the watering can onto the table.

Every atomic action shown in the video should be reported as a sub-task.

For each sub-task:
- Write a descriptive name using above patterns
- Mark key frame (integer) where it is completed
- Write short optional note if unclear

INPUT: 
Task: "{task}"

OUTPUT:
Return JSON:
{{
  "task": "<input task>",
  "subtasks": [
    {{"id":1,"complete_frame":<int, same as the input frame timestamp>,"name":"<write a descriptive name using above patterns>","notes":"<optional>"}},
    ...
  ],
  "overall_notes":"<<=30 words, optional>"
}}

Hints: Use gripper pose, object motion, contact events. If multiple candidate frames, pick the clear completion. If retries, take final success.

Now process the provided video and planned sub-tasks and return the JSON result ONLY.
'''

plan_instruction = '''You will be shown a VIDEO of a robot task and an UNORDERED list of planned sub-tasks.

Task: "{task}"
Planned sub-tasks:
{plans}

OBJECTIVE:
For each planned sub-task, if present, mark the frame where it finishes. If not found, set complete_frame=null and notes="not present".
If the video shows any action was interrupted and the overall task was not completed, set overall_notes="task not completed".

OUTPUT FORMAT:
{{
  "task": "<same as input task>",
  "subtasks": [
    {{"id":1,"notes":"<<=40 words optional>","complete_frame":<int|null>,"name":"<same text from plans>"}},
    ...
  ],
  "overall_notes":"<<=30 words optional>"
}}

EXAMPLE
Task: "assemble the box"
Plans: ["pick up lid","place lid on box","close latch"]
Output:
{{
  "task":"assemble the box",
  "subtasks":[
    {{"id":1,"notes":"gripper closed on lid at frame 24","complete_frame":24,"name":"pick up lid"}},
    {{"id":2,"notes":"lid aligned with box at frame 58","complete_frame":58,"name":"place lid on box"}},
    {{"id":3,"notes":"not present","complete_frame":null,"name":"close latch"}}
  ],
  "overall_notes":"The box was assembled successfully."
}}

HINTS:
- Find the changes in effector pose, object motion, contacts, appearance/disappearance as the candidate start/complete frames.
- Pick the frame where change is clearly complete.
- If multiple candidates, pick the final success.
- If retries, record final success.

Return JSON ONLY.

Now process the provided video and planned sub-tasks and return the JSON result ONLY.
'''

plan_instruction_bidirectional = '''You will be shown a VIDEO of a robot task and an UNORDERED list of planned sub-tasks.

Task: "{task}"
Planned sub-tasks:
{plans}

OBJECTIVE:
For each planned sub-task, if present, mark the frames where it starts and finishes. If a sub-task is started but not finished, set complete_frame=null. If not present at all, set both start_frame=null and complete_frame=null, and notes="not present".
If the video shows any action was interrupted and the overall task was not completed, set overall_notes="task not completed".

OUTPUT FORMAT:
{{
  "task": "<same as input task>",
  "subtasks": [
    {{"id":1, "notes":"<<=60 words optional>", "start_frame":<int|null>,"complete_frame":<int|null>, "name":"<same text from plans>"}},
    ...
  ],
  "overall_notes":"<<=30 words optional>"
}}

EXAMPLE
Task: "put the green apple and the banana on the green plate"
Plans: ["Grasp the green apple.","Place the green apple onto the green plate.","Grasp the banana.","Place the banana onto the green plate."]
Output:
{{
  "task":"put the green apple and the banana on the green plate",
  "subtasks":[
    {{"id":1,"notes":"gripper moving towards green apple at frame 12, closed on green apple at frame 24","start_frame":12,"complete_frame":24,"name":"Grasp the green apple."}},
    {{"id":2,"notes":"the robot stuck while placing the green apple at frame 30","start_frame":30,"complete_frame":null,"name":"Place the green apple onto the green plate."}},
  ],
  "overall_notes":"The green apple and banana were not placed on the green plate successfully. task not completed."
}}

HINTS:
- Find the changes in effector pose, object motion as the candidate start/complete frames.
- The start frame can be picked slightly earlier and the complete frame slightly later to ensure the action is fully captured.
- If multiple candidates, pick the final success. If retries, record final success.
- Use the last frame of the video as reference for overall_notes.

Now process the provided video and planned sub-tasks and return the JSON result ONLY.
'''

import json_repair
import logging
import time
import pickle
import shutil
from pathlib import Path
from ray.util.queue import Queue
from ray.experimental.tqdm_ray import tqdm
from core.utils.common import load_jsonlines, append_jsonlines, draw_text_block, images_to_video, episode2num_token
from core.data.reader import FastLerobotVLReader
from core.data.generals import EXO_CAMERA_MAP, EGO_CAMERA_MAP
from typing import List, Tuple, Optional, Dict, Any


# ===== helper functions =====
def get_sample_image_indices_bounded(ts: int, te: int, max_frames: int=192) -> List[int]:
    total_frames = te - ts
    if total_frames <= max_frames:
        return list(range(ts, te))
    step = (total_frames + max_frames - 1) // max_frames  # ensure at least max_frames frames
    return list(range(ts, te, step))

def get_sample_image_indices_sampled(ts: int, te: int, step: int, min_frames: int=16, max_frames: int=128) -> List[int]:
    step = int(max(1, min((te - ts) // min_frames, step))) # sample at least min_frames frames, and prefer step
    if len(range(ts, te, step)) > max_frames:
        step = (te - ts + max_frames - 1) // max_frames  # at most max_frames frames
    sampled_frame_indices = list(range(ts, te, step))
    return sampled_frame_indices

def post_process_gen_text(gen_texts: str, frame_ids: List[int]) -> Tuple[Dict[str, Any] | str, bool]:
    jd = json_repair.loads(gen_texts)
    if not isinstance(jd, dict):
        return gen_texts, False
    if "subtasks" not in jd or not isinstance(jd["subtasks"], list):
        return gen_texts, False
    valid_subtask_num = 0
    for sub in jd["subtasks"]:
        if "name" not in sub or "complete_frame" not in sub:
            continue
        # re-map frame index
        if sub['complete_frame'] is None:
            continue
        sub['complete_frame'] = int(sub["complete_frame"])
        if sub['complete_frame'] < 0 or sub['complete_frame'] >= len(frame_ids):
            sub['complete_frame'] = None
            continue
        sub['complete_frame'] = frame_ids[sub['complete_frame']]
        valid_subtask_num += 1
    return jd, (valid_subtask_num > 0)

def post_process_gen_text_bidirectional(gen_texts: str, frame_ids: List[int]) -> Tuple[Dict[str, Any] | str, bool]:
    jd = json_repair.loads(gen_texts)
    if not isinstance(jd, dict):
        return gen_texts, False
    if "subtasks" not in jd or not isinstance(jd["subtasks"], list):
        return gen_texts, False
    valid_subtask_num = 0
    for sub in jd["subtasks"]:
        if "name" not in sub or "complete_frame" not in sub or "start_frame" not in sub:
            continue
        # re-map frame index
        if sub['complete_frame'] is not None:
            sub['complete_frame'] = int(sub["complete_frame"])
            if sub['complete_frame'] < 0 or sub['complete_frame'] >= len(frame_ids):
                sub['complete_frame'] = None
            else:
                sub['complete_frame'] = frame_ids[sub['complete_frame']]
                valid_subtask_num += 1
        if sub['start_frame'] is not None:
            sub['start_frame'] = int(sub["start_frame"])
            if sub['start_frame'] < 0 or sub['start_frame'] >= len(frame_ids):
                sub['start_frame'] = None
            else:
                sub['start_frame'] = frame_ids[sub['start_frame']]
                valid_subtask_num += 1
    return jd, (valid_subtask_num > 0)

def sub_task_json_to_flattened_list(results_json: Dict[str, Any], task_start: int, task_end: int, delay: float=0.00) -> List[Dict[str, Any]]:
    subtasks = results_json['subtasks'] # List[Dict[str, Any]]
    subtasks = [st for st in subtasks if 'start_frame' in st and 'complete_frame' in st and 'name' in st]
    # sort by complete_frame ascending, None at last
    subtasks = sorted(subtasks, key=lambda x: (x['complete_frame'] is None, x['complete_frame']))
    frame2note = {}
    last_end = task_start
    last_task = None
    task_len = task_end - task_start
    for sub in subtasks:
        if sub['complete_frame'] is not None and sub['start_frame'] is not None:
            # for fid in range(sub['start_frame'], sub['complete_frame']):
        # if sub['complete_frame'] is not None:
            end = min(task_end - 1, sub['complete_frame'] + int(delay * task_len))
            for fid in range(last_end, end):
                frame2note[fid] = sub['name']
            last_end = end
            last_task = sub['name']
    # fill remaining frames
    # reserved_frame_start = min(task_end - 1, task_start + max(int(0.9 * task_len), task_len - 30))
    # if last_end < reserved_frame_start and last_task is not None:
    #     for fid in range(last_end, reserved_frame_start):
    #         frame2note[fid] = last_task
    #     last_end = reserved_frame_start
    done = True
    if 'not completed' in results_json.get('overall_notes', '').lower():
        done = False
    else:
        failed_count = sum(1 for sub in subtasks if sub['complete_frame'] is None or sub['start_frame'] is None)
        if failed_count >= len(subtasks) / 2:
            done = False
    for fid in range(last_end, task_end):
        if done:
            frame2note[fid] = 'done' 
        else:
            frame2note[fid] = last_task if last_task is not None else 'not done'
    new = []
    for fid in range(task_start, task_end):
        new.append({
            'frame_id': fid,
            'task': results_json['task'],
            'sub_task': frame2note[fid],
        })
    return new


# ===== instruction template =====
instruction = plan_instruction_bidirectional
post_process_func = post_process_gen_text_bidirectional


# ===== pipeline functions =====
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
        with open(Path(output_dir) / dataset.name / "borders.pkl", 'rb') as f:
            task_borders = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        input_queue.put(None)
        return
    logging.info(f"Producer started for rank {rank}/{world_size}.")

    # ----- fast skip processed parquets -----
    last_ep_stem = None
    for ep_stem in dataset.all_episode_names:
        if episode2num_token(ep_stem) % world_size != rank:
            continue
        s, t = dataset.get_episode_range(ep_stem)
        if t - s < 10:
            continue
        pej = Path(output_dir) / dataset.name / f"{ep_stem}.sub_task.jsonl"
        if not pej.exists():
            break
        last_ep_stem = ep_stem
    start_global_idx = 0
    if last_ep_stem is not None:
        start_global_idx, _ = dataset.get_episode_range(last_ep_stem)
        logging.info(f"Producer rank {rank}/{world_size} fast skipped to episode {last_ep_stem} at global index {start_global_idx}.")

    # ----- process each episode -----
    start_time = time.time()
    fps = int(dataset.meta.fps)
    total_frames = len(task_borders) - 1
    for _tidx, task_start in enumerate(tqdm(task_borders[:-1], desc=f"[{dataset.name}] SubTask {rank}/{world_size}", total=total_frames)):
        if task_start < start_global_idx:
            continue

        task_end = task_borders[_tidx + 1]
        if int(task_end - task_start) < 10: # skip too short tasks
            continue

        data = dataset[task_start]
        ep_stem = data['episode_name']
        task_id = int(data['task_index'])
        task = data['task']

        # if _tidx >= 24: # needs update #
        #     break
        # if _tidx % 3 != 0: # needs update #
        #     continue
        # if _tidx % (total_frames / 8 / world_size) != 0: # needs update #
        #     continue

        if episode2num_token(ep_stem) % world_size != rank:
            continue
        
        # needs update #
        pej = Path(output_dir) / dataset.name / f"{ep_stem}.sub_task.jsonl"
        if pej.exists():
            jl = load_jsonlines(pej)
            if any(item['task_id'] == task_id for item in jl):
                # already processed
                continue
        
        pej = Path(output_dir) / dataset.name / f"{ep_stem}.plan.jsonl"
        if not pej.exists():
            raise FileNotFoundError(f"Plan file not found: {pej}")
        jl = load_jsonlines(pej)
        plan_item = next((item for item in jl if item['task_index'] == task_id), None)
        if plan_item is None:
            logging.warning(f"Plan for task_id {task_id} not found in {pej}.")
            # check sub_task_jsonl/fractal20220817_data/episode_045712.plan.jsonl
            plan_str = "There are no detailed sub-tasks provided.  Please analyze the task and break it down into logical sub-tasks based on the robot's actions and objectives."
            # raise ValueError(f"Plan for task_id {task_id} not found in {pej}")
        else:
            plan_list = plan_item['plan']
            if plan_list[-1].lower() == 'done.':
                plan_list = plan_list[:-1]
            plan_str = '\n'.join([f'{idx+1}. "{p}"' for idx, p in enumerate(plan_list)])

        sampled_frame_indices = get_sample_image_indices_sampled(task_start, task_end, fps//2, min_frames=max(16, min(64, 8*len(plan_list))))
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
        #     frames = [dataset[i]['exo_image'] for i in sampled_frame_indices]
        frames = [dataset[i]['exo_image'] for i in sampled_frame_indices]
        task_start_fid = int(dataset[task_start]['frame_index'])
        task_end_fid = int(dataset[task_end - 1]['frame_index']) + 1
        frame_ids = [int(dataset[i]['frame_index']) for i in sampled_frame_indices]
        video = [(i, f) for i, f in enumerate(frames)]
        instr = instruction.format(
            task=task,
            plans=plan_str,
        )
        supps = (ep_stem, task_id, task_start_fid, task_end_fid, frame_ids, frames)
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
        # generated_text = infer_instance([frames], [instr])[0]
        # jd, valid = post_process_func(generated_text, frame_ids)
        # print(jd, valid)
    input_queue.put(None)

def consumer_task(output_queue: Queue, dataset_path: str, output_dir: str, refresh_freq: int, test_name: Optional[str]=None) -> List[int]:
    dataset_name = Path(dataset_path).name
    output_dir = Path(output_dir)
    if test_name is not None:
        output_dir = output_dir / "test" / test_name

    frame_count, success_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        _, question, gen_txts, _supp = item
        ep_stem, task_id, task_start_fid, task_end_fid, frame_ids, frames = _supp

        jd, valid = post_process_func(gen_txts, frame_ids)
        if valid:
            success_count += 1
            sub_task_item = {
                "task_id": task_id,
                "task_start": task_start_fid,
                "task_end": task_end_fid,
                "results": jd,
            }
        else:
            sub_task_item = {
                "task_id": task_id,
                "task_start": task_start_fid,
                "task_end": task_end_fid,
                "generated_text": jd,
            }
            # logging.warning(f"Failed to parse LLM output for {ep_stem}, task_id {task_id}: {gen_txts}")
        jsl_path = output_dir / dataset_name / f"{ep_stem}.sub_task.jsonl"
        jsl_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(sub_task_item, jsl_path)

        frame_count += 1
        if frame_count % refresh_freq == 0:
            speed = refresh_freq / (time.time() - last_time)
            logging.info(
                f"Consumer throughput: {speed:.2f} frames/s. "
                f"Processed {frame_count} frames, {success_count} successful. "
                f"Failed rate: {(frame_count - success_count)/frame_count:.2%}."
            )
            last_time = time.time()

            # output inspection images
            if valid:
                fl_st_jl = sub_task_json_to_flattened_list(jd, task_start_fid, task_end_fid)
                frame2note = {item['frame_id']: item['sub_task'] for item in fl_st_jl}
                Fs = []
                for fid, F in zip(frame_ids, frames):
                    note = frame2note[fid]
                    F = draw_text_block(F, note)
                    Fs.append(F)
                frames = Fs
                video_path = output_dir / dataset_name / f"inspect/{ep_stem}.mp4"
                images_to_video(frames, video_path, fps=10, show_log=False)
            else:
                video_path = output_dir / dataset_name / f"inspect/{ep_stem}_failed.mp4"
                images_to_video(frames, video_path, fps=10, show_log=False)
            shutil.copy(jsl_path, output_dir / dataset_name / f"inspect/{ep_stem}.sub_task.jsonl")
