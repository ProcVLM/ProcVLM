import sys
import json
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from PIL.Image import Image
from evqa.tools.data_picker import AnnotatedLerobotVLReader
from evqa.tools.functions.progress import p as p_function
from core.utils.common import images_to_video, load_jsonlines, append_jsonlines, episode2num_token
from core.utils.runner_utils import auto_split_datasets
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)

templates_dir: str
data_output_dir: str
json_output_dir: str
_cached_templates: Dict[str, List[str]] = {}

def get_real_actions(sub_tasks: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    real_actions = []
    for s, e, st in sub_tasks:
        if st and st.lower() not in ['done', 'done.', 'not done', 'not done.']:
            real_actions.append((s, e, st))
    return real_actions

def get_episode_stats(episode_frames: Dict, episode_name: str, fps: int) -> Optional[Dict[str, Any]]:
    episode_stats = {
        'len': len(episode_frames),
        'fps': fps,
    }
                
    start_fid = 0
    task_splits = []
    for fid in range(1, episode_stats['len']):
        if episode_frames[fid]['task'] != episode_frames[fid - 1]['task']:
            task_splits.append((start_fid, fid, episode_frames[fid - 1]['task']))
            start_fid = fid
    task_splits.append((start_fid, episode_stats['len'], episode_frames[-1]['task']))
    if len(task_splits) > 1:
        return None  # skip episodes with multiple tasks, as they are more complex to handle and less common
    
    episode_splits = []
    episode_plan = []
    current_sub_task_start_fid = 0
    for fid in range(1, episode_stats['len']):
        if episode_frames[fid]['sub_task'] != episode_frames[fid - 1]['sub_task']:
            episode_splits.append((current_sub_task_start_fid, fid, episode_frames[fid - 1]['sub_task']))
            episode_plan.append(episode_frames[fid - 1]['sub_task'])
            current_sub_task_start_fid = fid
    episode_splits.append((current_sub_task_start_fid, episode_stats['len'], episode_frames[episode_stats['len'] - 1]['sub_task']))
    episode_plan.append(episode_frames[episode_stats['len'] - 1]['sub_task'])

    real_sub_tasks_splits = get_real_actions(episode_splits)
    if len(real_sub_tasks_splits) == 0:
        return None  # skip episodes without real sub-tasks, as they are less interesting for procedural understanding

    episode_stats['task'] = task_splits[0][2]  # get the task name
    episode_stats['plan'] = episode_plan
    episode_stats['sub_task_splits'] = episode_splits
    episode_stats['real_sub_task_splits'] = real_sub_tasks_splits
    return episode_stats

def get_progress(episode_frames: List[Dict]) -> List[float]:
    p_list = p_function(episode_frames)
    # add a 'progress' field to each frame dict
    for frame, p in zip(episode_frames, p_list):
        frame['progress'] = float(p)
    return episode_frames

def check_disk_file_exists(task_id: str, dataset_name: str, episode_name: str) -> bool:
    """check if the QA pair is already saved on disk"""
    save_path = Path(json_output_dir) / task_id / dataset_name / f"{episode_name}.jsonl"
    return save_path.exists()

def check_disk_file_exists_for_any_task(dataset_name: str, episode_name: str, task_ids: List[str]) -> bool:
    """check if the QA pair is already saved on disk for any of the given task_ids"""
    for task_id in task_ids:
        if check_disk_file_exists(task_id, dataset_name, episode_name):
            return True
    return False

### iterator
def pick_process_understanding(
        datasets: List[str], 
        handlers: List[callable], 
        task_ids: Optional[List[str]] = None, 
        selections: Optional[Dict[str, Set[str]]] = None,
        exceptions: Optional[Dict[str, Set[str]]] = None,
        world_size: int = 1, 
        rank: int = 0, 
        split_factor: Optional[int] = None
    ):
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).name
        reader = AnnotatedLerobotVLReader(root=dataset_path, versions=['v2'])
        reader.set_run_mode(['SUB_TASK', 'PLANNED_ACTIONS', 'COT'])
        if task_ids:
            desc_str = f"[{dataset_name}: {', '.join(task_ids)}]"
        else:
            desc_str = f"[{dataset_name}]"
        for episode_name in tqdm(reader.vl_reader.all_episode_names, desc=desc_str+" Processing Episodes"):
            if episode2num_token(episode_name, num_tokens_factor=split_factor) % world_size != rank:
                continue
            if selections is not None and episode_name not in selections.get(dataset_name, set()):
                continue
            if exceptions is not None and episode_name in exceptions.get(dataset_name, set()):
                continue
            # fast skip: check if all outputs are exist for each task_id, if exist, skip this episode
            if task_ids and check_disk_file_exists_for_any_task(dataset_name, episode_name, task_ids):
                continue

            global_s, global_t = reader.vl_reader.get_episode_range(episode_name)
            episode_frames = reader[global_s:global_t]
            episode_stats = get_episode_stats(episode_frames, episode_name, reader.vl_reader.meta.fps)
            # call handlers
            if episode_stats:
                episode_frames = get_progress(episode_frames)
                for handler in tqdm(handlers, desc=desc_str+" Applying Handlers", leave=False):
                    handler(dataset_name, episode_name, episode_frames, episode_stats)

def load_handlers(task_ids: List[str]) -> List[callable]:
    module = sys.modules[__name__]
    handlers = []
    for task_id in task_ids:
        handler_name = f"episode_handler_{task_id}"
        if not hasattr(module, handler_name):
            raise ValueError(
                f"Handler '{handler_name}' not found. "
                f"Task ID '{task_id}' requires a function named '{handler_name}'."
            )
        handler = getattr(module, handler_name)
        if not callable(handler):
            raise TypeError(f"'{handler_name}' exists but is not callable.")
        handlers.append(handler)
    return handlers

def load_templates(template_name: str) -> List[str]:
    global _cached_templates
    if template_name in _cached_templates:
        return _cached_templates[template_name]
    template_path = Path(templates_dir) / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file '{template_path}' not found.")
    with open(template_path, 'r') as f:
        templates = [line.strip() for line in f if line.strip()]
    _cached_templates[template_name] = templates
    return templates

def QA_exists_on_disk(task_id: str, dataset_name: str, episode_name: str) -> bool:
    """check if the QA pair is already saved on disk"""
    save_path = Path(json_output_dir) / task_id / dataset_name / f"{episode_name}.jsonl"
    return save_path.exists()

def ensure_image_on_disk(dataset_name: str, camera_key: str, episode_name: str, frame_id: int, image_data: Image) -> str:
    """make sure the image is saved on disk, return the relative path to data_output_dir"""
    possible_path = Path(data_output_dir) / "images" / dataset_name / camera_key / episode_name / f"{frame_id:06d}.jpg"
    if possible_path.exists():
        return str(possible_path.relative_to(data_output_dir))
    save_path = possible_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_data.save(save_path)
    return str(save_path.relative_to(data_output_dir))

def ensure_video_on_disk(dataset_name: str, camera_key: str, episode_name: str, start_fid: int, end_fid: int, frame_images: List[Image], fps: int) -> str:
    """make sure the video is saved on disk, return the relative path to data_output_dir"""
    possible_path = Path(data_output_dir) / "videos" / dataset_name / camera_key / episode_name / f"{start_fid:06d}_{end_fid:06d}.mp4"
    if possible_path.exists():
        return str(possible_path.relative_to(data_output_dir))
    save_path = possible_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images_to_video(frame_images, save_path, fps=fps, show_log=False)
    return str(save_path.relative_to(data_output_dir))

def ensure_QA_on_disk(task_id: str, dataset_name: str, episode_name: str, QA_pairs: List[Dict[str, Any]]) -> str:
    """make sure the qa_pair is saved on disk, return the relative path to data_output_dir"""
    save_path = Path(json_output_dir) / task_id / dataset_name / f"{episode_name}.jsonl"
    append_jsonlines(QA_pairs, save_path)
    return str(save_path.relative_to(json_output_dir))

def preview_random_qa(task_id: str, dataset_path: str):
    dataset_name = Path(dataset_path).name
    task_dir = Path(json_output_dir) / task_id / dataset_name
    if not task_dir.exists():
        print(f"[Error] Task directory does not exist: {task_dir}")
        return
    jsonl_files = list(task_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"[Error] No .jsonl files found in: {task_dir}")
        return
    jsonl_path = random.choice(jsonl_files)
    with open(jsonl_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        print(f"[Error] No QA pairs found in: {jsonl_path}")
        return
    qa = random.choice(lines)
    images: list = qa.get("image", [])
    if not isinstance(images, list):
        images = [images]
    videos: list = qa.get("video", [])
    if not isinstance(videos, list):
        videos = [videos]
    conversations = qa.get("conversations", [])
    if not conversations or len(conversations) < 1:
        print(f"[Error] This QA pair has no conversations: {jsonl_path}")
        return
    human_msg = conversations[0]["value"]
    model_msg = conversations[1]["value"]
    for img_path in images:
        human_msg = human_msg.replace("<image>", img_path, 1)
    for vid_path in videos:
        human_msg = human_msg.replace("<video>", vid_path, 1)
    print("\n================ RANDOM QA PREVIEW ================\n")
    print(f"[Source File] {jsonl_path}")
    print("\n[Human Prompt]\n")
    print(human_msg)
    print("\n[Model Answer]\n")
    print(model_msg)
    print("\n====================================================\n")



# --- Add your episode handlers below ---
def _episode_handler_template(dataset_name: str, episode_name: str, episode_frames: List[Dict[str, Any]], episode_stats: Dict[str, Any]):
    """
    episode_frames: List of frame dicts, each containing:
        - 'frame_index': int
        - 'task': str, task description
        - 'sub_task': str, sub-task description
        - 'planned_actions': List[str], list of sub_tasks planned after this frame
        - 'progress': float, progress value computed from p_function
        - 'exo_image': PIL.Image, image from the third-person camera
        - 'exo_camera_key': str, key for the third-person camera
        - 'episode_path': str, path to the episode data
        - 'episode_name': str, name of the episode
        - 'chunk_name': str, name of the data chunk

    episode_stats: Dict containing:
        - 'len': int, number of frames in the episode
        - 'fps': int, frames per second of the episode
        - 'task_splits': List of tuples (start_fid, end_fid, task), splits of frames by task
        - 'sub_task_splits': Dict[task:str, List of tuples] (start_fid, end_fid, sub_task)
        - 'real_sub_task_splits': List of tuples (start_fid, end_fid, sub_task) that are not 'done' or 'not done'
        - 'plan': Dict[task:str, List of str], list of sub_tasks planned in the episode
    """
    templates = load_templates('template_name in evqa/templates')
    return

# expected training tokens: 20-40B tokens for 2B model
# a single frame contains ~1000-1500 tokens
#
# optimal generation ratio: 100 frames per episode (video)
# the number of generated QA pairs is proportional to the number of images it contains
# if a question contains N images, yield 100/N QA pairs from that episode
def episode_handler_a(dataset_name: str, episode_name: str, episode_frames: List[Dict[str, Any]], episode_stats: Dict[str, Any]):
    if check_disk_file_exists("a", dataset_name, episode_name):
        return
    
    from ecot.utils.sub_task_utils import get_sample_image_indices_bounded
    a1_template = load_templates('procedural_a1')[0]
    a2_template = load_templates('procedural_a2')[0]
    N = episode_stats['len']
    task = episode_stats['task']
    sub_tasks = episode_stats['real_sub_task_splits']
    QA_pairs = []

    def sample_for_frame_range(start_fid: int, end_fid: int, lb: int, ub: int) -> Tuple[List[str], str, str]:
        sampled_fids = get_sample_image_indices_bounded(start_fid, end_fid, max_frames=random.randint(lb, ub))
        sampled_image_paths = [
            ensure_image_on_disk(
                dataset_name,
                episode_frames[fid]['exo_camera_key'],
                episode_name,
                fid,
                episode_frames[fid]['exo_image']
            ) for fid in sampled_fids
        ]
        interleaved_parts = ""
        for idx, fid in enumerate(sampled_fids):
            interleaved_parts += f"<frame {idx}>\n<image>\n"
        actions = []
        start_idx = 0
        for idx, fid in enumerate(sampled_fids):
            if idx == 0: 
                continue
            prev_fid = sampled_fids[idx - 1]
            if episode_frames[fid]['sub_task'] != episode_frames[prev_fid]['sub_task']:
                actions.append((start_idx, idx - 1, episode_frames[sampled_fids[start_idx]]['sub_task']))
                start_idx = idx
        actions.append((start_idx, len(sampled_fids) - 1, episode_frames[sampled_fids[start_idx]]['sub_task']))
        actions = get_real_actions(actions)
        answer_list = [{"action_description": st, "start_frame": s, "end_frame": t} for s, t, st in actions]
        answer_str = json.dumps(answer_list, ensure_ascii=False)
        return sampled_image_paths, interleaved_parts, answer_str

    # a.1.
    sampled_image_paths, interleaved_parts, answer_str = sample_for_frame_range(0, N, 32, 96)
    prompt_part = a1_template.format(task=task)
    question = interleaved_parts + prompt_part
    QA_pairs.append({
        "image": sampled_image_paths,
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": answer_str
            }
        ],
    })

    # a.2.
    repeat_times = random.randint(0, 2)
    if repeat_times == 1:
        sampled_image_paths, interleaved_parts, answer_str = sample_for_frame_range(0, N, 32, 96)
        prompt_part = a2_template.format()
        question = interleaved_parts + prompt_part
        QA_pairs.append({
            "image": sampled_image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer_str
                }
            ],
        })
    elif repeat_times == 2:
        # random sample two distinct (start_sub_task_idx, end_sub_task_idx) pairs in C(len(sub_tasks), 2) options
        repeat_pairs = []
        for i in range(0, len(sub_tasks)):
            for j in range(i, len(sub_tasks)):
                if not (i == 0 and j == len(sub_tasks) - 1):  # exclude the pair that covers the whole episode
                    repeat_pairs.append((i, j))
        random.shuffle(repeat_pairs)
        repeat_pairs = repeat_pairs[:repeat_times]
        for start_sub_task_idx, end_sub_task_idx in repeat_pairs:
            start_fid = sub_tasks[start_sub_task_idx][0]
            end_fid = sub_tasks[end_sub_task_idx][1]
            sampled_image_paths, interleaved_parts, answer_str = sample_for_frame_range(start_fid, end_fid, 16, 64)
            prompt_part = a2_template.format()
            question = interleaved_parts + prompt_part
            QA_pairs.append({
                "image": sampled_image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer_str
                    }
                ],
            })

    # write to disk
    ensure_QA_on_disk("a", dataset_name, episode_name, QA_pairs)

def sample_previous_frames(cur_id: int, single_frame_P: float=0.3, min_frames: int=4, max_frames: int=16) -> List[int]:
    if random.random() < single_frame_P:
        return [cur_id]  # Return only the current frame
    num_frames = random.randint(min_frames, max_frames)
    start_id = max(0, cur_id - num_frames)
    return list(range(start_id, cur_id + 1))

def episode_handler_b(dataset_name: str, episode_name: str, episode_frames: List[Dict[str, Any]], episode_stats: Dict[str, Any]):
    if check_disk_file_exists("b", dataset_name, episode_name):
        return
    
    b1_template = load_templates('procedural_b1')[0]
    b2_template = load_templates('procedural_b2')[0]
    N = episode_stats['len']
    task = episode_stats['task']
    QA_pairs = []
    SKIP_RATIO = 0.4

    #
    no_skip_ids = []
    finished_ids = []
    for cur_id in range(N - 1):  # Ensure we don't go out of bounds
        if random.random() < SKIP_RATIO:
            continue
        prg = episode_frames[cur_id + 1]['progress']
        if prg < 1:
            no_skip_ids.append(cur_id)
        else:
            finished_ids.append(cur_id)
    random.shuffle(finished_ids)
    finished_ids = finished_ids[:10]
    no_skip_ids = sorted(no_skip_ids + finished_ids)

    # 
    for cur_id in no_skip_ids:  # Ensure we don't go out of bounds
        prg = episode_frames[cur_id + 1]['progress']
        try:
            if prg < 1:
                finished = False
            else:
                finish_reason = episode_frames[cur_id]['cot'].split(".")[1].strip()
                finished = True
        except Exception as e:
            logging.warning(f"Failed to determine finished reason for episode {episode_name} at frame {cur_id}: {e}")
            continue
        sampled_fids = sample_previous_frames(cur_id)
        sampled_image_paths = [
            ensure_image_on_disk(
                dataset_name,
                episode_frames[fid]['exo_camera_key'],
                episode_name,
                fid,
                episode_frames[fid]['exo_image']
            ) for fid in sampled_fids
        ]
        
        if random.randint(0, 1):
            # b.1.
            prompt = b1_template.format(task=task)
            question = ''.join(['<image>\n' for _ in sampled_fids]) + prompt
            if finished:
                answer = f"No action required. {finish_reason}."
            else:
                answer = episode_frames[cur_id + 1]['sub_task']
            QA_pairs.append({
                "image": sampled_image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
            })
        else:
            # b.2.
            prompt = b2_template.format(task=task)
            question = ''.join(['<image>\n' for _ in sampled_fids]) + prompt
            if finished:
                answer = f"No action required. {finish_reason}."
            else:
                planned_actions = episode_frames[cur_id + 1]['planned_actions']
                answer = '\n'.join([f"{i+1}. {action}" for i, action in enumerate(planned_actions)])
            QA_pairs.append({
                "image": sampled_image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
            })

    # write to disk
    ensure_QA_on_disk("b", dataset_name, episode_name, QA_pairs)

def episode_handler_c(dataset_name: str, episode_name: str, episode_frames: List[Dict[str, Any]], episode_stats: Dict[str, Any]):
    if check_disk_file_exists("c", dataset_name, episode_name):
        return
    
    template = load_templates('procedural_c')[0]
    N = episode_stats['len']
    task = episode_stats['task']
    QA_pairs = []
    SKIP_RATIO = 0.2

    #
    no_skip_ids = []
    finished_ids = []
    for cur_id in range(N):  # Ensure we don't go out of bounds
        if random.random() < SKIP_RATIO:
            continue
        prg = episode_frames[cur_id]['progress']
        if prg < 1:
            no_skip_ids.append(cur_id)
        else:
            finished_ids.append(cur_id)
    random.shuffle(finished_ids)
    finished_ids = finished_ids[:10]
    no_skip_ids = sorted(no_skip_ids + finished_ids)

    # 
    for cur_id in no_skip_ids:  # Ensure we don't go out of bounds
        prg = episode_frames[cur_id]['progress']
        try:
            if prg < 1:
                finished = False
            else:
                finish_reason = episode_frames[cur_id]['cot'].split(".")[1].strip()
                finished = True
        except Exception as e:
            logging.warning(f"Failed to determine finished reason for episode {episode_name} at frame {cur_id}: {e}")
            continue
        sampled_fids = sample_previous_frames(cur_id)
        sampled_image_paths = [
            ensure_image_on_disk(
                dataset_name,
                episode_frames[fid]['exo_camera_key'],
                episode_name,
                fid,
                episode_frames[fid]['exo_image']
            ) for fid in sampled_fids
        ]

        # c.1.
        prompt = template.format(task=task)
        question = ''.join(['<image>\n' for _ in sampled_fids]) + prompt
        if finished:
            answer = f"{finish_reason}.\nTherefore, the estimated progress is <progress>100%</progress>."
        else:
            planned_actions = episode_frames[cur_id]['planned_actions']
            steps_str = '\n'.join([f"{i+1}. {action}" for i, action in enumerate(planned_actions)])
            answer = f"The following actions are required: \n{steps_str}\n"
            answer += f"Therefore, the estimated progress is <progress>{prg*100:.2f}%</progress>."
        QA_pairs.append({
            "image": sampled_image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ],
        })
    
    # write to disk
    ensure_QA_on_disk("c", dataset_name, episode_name, QA_pairs)


# --- Main Execution ---
if __name__ == "__main__":   
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_paths", type=str, nargs='+', help="List of dataset paths to process.", required=True)
    argparser.add_argument("--json_output_dir", type=str, help="Directory to save output JSON files.", required=True)
    argparser.add_argument("--data_output_dir", type=str, help="Directory to save processed images or videos.", required=True)
    argparser.add_argument("--task_ids", type=str, nargs='+', help="List of task IDs to process.", choices=["a", "b", "c"], required=True)
    argparser.add_argument("--templates_dir", type=str, help="Path to templates directory.", default="evqa/templates")
    argparser.add_argument("--preview", action='store_true', help="Preview a random QA pair after processing.")
    argparser.add_argument("--skip_processing", action='store_true', help="Skip data processing step.")
    argparser.add_argument("--world_size", type=int, default=1, help="Number of parallel processes for data processing.")
    argparser.add_argument("--rank", type=int, default=0, help="Rank of the current process for parallel data processing.")
    argparser.add_argument("--split_factor", type=int, choices=[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], default=None, help="Modulo for dataset splitting, used as num_tokens_factor in episode2num_token(). Set to different prime numbers for different runs to get different splits. If not set, no modulo is applied.")
    argparser.add_argument("--exception_path", type=str, default=None, help="Path to JSONL file containing episodes to be excepted from processing. The format should be a list of objects with 'dataset_name' and 'episode_name' fields.")
    argparser.add_argument("--selection_path", type=str, default=None, help="Path to JSONL file containing selected episodes to process. The format should be a list of objects with 'dataset_name' and 'episode_name' fields.")
    args = argparser.parse_args()

    templates_dir = args.templates_dir
    data_output_dir = args.data_output_dir
    json_output_dir = args.json_output_dir
    raw_datasets = args.dataset_paths
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(raw_datasets, args.world_size, args.rank)

    selection = None
    if args.selection_path is not None:
        selection_jl = load_jsonlines(args.selection_path)
        selection = {}
        for item in selection_jl:
            dn = item['dataset_name']
            en = item['episode_name']
            if dn not in selection:
                selection[dn] = set()
            selection[dn].add(en)
        print(f"Loaded selection for {len(selection)} datasets.")
    
    exception = None
    if args.exception_path is not None:
        exception_jl = load_jsonlines(args.exception_path)
        exception = {}
        for item in exception_jl:
            dn = item['dataset_name']
            en = item['episode_name']
            if dn not in exception:
                exception[dn] = set()
            exception[dn].add(en)
        print(f"Loaded exceptions for {len(exception)} datasets.")

    if not args.skip_processing:
        pick_process_understanding(
            datasets=DATASETS,
            handlers=load_handlers(args.task_ids),
            task_ids=args.task_ids,
            selections=selection,
            exceptions=exception,
            world_size=WORLD_SIZE,
            rank=RANK,
            split_factor=args.split_factor
        )
    
    if args.preview:
        for task_id in args.task_ids:
            for dataset_path in DATASETS:
                preview_random_qa(task_id, dataset_path)