"""
python archive_jsonl.py --category subtask --name subtask_v1
python ecot/postprocess/flatten_sub_task.py
"""

import os
import logging
import pickle
import argparse
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from core.data.generals import EXO_CAMERA_MAP
from core.data.reader import FastLerobotVLReader
from core.utils.common import load_jsonlines, write_jsonlines
from ecot.utils.sub_task_utils import sub_task_json_to_flattened_list
load_dotenv()

#
argparser = argparse.ArgumentParser()
argparser.add_argument("--src", type=str, required=True, help="source directory of jsonl files")
argparser.add_argument("--dst", type=str, required=True, help="destination directory of jsonl files")
argparser.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = argparser.parse_args()
_raw_datasets = args.dataset_paths

in_dir = args.src
out_dir = args.dst
borders_dir = os.getenv('OUTPUT_PLAN_JSONL_DIR')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)
logging.getLogger("vl_reader").setLevel(logging.WARNING)

def work(dataset_path: str):
    # ----- load dataset -----
    dataset = FastLerobotVLReader(
        root=dataset_path,
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        return_record_meta=True,
    )
    try:
        with open(Path(borders_dir) / dataset.name / "borders.pkl", 'rb') as f:
            task_borders = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"Borders file not found, assume this dataset has native subtask annotations and skip: {dataset_path}")
        return

    # ----- process each episode -----
    total_frames = len(task_borders) - 1
    ep_stem2tidx = {}
    for _tidx, task_start in enumerate(tqdm(task_borders[:-1], total=total_frames, desc=f"Building Episode Index for {dataset.name}")):
        ep_stem = dataset.get_episode_name_by_global_index(task_start)
        if ep_stem not in ep_stem2tidx:
            ep_stem2tidx[ep_stem] = [_tidx]
        else:
            ep_stem2tidx[ep_stem].append(_tidx)
    logging.info(f"Total episodes to process: {len(ep_stem2tidx)}")
    
    for ep in tqdm(dataset.all_episode_names, desc=f"[{dataset.name}] Flattening SubTask", total=len(dataset.all_episode_names)):
        out_jl_path = Path(out_dir) / dataset.name / f"{ep}.sub_task.jsonl"
        if out_jl_path.exists():
            # logging.info(f"Episode jsonl file already exists, skipping: {out_jl_path}")
            continue
        flatten_jl = []
        jl_path = Path(in_dir) / dataset.name / f"{ep}.sub_task.jsonl"
        if not jl_path.exists():
            logging.warning(f"Episode jsonl file not found: {jl_path}, this means not all episodes have sub-tasks in '{dataset.name}', skipping...")
            continue
        jl = load_jsonlines(jl_path)
        if any('task_start' not in item for item in jl):
            logging.error(f"Old format jsonl file found (missing 'task_start'), please regenerate the jsonl file: {jl_path}")
            continue
        assert len(set(item['task_start'] for item in jl)) == len(jl), f"Duplicate task_start found in {jl_path}"
        tid2jd = {item['task_start']: item for item in jl}
        for _tidx in ep_stem2tidx.get(ep, []):
            task_start = task_borders[_tidx]
            task_start_fid = task_start - dataset.get_episode_range(ep)[0]
            if task_start_fid not in tid2jd:
                logging.warning(f"Task start frame id {task_start_fid} not found in {jl_path}, maybe the task is too short, skipping...")
                continue
            jd = tid2jd[task_start_fid]
            if 'results' not in jd:
                logging.info(f"No subtasks found for task start at {task_start_fid} in {jl_path}, this means the generation failed, skipping...")
                continue
            new = sub_task_json_to_flattened_list(jd['results'], jd['task_start'], jd['task_end'])
            flatten_jl.extend(new)
        write_jsonlines(flatten_jl, out_jl_path)

def pad_with_task(dataset_path: str):
    # ----- load dataset -----
    dataset = FastLerobotVLReader(
        root=dataset_path,
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        return_record_meta=True,
    )

    working_dir = Path(out_dir) / dataset.name
    if not working_dir.exists():
        logging.warning(f"Output directory not found, this means no flattened sub-task jsonl files exist, skipping: {working_dir}")
        return

    for ep in tqdm(dataset.all_episode_names, desc=f"Pad with task {dataset.name}", total=len(dataset.all_episode_names)):
        jl_path = working_dir / f"{ep}.sub_task.jsonl"
        if not jl_path.exists():
            logging.warning(f"Episode jsonl file not found, all frames will be padded with task: {jl_path}")
            jl = []
        else:
            jl = load_jsonlines(jl_path)
        ep_start, ep_end = dataset.get_episode_range(ep)
        en_len = ep_end - ep_start
        existing_fids = set([item['frame_id'] for item in jl])
        for fid in range(en_len):
            if fid not in existing_fids:
                data = dataset[ep_start + fid]
                task = data['task']
                new_item = {
                    'frame_id': fid,
                    'task': task,
                    'sub_task': task,
                }
                jl.append(new_item) 
        # sort jl by frame_id
        jl = sorted(jl, key=lambda x: x['frame_id'])
        write_jsonlines(jl, jl_path)

def pad_with_null(dataset_path: str):
    # ----- load dataset -----
    dataset = FastLerobotVLReader(
        root=dataset_path,
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        return_record_meta=True,
    )

    working_dir = Path(out_dir) / dataset.name
    if not working_dir.exists():
        logging.warning(f"Output directory not found, this means no flattened sub-task jsonl files exist, skipping: {working_dir}")
        return

    for ep in tqdm(dataset.all_episode_names, desc=f"Pad with null {dataset.name}", total=len(dataset.all_episode_names)):
        jl_path = working_dir / f"{ep}.sub_task.jsonl"
        if not jl_path.exists():
            logging.warning(f"Episode jsonl file not found, all frames will be padded with null: {jl_path}")
            jl = []
        else:
            jl = load_jsonlines(jl_path)
        ep_start, ep_end = dataset.get_episode_range(ep)
        en_len = ep_end - ep_start
        existing_fids = set([item['frame_id'] for item in jl])
        for fid in range(en_len):
            if fid not in existing_fids:
                new_item = {
                    'frame_id': fid,
                    'task': None,
                    'sub_task': None,
                }
                jl.append(new_item) 
        # sort jl by frame_id
        jl = sorted(jl, key=lambda x: x['frame_id'])
        write_jsonlines(jl, jl_path)

if __name__ == "__main__":
    for dataset_path in tqdm(_raw_datasets):
        work(dataset_path)
    # for dataset_path in tqdm(_raw_datasets):
    #     pad_with_task(dataset_path)
    for dataset_path in tqdm(_raw_datasets):
        pad_with_null(dataset_path)

