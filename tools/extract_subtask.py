"""
python tools/extract_subtask.py
"""

import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from core.data.reader import FastLerobotVLReader
from core.utils.common import write_jsonlines
load_dotenv()

def extract_sub_task_dataset(dataset_path, output_root):
    dn = Path(dataset_path).name
    output_dir = Path(output_root) / dn
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    reader = FastLerobotVLReader(dataset_path)
    if not reader.sub_task_available:
        print(f"Sub-task data not available in {dataset_path}, skipping...")
        return
    reader.ego_camera_key = None
    reader.exo_camera_key = None
    reader.loaded_camera_keys = []
    for ep in tqdm(reader.all_episode_names, desc=f"Extracting sub-task [{dn}]"):
        s, t = reader.get_episode_range(ep)
        output_path = output_dir / f"{ep}.sub_task.jsonl"
        if output_path.exists():
            print(f"Sub-task data for episode {ep} already exists, skipping...")
            continue
        jl = []
        for d in reader[s:t]:
            jl.append({
                'frame_id': int(d['frame_index']),
                'task': d['task'],
                'sub_task': d.get('sub_task', ''),
            })
        write_jsonlines(jl, output_path)

def extract_task_as_sub_task_dataset(dataset_path, output_root):
    dn = Path(dataset_path).name
    output_dir = Path(output_root) / dn
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    reader = FastLerobotVLReader(dataset_path)
    reader.ego_camera_key = None
    reader.exo_camera_key = None
    reader.loaded_camera_keys = []
    for ep in tqdm(reader.all_episode_names, desc=f"Extracting task as sub-task [{dn}]"):
        s, t = reader.get_episode_range(ep)
        output_path = output_dir / f"{ep}.sub_task.jsonl"
        if output_path.exists():
            print(f"Sub-task data for episode {ep} already exists, skipping...")
            continue
        jl = []
        for d in reader[s:t]:
            jl.append({
                'frame_id': int(d['frame_index']),
                'task': d['task'],
                'sub_task': d['task'],
            })
        write_jsonlines(jl, output_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    agrs = args.parse_args()
    _raw_datasets = agrs.dataset_paths
    sub_task_jsonl_dir = Path(os.getenv('ANNOTATION_SUB_TASK_DIR'))
    for dataset_path in _raw_datasets:
        # extract_sub_task_dataset(dataset_path, sub_task_jsonl_dir)
        extract_task_as_sub_task_dataset(dataset_path, sub_task_jsonl_dir)