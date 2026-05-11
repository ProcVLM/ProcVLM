"""
python tools/task_borders.py
"""
import logging
import os
import pickle
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from core.data.reader import FastLerobotVLReader
from core.data.generals import (
    EXO_CAMERA_MAP
)
from ecot.utils.sub_task_utils import get_sample_image_indices_sampled
load_dotenv()

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
argparser.add_argument("--yes", action='store_true', help="Automatic yes to prompts; run non-interactively.")
args = argparser.parse_args()
_raw_datasets = args.dataset_paths

print("Datasets to process:", _raw_datasets)
if not args.yes:
    proceed = input("Do you want to proceed? (y/n) ")
    if proceed.lower() != 'y':
        print("Aborting.")
        exit(0)
        
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)

OUTPUT_PLAN_JSONL_DIR = os.getenv("OUTPUT_PLAN_JSONL_DIR")

def work(dataset_path):
    pkl_path = Path(OUTPUT_PLAN_JSONL_DIR) / Path(dataset_path).name / "borders.pkl"
    if pkl_path.exists():
        os.unlink(pkl_path)
        logging.info(f"Removed existing file {pkl_path}")
    
    # ----- load dataset -----
    try:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            return_record_meta=True,
        )
        dataset.ego_camera_key = None
        dataset.loaded_camera_keys = [dataset.exo_camera_key]
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return

    task_borders = []
    last_task, last_ep = None, None
    for idx, data in tqdm(enumerate(dataset), desc=f"[{dataset.name}] Pre-iterating", total=len(dataset)):
        ep_stem = data['episode_name']
        if ep_stem != last_ep:
            last_ep = ep_stem
            last_task = None
        task_id = int(data['task_index'])
        if task_id != last_task:
            if task_borders:
                logging.info(f"Found task {task_id} from index {task_borders[-1]} to {idx-1} in episode {ep_stem}, total frames {idx - task_borders[-1]}")
            task_borders.append(idx)
            last_task = task_id
    task_borders.append(len(dataset))  # end border

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(task_borders, f)
    logging.info(f"Task borders for dataset '{dataset.name}' saved to {pkl_path}")

def statistics():
    overall_tasks = []
    overall_datasets = []
    task_sampled_lens = []
    for dataset_path in _raw_datasets:
        pkl_path = Path(OUTPUT_PLAN_JSONL_DIR) / Path(dataset_path).name / "borders.pkl"
        if not pkl_path.exists():
            logging.warning(f"No borders.pkl found for dataset '{dataset_path}'. Please run the work function first.")
            return
        
        with open(pkl_path, 'rb') as f:
            task_borders = pickle.load(f)
        
        num_tasks = len(task_borders) - 1
        task_lengths = [task_borders[i+1] - task_borders[i] for i in range(num_tasks)]
        overall_tasks.extend(task_lengths)
        overall_datasets.extend([Path(dataset_path).name] * num_tasks)
        
        print(f"---------- '{Path(dataset_path).name}' Statistics ----------")
        print(f"Total tasks: {num_tasks}")
        print(f"Average task length: {sum(task_lengths)/num_tasks:.2f} frames")
        print(f"Max task length: {max(task_lengths)} frames")
        print(f"Min task length: {min(task_lengths)} frames")


        ''' add sampling logic to estimate sample rate '''
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            return_record_meta=True,
        )
        dataset.ego_camera_key = None
        dataset.loaded_camera_keys = [dataset.exo_camera_key]
        fps = int(dataset.meta.fps)
        for i in range(len(task_borders)-1):
            ts, te = task_borders[i], task_borders[i+1]
            if te - ts < 10:
                continue
            slen = len(get_sample_image_indices_sampled(ts, te, fps))
            assert slen >= 10, f"{ts}, {te}, {slen}"
            task_sampled_lens.append(slen)

        if task_sampled_lens:
            print(f"---------- '{Path(dataset_path).name}' Sampled Frame Statistics ----------")
            print(f"Total sampled tasks: {len(task_sampled_lens)}")
            print(f"Average sampled frames per episode: {sum(task_sampled_lens)/len(task_sampled_lens):.2f} frames")
            print(f"Max sampled frames: {max(task_sampled_lens)} frames")
            print(f"Min sampled frames: {min(task_sampled_lens)} frames")

    if overall_tasks:
        print("========== Overall Statistics ==========")
        print(f"Total tasks across all datasets: {len(overall_tasks)}")
        print(f"Average task length: {sum(overall_tasks)/len(overall_tasks):.2f} frames")
        print(f"Max task length: {max(overall_tasks)} frames, at '{overall_datasets[overall_tasks.index(max(overall_tasks))]}'")
        print(f"Min task length: {min(overall_tasks)} frames, at '{overall_datasets[overall_tasks.index(min(overall_tasks))]}'")
    if task_sampled_lens:
        print("========== Overall Sampled Frame Statistics ==========")
        print(f"Total sampled tasks across all datasets: {len(task_sampled_lens)}")
        print(f"Average sampled frames per episode: {sum(task_sampled_lens)/len(task_sampled_lens):.2f} frames")
        print(f"Median sampled frames per episode: {sorted(task_sampled_lens)[len(task_sampled_lens)//2]} frames")
        print(f"Max sampled frames: {max(task_sampled_lens)} frames")
        print(f"Min sampled frames: {min(task_sampled_lens)} frames")
    print("========================================")

def check_task_with_episode():
    for dataset_path in _raw_datasets:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
            return_record_meta=True,
        )
        dataset.ego_camera_key = None
        dataset.loaded_camera_keys = [dataset.exo_camera_key]

        for ep_stem in tqdm(dataset.all_episode_names):
            task_ids = set()
            ts, te = dataset.get_episode_range(ep_stem)
            last_task = None
            task_sts = []
            for idx in range(ts, te):
                data = dataset[idx]
                task_ids.add(int(data['task_index']))
                if int(data['task_index']) != last_task:
                    task_sts.append(idx)
                    last_task = int(data['task_index'])
            task_sts.append(te)
            task_lens = [int(task_sts[i+1] - task_sts[i]) for i in range(len(task_sts)-1)]
            
            if len(task_ids) > 1:
                print("--------------------------------------------------")
                print(f"Episode {ep_stem} in dataset '{dataset.name}' has {len(task_ids)} tasks: {task_ids}")
                print(f"Task lengths: {task_lens}")
            if any(l < 10 for l in task_lens):
                print("--------------------------------------------------")
                print(f"Episode {ep_stem} in dataset '{dataset.name}' has short tasks: {task_lens}")


if __name__ == "__main__":
    for dataset_path in _raw_datasets:
        work(dataset_path)
    
    # # Uncomment below to use multiprocessing
    # with mp.Pool(processes=min(len(_raw_datasets), mp.cpu_count())) as pool:
    #     pool.map(work, _raw_datasets)
    
    # statistics()
    # check_task_with_episode()