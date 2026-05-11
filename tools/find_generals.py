import os
import argparse
from dotenv import load_dotenv
from core.data.generals import (
    DatasetStatus,
)
from core.utils.common import load_parquet
from core.data.reader import FastLerobotVLReader
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
load_dotenv()

def print_list(items: list):
    print("[")
    for it in items:
        print(f'    "{it}",')
    print("]    " + f"# {len(items)} items")

args = argparse.ArgumentParser()
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = args.parse_args()
DATASETS = args.dataset_paths


def label_dataset(dpath: str, mode: callable = all) -> DatasetStatus:
    """
    check:
    1. if dataset has existing .plan.jsonl
    2. if dataset has existing .sub_task.jsonl
    3. if dataset has existing .cot.jsonl
    4. if dataset has existing .bbox.jsonl
    """
    plan_base_dir = Path(os.getenv('OUTPUT_PLAN_JSONL_DIR'))
    sub_task_base_dir = Path(os.getenv('OUTPUT_SUB_TASK_JSONL_DIR'))
    cot_base_dir = Path(os.getenv('OUTPUT_COT_JSONL_DIR'))
    bbox_base_dir = Path(os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR') + '/interpolated')

    dataset = FastLerobotVLReader(dpath)
    dataset_name = dataset.name
    # all_parquets = load_parquet(dpath)
    # all_parquets_stem = set([Path(p).stem for p in all_parquets])
    all_parquets_stem = set(dataset.all_episode_names)

    status = DatasetStatus.VANILLA

    plan_dir = plan_base_dir / dataset_name 
    if plan_dir.exists() and mode((plan_dir / f"{st}.plan.jsonl").exists() for st in all_parquets_stem):
        status |= DatasetStatus.WITH_PLAN
    else:
        print(f"Dataset {dataset_name} is missing plans:")
        for st in all_parquets_stem:
            if not (plan_dir / f"{st}.plan.jsonl").exists():
                print(f"Missing plan for {dataset_name}/{st}")
    
    sub_task_dir = sub_task_base_dir / dataset_name
    if sub_task_dir.exists() and mode((sub_task_dir / f"{st}.sub_task.jsonl").exists() for st in all_parquets_stem):
        status |= DatasetStatus.WITH_SUB_TASK

    cot_dir = cot_base_dir / dataset_name
    if cot_dir.exists() and  mode((cot_dir / f"{st}.cot.jsonl").exists() for st in all_parquets_stem):
        status |= DatasetStatus.WITH_COT
    
    bbox_dir = bbox_base_dir / dataset_name
    if bbox_dir.exists() and mode((bbox_dir / f"{st}.bbox.jsonl").exists() for st in all_parquets_stem):
        status |= DatasetStatus.WITH_BBOX

    return status

if __name__ == "__main__":
    DATASETS_WITH_SUBTASK = []
    DATASETS_WITH_PLAN = []
    DATASETS_WITH_COT = []
    DATASETS_WITH_BBOX = []
    for d in DATASETS:
        status = label_dataset(d, all)
        if status & DatasetStatus.WITH_PLAN:
            DATASETS_WITH_PLAN.append(d)
        if status & DatasetStatus.WITH_SUB_TASK:
            DATASETS_WITH_SUBTASK.append(d)
        if status & DatasetStatus.WITH_COT:
            DATASETS_WITH_COT.append(d)
        if status & DatasetStatus.WITH_BBOX:
            DATASETS_WITH_BBOX.append(d)

    print(f"dataset withou plan: {len(DATASETS) - len(DATASETS_WITH_PLAN)} / {len(DATASETS)}")

    print("DATASETS_WITH_PLAN = ", end='')
    print_list(DATASETS_WITH_PLAN)

    print("DATASETS_WITH_SUBTASK = ", end='')
    print_list(DATASETS_WITH_SUBTASK)

    print("DATASETS_WITH_COT = ", end='')
    print_list(DATASETS_WITH_COT)

    print("DATASETS_WITH_BBOX = ", end='')
    print_list(DATASETS_WITH_BBOX)
