"""
CUDA_VISIBLE_DEVICES=0,1 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 0 &
CUDA_VISIBLE_DEVICES=2,3 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 1 &
CUDA_VISIBLE_DEVICES=4,5 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 2 &
CUDA_VISIBLE_DEVICES=6,7 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 3 &
wait
"""

import os
import argparse
from dotenv import load_dotenv
from core.utils.runner_utils import auto_split_datasets
load_dotenv()

args = argparse.ArgumentParser()
args.add_argument("--rank", type=int, default=0)
args.add_argument("--world_size", type=int, default=1)
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = args.parse_args()
_raw_datasets = args.dataset_paths

# DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank

from grounding.postprocess.clean_bbox import stage2_clean

# eps = [
#     'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
#     'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
#     'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
#     'episode_006678', 'episode_004399'
# ]
if __name__ == "__main__":
    bbox_jsonl_path=os.getenv('OUTPUT_BBOX_JSONL_DIR')
    clean_bbox_jsonl_path=os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR')
    stage2_clean(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK) #, test_eps=eps)