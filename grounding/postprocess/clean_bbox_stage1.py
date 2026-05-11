"""
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 0 & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 1 &
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 2 &
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 3 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 4 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 5 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 6 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 7 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 8 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 9 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 10 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 11 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 12 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 13 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 14 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 15 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 16 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 17 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 18 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 19 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 20 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 21 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 22 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 23 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 24 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 25 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 26 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 27 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 28 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 29 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 30 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 31 &
wait
"""

import os
import argparse
from core.utils.runner_utils import auto_split_datasets
from dotenv import load_dotenv
load_dotenv()
# eps = [
#     'episode_000000'
#     # 'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
#     # 'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
#     # 'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
#     # 'episode_006678', 'episode_004399'
# ]

args = argparse.ArgumentParser()
args.add_argument("--rank", type=int, default=0)
args.add_argument("--world_size", type=int, default=1)
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = args.parse_args()
_raw_datasets = args.dataset_paths

DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
# DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank

from grounding.postprocess.clean_bbox import stage1_clean

if __name__ == "__main__":
    bbox_jsonl_path=os.getenv('OUTPUT_BBOX_JSONL_DIR')
    clean_bbox_jsonl_path=os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR')
    stage1_clean(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK) #, test_eps=eps)