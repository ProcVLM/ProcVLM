"""
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 0 & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 1 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 2 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 3 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 4 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 5 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 6 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 7 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 8 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 9 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 10 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 11 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 12 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 13 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 14 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 15 &
wait
"""

import os
import argparse
from dotenv import load_dotenv
from core.utils.runner_utils import auto_split_datasets
from grounding.postprocess.clean_bbox import stage3_clean, bbox_generation_by_velocity_interpolation
load_dotenv()

args = argparse.ArgumentParser()
args.add_argument("--rank", type=int, default=0)
args.add_argument("--world_size", type=int, default=1)
args.add_argument("--test_run", action='store_true', help="whether to run in test mode with limited data for quick checking", default=False)
args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args.add_argument("--enable_auto_dp", action='store_true', help="whether to enable automatic data parallel splitting based on world size")
args = args.parse_args()
_raw_datasets = args.dataset_paths

if args.test_run:
    eps = [   # needs update #
        'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
        'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
        'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
        'episode_006678', 'episode_004399'
    ]

if args.enable_auto_dp:
    DATASETS, WORLD_SIZE, RANK = auto_split_datasets(_raw_datasets, args.world_size, args.rank)
else:
    DATASETS, WORLD_SIZE, RANK = _raw_datasets, args.world_size, args.rank

if __name__ == "__main__":
    if not args.test_run:
        bbox_jsonl_path=os.getenv('OUTPUT_BBOX_JSONL_DIR')
        clean_bbox_jsonl_path=os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR')
        stage3_clean(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK)
        bbox_generation_by_velocity_interpolation(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK)
    else:
        bbox_jsonl_path=os.getenv('OUTPUT_BBOX_JSONL_DIR')    # needs update #
        clean_bbox_jsonl_path=os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR')
        stage3_clean(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK, test_eps=eps)   # needs update #
        bbox_generation_by_velocity_interpolation(DATASETS, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=WORLD_SIZE, rank=RANK, test_eps=eps)