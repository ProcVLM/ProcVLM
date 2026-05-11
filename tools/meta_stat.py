"""
python tools/meta_stat.py --dataset_paths $DATASETS
"""
import os
import logging
import argparse
import pickle
from dotenv import load_dotenv
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from core.data.reader import LeRobotDatasetMetadata, FastLerobotVLReader
from core.utils.common import load_jsonlines
load_dotenv()

_raw_datasets = []
_cache_d = {}
def fetch_dataset(dp):
    if dp not in _cache_d:
        _cache_d[dp] = FastLerobotVLReader(
            root=dp,
            # exo_name=EXO_CAMERA_MAP.get(Path(dp).name, None),
            # ego_name=EGO_CAMERA_MAP.get(Path(dp).name, None),
        )
    return _cache_d[dp]

def print_list(datasets: list):
    print("[")
    for ds in datasets:
        print(f'{ds}')
    print("]")

def statistics():
    frame_nums = []
    ep_nums = []
    for dataset_path in _raw_datasets:
        droot = Path(dataset_path)
        dname = droot.name
        meta = LeRobotDatasetMetadata(repo_id=dname, root=droot)
        ep_nums.append(meta.total_episodes)
        frame_nums.append(meta.total_frames)
    print("Dataset Statistics:")
    print(f"Total datasets: {len(_raw_datasets)}")
    print(f"Total episodes: {sum(ep_nums)}")
    print(f"Total frames: {sum(frame_nums)}")

    print("ep_nums = ")
    print_list(ep_nums)
    print("frame_nums = ")
    print_list(frame_nums)

OUTPUT_PLAN_JSONL_DIR = os.getenv("OUTPUT_PLAN_JSONL_DIR")
def statistics_task():
    overall_tasks = []
    task_nums = []
    for dataset_path in _raw_datasets:
        pkl_path = Path(OUTPUT_PLAN_JSONL_DIR) / Path(dataset_path).name / "borders.pkl"
        if not pkl_path.exists():
            logging.warning(f"No borders.pkl found for dataset '{dataset_path}'. Please run the work function first.")
            task_nums.append(-1)
            continue
        
        with open(pkl_path, 'rb') as f:
            task_borders = pickle.load(f)
        
        num_tasks = len(task_borders) - 1
        task_lengths = [task_borders[i+1] - task_borders[i] for i in range(num_tasks)]
        overall_tasks.extend(task_lengths)
        task_nums.append(num_tasks)
    print("task_nums = ")
    print_list(task_nums)

OUTPUT_SUB_TASK_JSONL_DIR = os.getenv("ANNOTATION_SUB_TASK_DIR")
def statistics_subtask():
    subtask_nums = []
    f_nums = []
    st_strs = []
    format_strs = []
    for dp in _raw_datasets:
        ds = fetch_dataset(dp)
        dataset_name = ds.name
        f_nums.append(ds.total_frames)
        jl_path = Path(OUTPUT_SUB_TASK_JSONL_DIR) / dataset_name
        if not jl_path.exists():
            subtask_nums.append(0)
            continue
        episode_names = ds.all_episode_names
        st_cnt = 0
        for ep in tqdm(episode_names):
            ep_jl_path = jl_path / f"{ep}.sub_task.jsonl"
            if not ep_jl_path.exists():
                continue
            jl = load_jsonlines(ep_jl_path)
            all_frame_ids = set([item['frame_id'] for item in jl if item.get('sub_task')])
            st_cnt += len(all_frame_ids)
        subtask_nums.append(st_cnt)
        cstr = f"{dataset_name}: {st_cnt} / {ds.total_frames} = {st_cnt / ds.total_frames:.4f}"
        # cstr = f"{st_cnt / ds.total_frames:.1%}"
        st_strs.append(cstr)
        format_strs.append(f"{st_cnt / ds.total_frames:.2%}") # only "xx.xx%"
    print("sub_tasks = ")
    print_list(st_strs)
    print(f"{sum(subtask_nums)} / {sum(f_nums)} = {sum(subtask_nums) / sum(f_nums):.1%}")
    print("sub_tasks = ")
    print_list(format_strs)

OUTPUT_BBOX_JSONL_DIR = os.getenv("ANNOTATION_BBOX_DIR")
def statistics_bbox():
    itcnt, allcnt = 0, 0
    total_frames = 0
    box_counter = Counter()
    box_coverage = {}
    box_coverage_strs = []
    format_strs = []
    for dp in _raw_datasets:
        d = fetch_dataset(dp)
        dn = d.name
        eps = d.all_episode_names
        in_base = Path(OUTPUT_BBOX_JSONL_DIR)
        box_coverage[dn] = (0, 0) # (with at least one box, total frames)
        for ep_stem in eps:
            pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
            st, ed = d.get_episode_range(ep_stem)
            ep_len = ed - st
            box_coverage[dn] = (box_coverage[dn][0], box_coverage[dn][1] + ep_len)
            if not pej.exists():
                continue
            jl = load_jsonlines(pej)
            total_frames += len(jl)
            for item in jl:
                num_boxes = len(item['bboxes'])
                if item['camera_key'] == d.exo_camera_key and num_boxes > 0:
                    box_coverage[dn] = (box_coverage[dn][0] + 1, box_coverage[dn][1])
                box_counter[num_boxes] += 1
                allcnt += num_boxes
                for b in item['bboxes']:
                    if str(b['score']) == 'interpolated':
                        itcnt += 1
        box_coverage_strs.append(f"{dn}: {box_coverage[dn][0]}/{box_coverage[dn][1]} = {box_coverage[dn][0]/box_coverage[dn][1]:.2%}")
        format_strs.append(f"{box_coverage[dn][0]/box_coverage[dn][1]:.2%}")
    print("========================================")
    print(f"All {len(_raw_datasets)} datasets have {allcnt} boxes in total after all processing stages.")
    print(f"{itcnt} boxes are generated by interpolation, interpolation ratio: {itcnt/allcnt:.2%}")
    print(f"Average boxes per frame after interpolation: {allcnt/total_frames:.4f}")
    print(f"Overall box coverage after interpolation: {sum([v[0] for v in box_coverage.values()])}/{sum([v[1] for v in box_coverage.values()])} = {sum([v[0] for v in box_coverage.values()])/sum([v[1] for v in box_coverage.values()]):.2%}")
    print(f"Box count distribution after interpolation: {box_counter}")
    print("========================================")
    print("bbox_nums = ")
    print_list(box_coverage_strs)
    print("bbox_nums = ")
    print_list(format_strs)

OUTPUT_COT_JSONL_DIR = os.getenv("ANNOTATION_COT_DIR")
def statistics_cot():
    cot_nums = []
    f_nums = []
    st_strs = []
    format_strs = []
    for dp in _raw_datasets:
        ds = fetch_dataset(dp)
        dataset_name = ds.name
        f_nums.append(ds.total_frames)
        jl_path = Path(OUTPUT_COT_JSONL_DIR) / dataset_name
        if not jl_path.exists():
            cot_nums.append(0)
            continue
        episode_names = ds.all_episode_names
        st_cnt = 0
        for ep in tqdm(episode_names):
            ep_jl_path = jl_path / f"{ep}.cot.jsonl"
            if not ep_jl_path.exists():
                continue
            jl = load_jsonlines(ep_jl_path)
            all_frame_ids = set([item['frame_id'] for item in jl if item.get('cot')])
            st_cnt += len(all_frame_ids)
        cot_nums.append(st_cnt)
        cstr = f"{dataset_name}: {st_cnt} / {ds.total_frames} = {st_cnt / ds.total_frames:.4f}"
        # cstr = f"{st_cnt / ds.total_frames:.1%}"
        st_strs.append(cstr)
        format_strs.append(f"{st_cnt / ds.total_frames:.2%}") # only "xx.xx%"
    print("cot_tasks = ")
    print_list(st_strs)
    print(f"{sum(cot_nums)} / {sum(f_nums)} = {sum(cot_nums) / sum(f_nums):.1%}")
    print("cot_tasks = ")
    print_list(format_strs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args.add_argument("--category", type=str, default="basic", help="Category of datasets to process. Options: basic, task, subtask, bbox, cot.", choices=["basic", "task", "subtask", "bbox", "cot"])
    args = args.parse_args()
    _raw_datasets = args.dataset_paths
    if args.category == "basic":
        statistics()
    elif args.category == "task":
        statistics_task()
    elif args.category == "subtask":
        statistics_subtask()
    elif args.category == "bbox":
        statistics_bbox()
    elif args.category == "cot":
        statistics_cot()