"""
python grounding/postprocess/flatten_bbox.py
"""

import os
import logging
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from core.utils.common import load_jsonlines, write_jsonlines
from core.data.reader import FastLerobotVLReader
from core.data.generals import (
    EGO_CAMERA_MAP, 
    EXO_CAMERA_MAP
)
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"  # make it more detailed
)

_raw_datasets = []
bbox_dir = os.getenv('OUTPUT_BBOX_JSONL_DIR')

_cached_raw_datasets = {}
def _fetch_raw_dataset(dataset_path):
    dataset_name = Path(dataset_path).name
    d_bbox_dir = Path(bbox_dir) / dataset_name
    # d_bbox_dir = Path(bbox_dir) / "mock"
    if not d_bbox_dir.exists():
        logging.warning(f"[{dataset_name}] bbox dir {d_bbox_dir} not found, skipped")
        return None, None
    if dataset_name in _cached_raw_datasets:
        return _cached_raw_datasets[dataset_name], d_bbox_dir
    raw_dataset = FastLerobotVLReader(
        root=dataset_path,
        ego_name=EGO_CAMERA_MAP.get(Path(dataset_path).name, None),
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        return_record_meta=True,
    )
    _cached_raw_datasets[dataset_name] = raw_dataset
    return raw_dataset, d_bbox_dir

def flatten_episode_bbox(pej: Path, name: str):
    if not pej.exists():
        return
    jl = load_jsonlines(pej)
    key2item = {(item['frame_id'], item['camera_key']): item for item in jl if 'ref' not in item.keys()}
    key2ref = {(item['frame_id'], item['camera_key']): item['ref'] for item in jl if 'ref' in item.keys()}
    new_jl = []
    refresh = False
    for line in jl:
        if 'ref' in line.keys():
            if (line['frame_id'], line['camera_key']) in key2item:
                # already flattened
                continue
            try:
                ref_key = (line['ref'], line['camera_key'])
                # retreive recursively
                while ref_key not in key2item:
                    ref_key = (key2ref[ref_key], ref_key[1])
                ref_item = key2item[ref_key]
                new_item = {
                    'frame_id': line['frame_id'],
                    'camera_key': line['camera_key'],
                }
                for k, v in ref_item.items():
                    if k not in ['frame_id', 'camera_key']:
                        new_item[k] = v
                new_jl.append(new_item)
                key2item[(line['frame_id'], line['camera_key'])] = new_item
                refresh = True
            except KeyError:
                # breakpoint()
                refresh = 'destroy'
                # break
                new_jl.append({
                    'frame_id': line['frame_id'],
                    'camera_key': line['camera_key'],
                    'bboxes': [],
                    'task_desc': '',
                    'answer_turn1': '',
                    'generated_text': '',
                })
        else:
            new_jl.append(line)
    if refresh == True:
        # assert len(new_jl) == len(jl)
        write_jsonlines(new_jl, pej)
        # logging.info(f"[{name}] flattened {pej}, {len(jl)} -> {len(new_jl)}")
    elif refresh == 'destroy':
        write_jsonlines(new_jl, pej)
        logging.warning(f"[{name}] failed to flatten {pej}, remained but some refs broken")
        # logging.warning(f"[{name}] failed to flatten {pej}, destroyed")
        # os.remove(pej)

def flatten_bbox(dataset_path):
    dataset, d_bbox_dir = _fetch_raw_dataset(dataset_path)
    if dataset is None:
        return
    for ep_stem in tqdm(dataset.all_episode_names, desc=f"[{dataset.name}] Flattening BBOX"):
        pej = d_bbox_dir / f"{ep_stem}.bbox.jsonl"
        flatten_episode_bbox(pej, dataset.name)

# def flatten_bbox(dataset_path):
#     dataset, d_bbox_dir = _fetch_raw_dataset(dataset_path)
#     if dataset is None:
#         return
#     for ep_stem in tqdm(dataset.all_episode_names, desc=f"[{dataset.name}]"):
#         pej = d_bbox_dir / f"{ep_stem}.bbox.jsonl"
#         if not pej.exists():
#             continue
#         jl = load_jsonlines(pej)
#         hash2item = {item['img_hash']: item for item in jl if 'ref' not in item.keys()}
#         new_jl = []
#         refresh = False
#         for line in jl:
#             if 'ref' in line.keys():
#                 try:
#                     ref_hash = line['ref']
#                     ref_item = hash2item[ref_hash]
#                     new_item = {
#                         "img_hash": line['img_hash'],
#                         "bboxes": ref_item['bboxes'],
#                         "task_desc": ref_item['task_desc'],
#                         "answer_turn1": ref_item['answer_turn1'],
#                         "generated_text": ref_item['generated_text'],
#                     }
#                     new_jl.append(new_item)
#                     hash2item[line['img_hash']] = new_item
#                     refresh = True
#                 except KeyError:
#                     # breakpoint()
#                     # raise NotImplementedError(f"ref hash {ref_hash} not found in {pej}")
#                     refresh = 'destory'
#                     break
#             else:
#                 new_jl.append(line)
#         if refresh == True:
#             assert len(new_jl) == len(jl)
#             write_jsonlines(new_jl, pej)
#             logging.info(f"[{dataset.name}] flattened {pej}, {len(jl)} -> {len(new_jl)}")
#         elif refresh == 'destory':
#             logging.warning(f"[{dataset.name}] failed to flatten {pej}, destoryed")
#             os.remove(pej)
            # breakpoint()
            # raise NotImplementedError(f"ref hash not found in {pej}")

# def restore_resize(dataset_path):
#     raw_dataset, d_bbox_dir = _fetch_raw_dataset(dataset_path)
#     if raw_dataset is None:
#         return
#     resized_dataset = FastLerobotVLReader(
#         root=dataset_path,
#         ego_name=EGO_CAMERA_MAP.get(Path(dataset_path).name, None),
#         exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
#         image_resize=(480, 480),
#         return_record_meta=True,
#     )
#     assert len(raw_dataset) == len(resized_dataset)
#     total_frames = len(raw_dataset)

#     for ep_stem in tqdm(raw_dataset.all_episode_names, desc=f"[{raw_dataset.name}]"):
#         pej = d_bbox_dir / f"{ep_stem}.bbox.jsonl"
#         if not pej.exists():
#             continue
#         jl = load_jsonlines(pej)
#         slices = raw_dataset.get_episode_range(ep_stem)
#         refresh = False
#         for raw_data, resized_data in zip(raw_dataset[slices], resized_dataset[slices]):
#             resized_third = resized_data['third_img']
#             resized_ego = resized_data.get('ego_img', None)
#             resized_third_hsh, resized_ego_hsh = hash_image(resized_third), hash_image(resized_ego) if resized_ego is not None else None
#             raw_third_hsh, raw_ego_hsh = hash_image(raw_data['third_img']), hash_image(raw_data['ego_img']) if resized_ego is not None else None
#             for line in jl:
#                 if line['img_hash'] == resized_third_hsh:
#                     refresh = True
#                     line['img_hash'] = raw_third_hsh
#                 if resized_ego_hsh is not None and line['img_hash'] == resized_ego_hsh:
#                     refresh = True
#                     line['img_hash'] = raw_ego_hsh
#         if refresh:
#             write_jsonlines(jl, pej)
#             logging.info(f"[{raw_dataset.name}] restored {pej}")


def pad_with_empty_boxes(dataset_path: str):
    dataset, working_dir = _fetch_raw_dataset(dataset_path)
    if dataset is None:
        return
    # working_dir = Path('annotation/v2/bbox') / dataset.name

    for ep in tqdm(dataset.all_episode_names, desc=f"Pad with empty boxes {dataset.name}", total=len(dataset.all_episode_names)):
        jl_path = working_dir / f"{ep}.bbox.jsonl"
        if not jl_path.exists():
            logging.warning(f"Episode jsonl file not found, all frames will be padded with empty boxes: {jl_path}")
            jl = []
        else:
            jl = load_jsonlines(jl_path)
        ep_start, ep_end = dataset.get_episode_range(ep)
        en_len = ep_end - ep_start
        existing = set([(item['frame_id'], item['camera_key']) for item in jl])
        for fid in range(en_len):
            for cam_key in dataset.loaded_camera_keys:
                if (fid, cam_key) not in existing:
                    new_item = {
                        'frame_id': fid,
                        'camera_key': cam_key,
                        'task_desc': '',
                        'bboxes': [],
                    }
                    jl.append(new_item) 
        # sort jl by frame_id
        jl = sorted(jl, key=lambda x: (x['camera_key'], x['frame_id']))
        write_jsonlines(jl, jl_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = args.parse_args()
    _raw_datasets = args.dataset_paths

    for dataset_path in tqdm(_raw_datasets):
        flatten_bbox(dataset_path)
        # restore_resize(dataset_path)
    for dataset_path in tqdm(_raw_datasets):
        pad_with_empty_boxes(dataset_path)