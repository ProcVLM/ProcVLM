import logging
import argparse
import os
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

cot_dir = os.getenv('OUTPUT_COT_JSONL_DIR')

_raw_datasets = []
_cached_raw_datasets = {}
def _fetch_raw_dataset(dataset_path):
    dataset_name = Path(dataset_path).name
    d_cot_dir = Path(cot_dir) / dataset_name
    # d_cot_dir = Path(cot_dir) / "mock"
    if not d_cot_dir.exists():
        logging.warning(f"[{dataset_name}] cot dir {d_cot_dir} not found, skipped")
        return None, None
    if dataset_name in _cached_raw_datasets:
        return _cached_raw_datasets[dataset_name], d_cot_dir
    raw_dataset = FastLerobotVLReader(
        root=dataset_path,
        ego_name=EGO_CAMERA_MAP.get(Path(dataset_path).name, None),
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        return_record_meta=True,
    )
    _cached_raw_datasets[dataset_name] = raw_dataset
    return raw_dataset, d_cot_dir

def flatten_episode_cot(pej: Path, name: str):
    if not pej.exists():
        return
    jl = load_jsonlines(pej)
    key2item = {item['frame_id']: item for item in jl if 'ref' not in item.keys()}
    key2ref = {item['frame_id']: item['ref'] for item in jl if 'ref' in item.keys()}
    new_jl = []
    refresh = False
    for line in jl:
        if 'ref' in line.keys():
            if line['frame_id'] in key2item:
                # already flattened
                continue
            try:
                ref_key = line['ref']
                # retreive recursively
                while ref_key not in key2item:
                    ref_key = key2ref[ref_key]
                ref_item = key2item[ref_key]
                new_item = {    # new item with frame_id copied, other fields from ref_item
                    'frame_id': line['frame_id'],
                }
                for k, v in ref_item.items(): # copy other fields from ref_item
                    if k not in ['frame_id']:
                        new_item[k] = v
                new_jl.append(new_item)
                key2item[line['frame_id']] = new_item
                refresh = True
            except KeyError:
                # breakpoint()
                refresh = 'destroy'
                # break
                new_jl.append({
                    'frame_id': line['frame_id'],
                    "task": None,
                    "sub_task": None,
                    "cot": None,
                    "generated_text": None,
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

def flatten_cot(dataset_path):
    dataset, d_cot_dir = _fetch_raw_dataset(dataset_path)
    if dataset is None:
        return
    for ep_stem in tqdm(dataset.all_episode_names, desc=f"[{dataset.name}] Flattening CoT"):
        pej = d_cot_dir / f"{ep_stem}.cot.jsonl"
        flatten_episode_cot(pej, dataset.name)

def pad_with_empty_cot(dataset_path: str):
    dataset, working_dir = _fetch_raw_dataset(dataset_path)
    if dataset is None:
        return
    for ep in tqdm(dataset.all_episode_names, desc=f"Pad with Empty CoT {dataset.name}", total=len(dataset.all_episode_names)):
        jl_path = working_dir / f"{ep}.cot.jsonl"
        if not jl_path.exists():
            logging.warning(f"Episode jsonl file not found, all frames will be padded with empty CoT: {jl_path}")
            jl = []
        else:
            jl = load_jsonlines(jl_path)
        ep_start, ep_end = dataset.get_episode_range(ep)
        en_len = ep_end - ep_start
        existing = set([item['frame_id'] for item in jl])
        for fid in range(en_len):
            for cam_key in dataset.loaded_camera_keys:
                if fid not in existing:
                    new_item = {
                        'frame_id': fid,
                        'task': None,
                        'sub_task': None,
                        'cot': None,
                        'generated_text': None,
                    }
                    jl.append(new_item) 
        # sort jl by frame_id
        jl = sorted(jl, key=lambda x: x['frame_id'])
        write_jsonlines(jl, jl_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = args.parse_args()
    _raw_datasets = args.dataset_paths
    
    for dataset_path in tqdm(_raw_datasets):
        flatten_cot(dataset_path)
    for dataset_path in tqdm(_raw_datasets):
        pad_with_empty_cot(dataset_path)