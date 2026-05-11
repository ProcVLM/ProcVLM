"""
CUDA_VISIBLE_DEVICES=3,4 python grounding/postprocess/clean_bbox.py
"""

import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from core.data.reader import FastLerobotVLReader
from core.data.generals import EXO_CAMERA_MAP, EGO_CAMERA_MAP
from core.utils.common import load_jsonlines, write_jsonlines, episode2num_token
load_dotenv()

"""
Bounding Box Cleaning Pipeline - Overview

Requirements:
- The boxes should be strictly correlated with the task, based on its label. 
  i.e. background objects and other irrelevant objects should be removed.
  This kind of cleaning can be finished by LLM-based filtering.

- The boxed parts in the image should be valid objects, and well aligned with the output labels.
  Invalid boxes can be detected by Vision-Language encoders, by computing the alignment score between the cropped image and the label.

- The boxes from a same camera, on a same task, should be temporally consistent as much as possible.
  i.e. if an object is boxed in one frame, it should also be boxed in the nearby frames. And jumping boxes should be removed.
  This kind of cleaning can be finished by velocity-based filtering.

The proper order of the above three steps should be:
Stage 1. VL Alignment Filtering 
Stage 2. Task Relevance Filtering
Stage 3. Temporal Consistency Filtering


stage0 item:
    {
        "frame_id": r_idx,
        "camera_key": cam_key,
        "bboxes": json_response,
        "task_desc": task,
        "answer_turn1": answer_turn1,
        "generated_text": generated_text,
    }
stage1 item:
    {
        "frame_id": r_idx,
        "camera_key": cam_key,
        "task_desc": task,
        "bboxes": [obj1, obj2, ...],
        "invalid_bboxes": [obj1, obj2, ...],
    }
stage2 item:
    {
        "frame_id": r_idx,
        "camera_key": cam_key,
        "task_desc": task,
        "bboxes": [obj1, obj2, ...],
        "deprecated_bboxes": [obj1, obj2, ...],
        "invalid_bboxes": [obj1, obj2, ...],
    }
stage3 item:
    {
        "frame_id": r_idx,
        "camera_key": cam_key,
        "task_desc": task,
        "bboxes": [obj1, obj2, ...],
        "outlier_bboxes": [obj1, obj2, ...],
        "deprecated_bboxes": [obj1, obj2, ...],
        "invalid_bboxes": [obj1, obj2, ...],
    }
"""

_cache_d = {}
def fetch_dataset(dp):
    if dp not in _cache_d:
        _cache_d[dp] = FastLerobotVLReader(
            root=dp,
            # ego_name=EGO_CAMERA_MAP.get(Path(dp).name, None),
            # exo_name=EXO_CAMERA_MAP.get(Path(dp).name, None),
            load_all_camera_keys=True,
            prefetch_num=0,
            cache_num=1
        )
    return _cache_d[dp]

def stage1_clean(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=1, rank=0, test_eps=None):
    from grounding.postprocess.functions.siglip import process_a_frame as vl_filter
    threshold = 0.005

    in_base = Path(bbox_jsonl_path)
    out_base = Path(clean_bbox_jsonl_path) / "stage1"
    out_base.mkdir(parents=True, exist_ok=True)
    for i, dp in enumerate(raw_datasets):
        dn = Path(dp).name
        if not (in_base / dn).exists():
            print(f"Stage 2 input path not found: {in_base / dn}, skip.")
            continue
        d = fetch_dataset(dp)
        eps = d.all_episode_names
        if test_eps is not None:
            eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
        for ep_stem in tqdm(eps, desc=f"Stage 1 {i+1}/{len(raw_datasets)}: {dn}"):
            pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
            pec = out_base / dn / f"{ep_stem}.bbox.jsonl"
            if episode2num_token(ep_stem) % world_size != rank:
                continue
            if pec.exists():
                continue
            if not pej.exists():
                print(f"Raw bbox jsonl not found: {pej}")
                breakpoint()
            pec.parent.mkdir(parents=True, exist_ok=True)
            jl = load_jsonlines(pej)
            key2item = {}
            for item in jl:
                key = (item['frame_id'], item['camera_key'])
                if key not in key2item:
                    key2item[key] = item
                else:
                    if len(item['bboxes']) > len(key2item[key]['bboxes']):
                        key2item[key] = item
                    print(f"Warning: duplicated bbox for frame '{item['frame_id']}' camera '{item['camera_key']}' in '{ep_stem}' of '{dn}'")
            new_jl = []
            st, ed = d.get_episode_range(ep_stem)
            for cam_key in d.loaded_camera_keys:
                for data in d[st:ed]:
                    frame_id = int(data['frame_index'])
                    item = key2item.get((frame_id, cam_key), None)
                    if item is None:
                        # new_jl.append({
                        #     "frame_id": frame_id,
                        #     "camera_key": cam_key,
                        #     "bboxes": [],
                        #     "invalid_bboxes": [],
                        #     "task_desc": data['task_description'],
                        # })
                        print(f"Warning: missing bbox for frame '{frame_id}' camera '{cam_key}' in '{ep_stem}' of '{dn}'")
                    else:
                        def area(box):
                            x1, y1, x2, y2 = box['bbox_2d']
                            return max(0, x2 - x1) * max(0, y2 - y1)
                        valid_boxes = [box for box in item['bboxes'] if area(box) < 0.4]
                        invalid_boxes = [box for box in item['bboxes'] if area(box) >= 0.4]
                        pil_img = data[cam_key]
                        valid_boxes = vl_filter(pil_img, valid_boxes)
                        invalid_boxes.extend([b for b in valid_boxes if b['score'] < threshold])
                        valid_boxes = [b for b in valid_boxes if b['score'] >= threshold]
                        new_jl.append({
                            "frame_id": item["frame_id"],
                            "camera_key": item["camera_key"],
                            "task_desc": item["task_desc"],
                            "bboxes": valid_boxes,
                            "invalid_bboxes": invalid_boxes,
                        })
            # assert len(new_jl) == len(jl), f"new jsonl has {len(new_jl)} records vs original {len(jl)}"
            write_jsonlines(new_jl, pec)

def run_stage1_clean_ray(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size):
    import ray
    ray.init()
    @ray.remote(num_gpus=0.25)
    def stage1_clean_ray_wrapper(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size, rank):
        stage1_clean(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size, rank)
    futures = [
        stage1_clean_ray_wrapper.remote(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size, rank)
        for rank in range(world_size)
    ]
    results = ray.get(futures)
    ray.shutdown()

def stage2_clean(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=1, rank=0, test_eps=None):
    from grounding.postprocess.functions.llm import llm_filter_batch
    in_base = Path(clean_bbox_jsonl_path) / "stage1"
    out_base = Path(clean_bbox_jsonl_path) / "stage2"
    out_base.mkdir(parents=True, exist_ok=True)
    for i, dp in enumerate(raw_datasets):
        dn = Path(dp).name
        if not (in_base / dn).exists():
            print(f"Stage 2 input path not found: {in_base / dn}, skip.")
            continue
        d = fetch_dataset(dp)
        eps = d.all_episode_names
        if test_eps is not None:
            eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
        for ep_stem in tqdm(eps, desc=f"Stage2 {i+1}/{len(raw_datasets)}: {dn}"):
            pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
            pec = out_base / dn / f"{ep_stem}.bbox.jsonl"
            if episode2num_token(ep_stem) % world_size != rank:
                continue
            if pec.exists():
                continue
            if not pej.exists():
                print(f"Stage 1 cleaned file not found: {pej}")
                breakpoint()
            pec.parent.mkdir(parents=True, exist_ok=True)
            jl = load_jsonlines(pej)
            batch_input = []
            for item in jl:
                bboxes = item["bboxes"]
                batch_input.append((bboxes, item["task_desc"]))
            batch_output = llm_filter_batch(batch_input)
            new_jl = []
            for idx, item in enumerate(jl):
                filtered, deprecated = batch_output[idx]
                new_jl.append({
                    "frame_id": item["frame_id"],
                    "camera_key": item["camera_key"],
                    "task_desc": item["task_desc"],
                    "bboxes": filtered,
                    "deprecated_bboxes": deprecated,
                    "invalid_bboxes": item["invalid_bboxes"],
                })
            assert len(new_jl) == len(jl), f"new jsonl has {len(new_jl)} records vs original {len(jl)}"
            write_jsonlines(new_jl, pec)

def stage3_clean(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=1, rank=0, test_eps=None):
    from grounding.postprocess.functions.filters import (
        Vel_vec, 
        split_episode_by_task,
        GammaDistribution as Distribution
    )
    # in_base = Path(clean_bbox_jsonl_path) / "stage2"
    in_base = Path(bbox_jsonl_path)
    out_base = Path(clean_bbox_jsonl_path) / "stage3"
    out_base.mkdir(parents=True, exist_ok=True)
    for i, dp in enumerate(raw_datasets):
        dn = Path(dp).name
        if not (in_base / dn).exists():
            print(f"Stage 2 input path not found: {in_base / dn}, skip.")
            continue
        d = fetch_dataset(dp)
        eps = d.all_episode_names
        if test_eps is not None:
            eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
        for ep_stem in tqdm(eps, desc=f"Stage3 {i+1}/{len(raw_datasets)}: {dn}"):
            pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
            pec = out_base / dn / f"{ep_stem}.bbox.jsonl"
            if episode2num_token(ep_stem) % world_size != rank:
                continue
            if pec.exists():
                continue
            if not pej.exists():
                print(f"Stage 2 cleaned file not found: {pej}")
                breakpoint()
            pec.parent.mkdir(parents=True, exist_ok=True)
            all_jl = load_jsonlines(pej)
            
            # bug fix:  seperate different cameras!
            new_jl = []
            splits = split_episode_by_task(all_jl)
            for spl in splits:
                all_boxes = []
                is_destroyed = []
                for item in spl:
                    boxes = item['bboxes']
                    if len(boxes) == 0:
                            is_destroyed.append(True)
                    else:
                        is_destroyed.append(False)
                        all_boxes.append([b['bbox_2d'] for b in boxes])

                valid_num = len(all_boxes)
                # assert valid_num > 1, f"All frames are destroyed in {ep_stem} of {dn}"
                if valid_num < 2:
                    print(f"Warning: {ep_stem} in {dn} has too few valid frames ({valid_num}), skipping velocity-based cleaning.")
                    for item in spl:
                        boxes = []
                        for b in item['bboxes']:
                            if 'score' not in b:
                                b['score'] = 1.0  # default score
                            boxes.append(b)
                        new_jl.append({
                            "frame_id": item["frame_id"],
                            "camera_key": item["camera_key"],
                            "task_desc": item["task_desc"],
                            "bboxes": boxes,
                            "outlier_bboxes": [],
                            # "deprecated_bboxes": item["deprecated_bboxes"],
                            # "invalid_bboxes": item["invalid_bboxes"],
                        })
                    continue

                all_vels = []
                for idx in range(valid_num):
                    cur_boxes = all_boxes[idx]
                    vels = Vel_vec(cur_boxes, all_boxes, idx) if len(cur_boxes) > 0 else []
                    all_vels.append(vels)

                acc_id = 0
                for i, item in enumerate(spl):
                    if is_destroyed[i]:
                        new_jl.append({
                            "frame_id": item["frame_id"],
                            "camera_key": item["camera_key"],
                            "task_desc": item["task_desc"],
                            "bboxes": [],
                            "outlier_bboxes": item["bboxes"],
                            # "deprecated_bboxes": item["deprecated_bboxes"],
                            # "invalid_bboxes": item["invalid_bboxes"],
                        })
                        continue
                    sampled_vels = all_vels
                    sampled_vels = [v for sublist in sampled_vels for v in sublist] # temporal values is included by the velocity computation
                                                                                        # so at this time we can simply flatten the list
                    ed = Distribution(sampled_vels)

                    # check outliers
                    new_boxes, outliers = [], []
                    for box_id, v in enumerate(all_vels[acc_id]):
                        box = item['bboxes'][box_id]
                        if 'score' not in box:
                            box['score'] = 1.0  # default score
                        if not ed.is_outlier(v):
                            new_boxes.append(box)
                        else:
                            outliers.append(box)
                    new_jl.append({
                        "frame_id": item["frame_id"],
                        "camera_key": item["camera_key"],
                        "task_desc": item["task_desc"],
                        "bboxes": new_boxes,
                        "outlier_bboxes": outliers,
                        # "deprecated_bboxes": item["deprecated_bboxes"],
                        # "invalid_bboxes": item["invalid_bboxes"],
                    })
                    acc_id += 1

            assert len(new_jl) == len(all_jl), f"new jsonl has {len(new_jl)} records vs original {len(all_jl)}"
            write_jsonlines(new_jl, pec)

# FEATURE: generate missing boxes by previous/next frames, based-on velocity-based interpolation
def bbox_generation_by_velocity_interpolation(raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=1, rank=0, test_eps=None):
    from grounding.postprocess.functions.filters import interpolate_boxes
    in_base = Path(clean_bbox_jsonl_path) / "stage3"
    out_base = Path(clean_bbox_jsonl_path) / "interpolated"
    out_base.mkdir(parents=True, exist_ok=True)
    for i, dp in enumerate(raw_datasets):
        dn = Path(dp).name
        if not (in_base / dn).exists():
            print(f"Stage 2 input path not found: {in_base / dn}, skip.")
            continue
        d = fetch_dataset(dp)
        eps = d.all_episode_names
        if test_eps is not None:
            eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
        itp_cnt, tot_cnt = 0, 0
        for ep_stem in tqdm(eps, desc=f"Interpolation {i+1}/{len(raw_datasets)}: {dn}"):
            pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
            pec = out_base / dn / f"{ep_stem}.bbox.jsonl"
            if episode2num_token(ep_stem) % world_size != rank:
                continue
            if pec.exists():
                continue
            if not pej.exists():
                print(f"Stage 3 cleaned file not found: {pej}")
                breakpoint()
            pec.parent.mkdir(parents=True, exist_ok=True)
            all_jl = load_jsonlines(pej)
            s, e = d.get_episode_range(ep_stem)
            img_dict = {}
            for data in d[s:e]:
                frame_id = int(data['frame_index'])
                for cam_key in d.loaded_camera_keys:
                    img_dict[(frame_id, cam_key)] = data[cam_key]
            for item in all_jl:
                item['image'] = img_dict[(item['frame_id'], item['camera_key'])]
            new_jl = interpolate_boxes(all_jl)
            assert len(new_jl) == len(all_jl), f"new jsonl has {len(new_jl)} records vs original {len(all_jl)}"
            write_jsonlines(new_jl, pec)
            for obj in new_jl:
                tot_cnt += len(obj['bboxes'])
                for b in obj['bboxes']:
                    if str(b['score']) == 'interpolated':
                        itp_cnt += 1
        if tot_cnt > 0:
            print(f"Total interpolated boxes in {dn}: {itp_cnt}, interpolation ratio: {itp_cnt/tot_cnt:.2%}")

def statistics(rank=0, test_eps=None):
    if rank != 0:
        return
    from collections import Counter
    stage_dirs = ['stage1', 'stage2', 'stage3']
    for stage in stage_dirs:
        total_frames = 0
        total_boxes = 0
        total_accumulated_cleaned_boxes = 0
        print(f"Statistics for {stage}:")
        box_counter = Counter()
        in_base = Path(clean_bbox_jsonl_path) / stage
        if not in_base.exists():
            print(f"{stage} not found, skip.")
            continue
        for dp in _raw_datasets:
            d = fetch_dataset(dp)
            dn = d.name
            eps = d.all_episode_names
            if test_eps is not None:
                eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
            _total_frames = 0
            _total_boxes = 0
            _total_accumulated_cleaned_boxes = 0
            _box_counter = Counter()
            for ep_stem in eps:
                pej = in_base / dn / f"{ep_stem}.bbox.jsonl"
                if not pej.exists():
                    # print(f"{stage} cleaned file not found: {dn}/{ep_stem}")
                    continue
                jl = load_jsonlines(pej)
                _total_frames += len(jl)
                for item in jl:
                    num_boxes = len(item['bboxes'])
                    _total_boxes += num_boxes
                    _box_counter[num_boxes] += 1
                    for k, v in item.items():
                        if k.endswith('bboxes') and k != 'bboxes':
                            _total_accumulated_cleaned_boxes += len(v)
            _avg_boxes_per_frame = _total_boxes / _total_frames if _total_frames > 0 else 0
            _rr = _total_boxes / (_total_boxes + _total_accumulated_cleaned_boxes) if (_total_boxes + _total_accumulated_cleaned_boxes) > 0 else 0
            # print(f"---------- {stage} '{dn}' Box Statistics ----------")
            # print(f"Valid box count: {_total_boxes}, filtered box count: {_total_accumulated_cleaned_boxes}, retention rate: {_rr:.2%}")
            # print(f"Average boxes per frame: {_avg_boxes_per_frame:.4f}")
            # print(f"Box count distribution: {_box_counter}")

            # if stage == stage_dirs[-1]:  # only aggregate final stage
            total_frames += _total_frames
            total_boxes += _total_boxes
            total_accumulated_cleaned_boxes += _total_accumulated_cleaned_boxes
            box_counter.update(_box_counter)
    
        # if stage == stage_dirs[-1]:  # only aggregate final stage
        avg_boxes_per_frame = total_boxes / total_frames if total_frames > 0 else 0
        rr = total_boxes / (total_boxes + total_accumulated_cleaned_boxes) if (total_boxes + total_accumulated_cleaned_boxes) > 0 else 0
        print(f"========== Overall {stage} Box Statistics ==========")
        print(f"Total frames: {total_frames}, valid box count: {total_boxes}, filtered box count: {total_accumulated_cleaned_boxes}, retention rate: {rr:.2%}")
        print(f"Average boxes per frame: {avg_boxes_per_frame:.4f}")
        print(f"Box count distribution: {box_counter}")

    itcnt, allcnt = 0, 0
    total_frames = 0
    box_counter = Counter()
    box_coverage = {}
    for dp in _raw_datasets:
        d = fetch_dataset(dp)
        dn = d.name
        eps = d.all_episode_names
        if test_eps is not None:
            eps = [ep for ep in eps if ep in test_eps and ep in d.all_episode_names]
        in_base = Path(clean_bbox_jsonl_path) / "interpolated"
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
    print("========================================")
    print(f"All {len(_raw_datasets)} datasets have {allcnt} boxes in total after all processing stages.")
    print(f"{itcnt} boxes are generated by interpolation, interpolation ratio: {itcnt/allcnt:.2%}")
    print(f"Average boxes per frame after interpolation: {allcnt/total_frames:.4f}")
    print(f"Overall box coverage after interpolation: {sum([v[0] for v in box_coverage.values()])}/{sum([v[1] for v in box_coverage.values()])} = {sum([v[0] for v in box_coverage.values()])/sum([v[1] for v in box_coverage.values()]):.2%}")
    print(f"Box count distribution after interpolation: {box_counter}")
    print("========================================")

if __name__ == "__main__":
    bbox_jsonl_path=os.getenv('OUTPUT_BBOX_JSONL_DIR')
    clean_bbox_jsonl_path=os.getenv('OUTPUT_CLEAN_BBOX_JSONL_DIR')
    # run_stage1_clean_ray(_raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path, world_size=8)
    # stage2_clean(_raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path)
    # stage3_clean(_raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path)
    # bbox_generation_by_velocity_interpolation(_raw_datasets, bbox_jsonl_path, clean_bbox_jsonl_path)
    statistics()