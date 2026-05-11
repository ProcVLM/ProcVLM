"""
python grounding/postprocess/viz_video_bbox.py
"""

import os
import random
import argparse
from dotenv import load_dotenv
from pathlib import Path
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import save_video_with_bboxes, load_jsonlines, json_response_to_bboxes
from grounding.postprocess.flatten_bbox import flatten_episode_bbox
load_dotenv()

viz_item_key = None

def save_episode_video_with_bboxes(
    reader: FastLerobotVLReader,
    episode_name: str, 
    camera_key: str,
    bbox_jsonl_path: Path | str, 
    save_dir: Path | str, 
    **kwargs # fps, show_object_name, etc.
):
    expected_bbox_jsonl_path = Path(bbox_jsonl_path) / reader.name / f"{episode_name}.bbox.jsonl"
    if not expected_bbox_jsonl_path.exists():
        return
        # raise FileNotFoundError(f"Expected bbox jsonl file not found: {expected_bbox_jsonl_path}")
    jl = load_jsonlines(expected_bbox_jsonl_path)
    if any('ref' in item.keys() for item in jl):
        flatten_episode_bbox(expected_bbox_jsonl_path, reader.name)
        jl = load_jsonlines(expected_bbox_jsonl_path)
    key2bboxes = {(item['frame_id'], item['camera_key']): item[viz_item_key] for item in jl}
    key2task = {(item['frame_id'], item['camera_key']): item['task_desc'] for item in jl}
    frames = []
    tasks = []
    bboxes_list = []
    ts, te = reader.get_episode_range(episode_name)
    for t in range(ts, te):
        data = reader[t]
        frame_id = int(data['frame_index'])
        frames.append(data[camera_key])
        if (frame_id, camera_key) not in key2bboxes:
            bboxes_list.append([])
            tasks.append(
                key2task.get((frame_id, camera_key), "")
            )
            continue
        json_response = key2bboxes[(frame_id, camera_key)]
        bboxes_list.append(json_response_to_bboxes(json_response, disable_shuffle=True))
        tasks.append(
            key2task.get((frame_id, camera_key), "")
        )
    save_path = Path(save_dir) / reader.name / f"{episode_name}_{camera_key}.mp4"
    save_video_with_bboxes(frames, bboxes_list, save_path, tasks, **kwargs)
    
def save_dataset_videos_with_bboxes(
    dataset_path: str | Path,
    bbox_jsonl_path: str | Path,
    save_dir: str | Path,
    episodes: list[str] | None = None,
    sample_len: int = 20,
    sample_rate: float = 0.01,
    **kwargs # fps, show_object_name, etc.
):
    dn = Path(dataset_path).name
    if not (Path(bbox_jsonl_path) / dn).exists():
        print(f"BBox jsonl path not found for dataset {dn}, skip.")
        return
    dataset = FastLerobotVLReader(
        root=dataset_path,
        ego_name=EGO_CAMERA_MAP.get(Path(dataset_path).name, None),
        exo_name=EXO_CAMERA_MAP.get(Path(dataset_path).name, None),
        prefetch_num=0,
    )
    if episodes is None:
        all_episodes = dataset.all_episode_names
        random.shuffle(all_episodes)
        sample_len = max(sample_len, len(all_episodes) * sample_rate)
        episodes = []
        for ep in all_episodes:
            if len(episodes) >= sample_len:
                break
            ebjp = Path(bbox_jsonl_path) / dataset.name / f"{ep}.bbox.jsonl"
            if ebjp.exists():
                episodes.append(ep)
    for ep_stem in episodes:
        for camera_key in dataset.loaded_camera_keys:
            print(f"Processing {dataset.name} - {ep_stem} - {camera_key}")
            save_episode_video_with_bboxes(
                reader=dataset,
                episode_name=ep_stem,
                camera_key=camera_key,
                bbox_jsonl_path=bbox_jsonl_path,
                save_dir=Path(save_dir),
                fps=max(20, dataset.meta.fps * 2),
                **kwargs
            )


viz_item_key = 'bboxes'
# viz_item_key = 'invalid_bboxes'
# viz_item_key = 'outlier_bboxes'
# bbox_videos/cs2/RH20T_cfg3_lerobot/
# episode_000697, episode_000442
# # real_tabletop_tasks_long_cot_lerobot-20250729/episode_000884

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=str, default='v2', help="Version of bbox annotations.")
    argparser.add_argument("--output_name", type=str, default='v2', help="Output directory name for saving videos.")
    argparser.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = argparser.parse_args()

    _raw_datasets = args.dataset_paths
    # eps = [
    #     'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
    #     'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
    #     'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
    #     'episode_006678', 'episode_004399'
    # ]

    for dataset_path in _raw_datasets:
        save_dataset_videos_with_bboxes(
            dataset_path=dataset_path,
            bbox_jsonl_path=os.getenv('ANNOTATION_ROOT') + f'/{args.version}/bbox/',
            save_dir=os.getenv('OUTPUT_BBOX_VIDEO_DIR') + args.output_name,
            # episodes=['episode_000697', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_000115', 'episode_000158', 'episode_000159'],
            # episodes=['episode_000613', 'episode_000202', 'episode_000882', 'episode_001326', 'episode_001040'],
            # episodes=['episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 'episode_006678', 'episode_004399'],
            # episodes=eps,
            sample_len=10,
            sample_rate=0.005,
            show_object_name=True,
        )

    # 1132
    # dataset = FastLerobotVLReader(
    #         root=dr,
    #         ego_name=EGO_CAMERA_MAP.get(Path(dr).name, None),
    #         exo_name=EXO_CAMERA_MAP.get(Path(dr).name, None),
    #     )
    # for k in dataset.loaded_camera_keys:
    #     save_episode_video_with_bboxes(
    #         reader=dataset,
    #         episode_name="episode_000168",
    #         camera_key=k,
    #         bbox_jsonl_path='bbox_jsonl/',
    #         save_dir='bbox_videos/',
    #         fps=20,
    #         show_object_name=True,
    #     )