"""
python ecot/postprocess/viz_annotation.py
"""

import os
import random
import time
import argparse
from dotenv import load_dotenv
from pathlib import Path
from PIL import ImageFont
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import load_jsonlines, annotate_frames, images_to_video
from grounding.postprocess.flatten_bbox import flatten_episode_bbox
load_dotenv()

def handle_episode(
    reader: FastLerobotVLReader,
    episode_name: str, 
    camera_key: str,
    bbox_jsonl_path: Path | str, 
    sub_task_jsonl_path: Path | str,
    cot_jsonl_path: Path | str,
    save_dir: Path | str, 
    **kwargs # fps, show_object_name, etc.
):
    box_jl = load_jsonlines(bbox_jsonl_path)
    if any('ref' in item.keys() for item in box_jl):
        flatten_episode_bbox(bbox_jsonl_path, reader.name)
        box_jl = load_jsonlines(bbox_jsonl_path)
    key2bboxes = {(item['frame_id'], item['camera_key']): item['bboxes'] for item in box_jl}

    sub_task_jl = load_jsonlines(sub_task_jsonl_path)
    key2sub_task = {item['frame_id']: item['sub_task'] for item in sub_task_jl}
    key2task = {item['frame_id']: item['task'] for item in sub_task_jl}

    cot_jl = load_jsonlines(cot_jsonl_path)
    key2cot = {item['frame_id']: item['cot'] for item in cot_jl}

    frames = []
    bbox_list = []
    tasks = []
    sub_tasks = []
    cots = []
    ts, te = reader.get_episode_range(episode_name)
    for t in range(ts, te):
        data = reader[t]
        frame_id = int(data['frame_index'])
        frames.append(data[camera_key])

        bbox_list.append(key2bboxes.get((frame_id, camera_key), []))

        task = key2task.get(frame_id, "")
        tasks.append(task if task else "[no specified task]")
        
        sub_task = key2sub_task.get(frame_id, "")
        sub_tasks.append(sub_task if sub_task else "[no specified sub-task]")

        cots.append(key2cot.get(frame_id, "[no CoT available]"))

    font = None
    font_path = kwargs.get('font_path', None)
    if font_path is not None:
        font = ImageFont.truetype(font_path, 20)
    start_time = time.time()
    # frames = annotate_frames(frames, bbox_list, tasks, sub_tasks, show_object_name=kwargs.get('show_object_name', True), font=font, min_width=768)
    frames = annotate_frames(frames, bbox_list, tasks, sub_tasks, cots, show_object_name=kwargs.get('show_object_name', True), font=font, min_width=768)
    print(f"Annotated {len(frames)} frames for episode {episode_name} in {time.time() - start_time:.2f}s.")

    camera_name = camera_key.split('.')[-1]
    save_path = Path(save_dir) / reader.name / f"{episode_name}_{camera_name}.mp4"
    images_to_video(
        images=frames,
        output_path=save_path,
        fps=kwargs.get('fps', 20),
    )
    
def handle_dataset(
    dataset_path: str | Path,
    bbox_jsonl_path: str | Path,
    cot_jsonl_path: str | Path,
    sub_task_jsonl_path: str | Path,
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
            episodes.append(ep)
    filtered_episodes = []
    for ep in episodes:
        ebjp = Path(bbox_jsonl_path) / dataset.name / f"{ep}.bbox.jsonl"
        esjp = Path(sub_task_jsonl_path) / dataset.name / f"{ep}.sub_task.jsonl"
        ecjp = Path(cot_jsonl_path) / dataset.name / f"{ep}.cot.jsonl"
        if ebjp.exists() and esjp.exists() and ecjp.exists():
                filtered_episodes.append(ep)
    print(f"Dataset {dataset.name}: {len(filtered_episodes)}/{len(episodes)} episodes found with all annotations.")
    for ep in filtered_episodes:
        for camera_key in dataset.loaded_camera_keys:
            print(f"Processing {dataset.name} - {ep} - {camera_key}")
            handle_episode(
                reader=dataset,
                episode_name=ep,
                camera_key=camera_key,
                bbox_jsonl_path=Path(bbox_jsonl_path) / dataset.name / f"{ep}.bbox.jsonl",
                sub_task_jsonl_path=Path(sub_task_jsonl_path) / dataset.name / f"{ep}.sub_task.jsonl",
                cot_jsonl_path=Path(cot_jsonl_path) / dataset.name / f"{ep}.cot.jsonl",
                save_dir=Path(save_dir),
                fps=dataset.meta.fps,
                **kwargs
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    parsed_args = args.parse_args()
    _raw_datasets = parsed_args.dataset_paths

    eps = None
    # eps = [
    #     'episode_000010', 'episode_000021', 'episode_000036', 'episode_000043', 'episode_000054',
    #     'episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 
    #     'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 
    #     'episode_000020'
    # ]
    
    for dataset_path in _raw_datasets:
        handle_dataset(
            dataset_path=dataset_path,
            bbox_jsonl_path=os.getenv("ANNOTATION_BBOX_DIR"),
            cot_jsonl_path=os.getenv("ANNOTATION_COT_DIR"),
            sub_task_jsonl_path=os.getenv("ANNOTATION_SUB_TASK_DIR"),
            save_dir=os.getenv("ANNOTATION_VIZ_SAVE_DIR"),
            episodes=eps,
            sample_len=10,
            sample_rate=0.005,
            show_object_name=True,
            font_path='assets/fonts/Ubuntu-Regular.ttf',
        )