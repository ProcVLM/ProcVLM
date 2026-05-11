"""
python ecot/postprocess/viz_sub_task.py
"""

import os
import random
import argparse
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from core.data.reader import FastLerobotVLReader
from core.utils.common import load_jsonlines, draw_text_block, images_to_video
from core.data.generals import EXO_CAMERA_MAP
load_dotenv()

viz_item_key = None

def save_episode_video_with_subtasks(
    reader: FastLerobotVLReader,
    episode_name: str, 
    camera_key: str,
    jsonl_path: Path | str, 
    save_dir: Path | str, 
    **kwargs # fps, etc.
):
    expected_st_jsonl_path = Path(jsonl_path) / reader.name / f"{episode_name}.sub_task.jsonl"
    if not expected_st_jsonl_path.exists():
        print(f"Warning: sub-task annotation file not found for episode {episode_name}, file {expected_st_jsonl_path}")
        return
    jl = load_jsonlines(expected_st_jsonl_path)
    frame_id2item = {item['frame_id']: item for item in jl}
    st, ed = reader.get_episode_range(episode_name)
    fs = []
    for d in reader[st:ed]:
        frame_id = int(d['frame_index'])
        img = d[camera_key]
        if frame_id not in frame_id2item:
            print(f"Error: frame {frame_id} not in sub-task annotations for episode {episode_name}, file {expected_st_jsonl_path}")
            continue
        if frame_id in frame_id2item:
            info = frame_id2item[frame_id]
            text = (
                f"Task: {info['task']}\n"
                f"Subtask: {info['sub_task']}\n"
            )
        else:
            text = "No sub-task info"
        # upscale too-small images for better visualization
        if img.width < 400:
            img = img.resize((400, int(img.height * 400 / img.width)), resample=Image.Resampling.LANCZOS)
        img = draw_text_block(img, text)
        fs.append(img)
    if not fs:
        print(f"Warning: no frames to save for episode {episode_name}, file {expected_st_jsonl_path}")
        return
    save_path = Path(save_dir) / reader.name / f"{episode_name}_{camera_key}.mp4"
    images_to_video(fs, save_path, **kwargs)

def save_dataset_videos_with_subtasks(
    dataset_path: str | Path,
    jsonl_path: str | Path,
    save_dir: str | Path,
    episodes: list[str] | None = None,
    sample_len: int = 20,
    sample_rate: float = 0.01,
    **kwargs # fps, show_object_name, etc.
):
    dataset = FastLerobotVLReader(
        root=dataset_path,
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
            ebjp = Path(jsonl_path) / dataset.name / f"{ep}.sub_task.jsonl"
            if ebjp.exists():
                episodes.append(ep)
    for ep_stem in episodes:
        # for camera_key in dataset.loaded_camera_keys:
        #     print(f"Processing {dataset.name} - {ep_stem} - {camera_key}")
        #     save_episode_video_with_subtasks(
        #         reader=dataset,
        #         episode_name=ep_stem,
        #         camera_key=camera_key,
        #         jsonl_path=jsonl_path,
        #         save_dir=Path(save_dir),
        #         fps=max(20, dataset.meta.fps * 2),
        #         **kwargs
        #     )
        save_episode_video_with_subtasks(
            reader=dataset,
            episode_name=ep_stem,
            camera_key=dataset.exo_camera_key,
            jsonl_path=jsonl_path,
            save_dir=Path(save_dir),
            fps=max(20, dataset.meta.fps * 2),
            **kwargs
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = args.parse_args()
    _raw_datasets = args.dataset_paths

    for dataset_path in _raw_datasets:
        save_dataset_videos_with_subtasks(
            dataset_path=dataset_path,
            jsonl_path=os.getenv("ANNOTATION_SUB_TASK_DIR"),
            save_dir=f"{os.getenv('ANNOTATION_VIZ_SAVE_DIR')}/subtask_videos",
            # episodes=['episode_000000', 'episode_000010'],
            # episodes=['episode_000697', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_000115', 'episode_000158', 'episode_000159'],
            # episodes=['episode_000613', 'episode_000202', 'episode_000882', 'episode_001326', 'episode_001040'],
            # episodes=['episode_000698', 'episode_000442', 'episode_000115', 'episode_000461', 'episode_001362', 'episode_002077', 'episode_001145', 'episode_005144', 'episode_003267', 'episode_004096', 'episode_006678', 'episode_004399'],
            sample_len=10,
            sample_rate=0.001,
        )
