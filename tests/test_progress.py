"""
python tests/test_progress.py --dataset_paths $EVQA_DATASETS_EXPAND --sample_rate 0.1
"""

import logging
import argparse
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from PIL.Image import Image
from evqa.tools.data_picker import AnnotatedLerobotVLReader
from core.utils.common import images_to_video, draw_text_block


def episode_data_generator(datasets: List[str], sample_rate: Optional[int] = None, max_episodes: Optional[int] = None):
    """ Data generator for episodes from multiple datasets. """
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).name
        try:
            reader = AnnotatedLerobotVLReader(root=dataset_path, versions=['v2'])
            reader.set_run_mode(['SUB_TASK', 'PLANNED_ACTIONS'])
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
        if max_episodes is not None:
            iterator = reader.vl_reader.all_episode_names[:max_episodes]
        else:
            iterator = reader.vl_reader.all_episode_names
        for episode_name in iterator:
            if sample_rate and random.random() > sample_rate:
                continue
            try:
                s, t = reader.vl_reader.get_episode_range(episode_name)
                episode_frames = reader[s:t]
                episode_stats = {
                    'len': len(episode_frames),
                    'fps': reader.vl_reader.meta.fps,
                }
                
                start_fid = 0
                task_splits = []
                for fid in range(1, episode_stats['len']):
                    if episode_frames[fid]['task'] != episode_frames[fid - 1]['task']:
                        task_splits.append((start_fid, fid, episode_frames[fid - 1]['task']))
                        start_fid = fid
                task_splits.append((start_fid, episode_stats['len'], episode_frames[-1]['task']))
                if len(task_splits) > 1:
                    continue
                episode_stats['task_splits'] = task_splits
                
                episode_stats['sub_task_splits'] = {}
                episode_stats['plan'] = {}
                for start_fid, end_fid, task in task_splits:
                    episode_splits = []
                    episode_plan = []
                    current_sub_task_start_fid = start_fid
                    for fid in range(start_fid + 1, end_fid):
                        if episode_frames[fid]['sub_task'] != episode_frames[fid - 1]['sub_task']:
                            episode_splits.append((current_sub_task_start_fid, fid, episode_frames[fid - 1]['sub_task']))
                            episode_plan.append(episode_frames[fid - 1]['sub_task'])
                            current_sub_task_start_fid = fid
                    episode_splits.append((current_sub_task_start_fid, end_fid, episode_frames[end_fid - 1]['sub_task']))
                    episode_plan.append(episode_frames[end_fid - 1]['sub_task'])
                    episode_stats['sub_task_splits'][task] = episode_splits
                    episode_stats['plan'][task] = episode_plan

                yield {
                    "dataset_name": dataset_name,
                    "episode_name": episode_name,
                    "episode_frames": episode_frames,
                    "episode_stats": episode_stats
                }
            except Exception as e:
                logging.error(f"Error processing episode {episode_name}: {e}")
                continue


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args.add_argument("--sample_rate", type=float, default=None, help="Sampling rate for episodes.")
    args.add_argument("--output_dir", type=str, default='tmp', help="Directory to save output videos and stats.")
    args.add_argument("--max_episodes", type=int, default=50, help="Maximum number of episodes to process per dataset.")
    args = args.parse_args()

    iterator = episode_data_generator(datasets=args.dataset_paths, sample_rate=args.sample_rate, max_episodes=args.max_episodes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from evqa.tools.functions.progress import p as p_function
    average_time_per_episode = 0.0
    episode_count = 0
    for episode_data in iterator:
        start_time = time.time()
        frames = episode_data['episode_frames']
        dn = episode_data['dataset_name']
        en = episode_data['episode_name']
        output_path = output_dir / "progress_videos" / dn / f"{en}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        progress_values = p_function(frames)
        images = []
        for idx, frame in enumerate(frames):
            img = draw_text_block(frame['exo_image'], f"Task: {frame['task']}", pos="top-left")
            img = draw_text_block(img, f"Progress: {progress_values[idx]*100:.2f}%\nSub-task: {frame['sub_task']}", pos="bottom-left") # use percentage
            images.append(img)
        images_to_video(images, output_path, fps=episode_data['episode_stats']['fps'])

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_time_per_episode = (average_time_per_episode * episode_count + elapsed_time) / (episode_count + 1)
        episode_count += 1
        print(f"Processed episode {episode_count} in {elapsed_time:.2f} seconds. Average time per episode: {average_time_per_episode:.2f} seconds.")