import argparse
import random
import shutil
from pathlib import Path
from core.utils.common import load_jsonlines, append_jsonlines

src_root: str
dst_root: str

def sample_dataset(task_id: str, dataset_path: str, sample_rate: float, overwrite: bool = True):
    dataset_name = Path(dataset_path).name
    src_dir = Path(src_root) / task_id / dataset_name
    if not src_dir.exists():
        print(f"[{task_id}: {dataset_name}] Source directory does not exist. Skipping.")
        return
    dst_file = Path(dst_root) / task_id / f"{dataset_name}.jsonl"
    if dst_file.exists() and overwrite:
        response = input(f"[{task_id}: {dataset_name}] Overwriting existing sampled dataset at {dst_file}, do you confirm? (y/n): ")
        if response.lower() != "y":
            print(f"[{task_id}: {dataset_name}] Skipping.")
            return
        dst_file.unlink()
    episode_files = sorted(src_dir.glob("*.jsonl"))
    for episode_file in episode_files:
        lines = load_jsonlines(episode_file)
        sampled_lines = random.sample(lines, max(1, int(len(lines) * sample_rate)))
        append_jsonlines(sampled_lines, dst_file)
    print(f"[{task_id}: {dataset_name}] Sampled dataset copied to {dst_file}")


# --- Main Execution ---
if __name__ == "__main__":   
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_paths", type=str, nargs='+', help="List of dataset paths to process.", required=True)
    argparser.add_argument("--src_root", type=str, help="Source root directory.", required=True)
    argparser.add_argument("--dst_root", type=str, help="Destination root directory.", required=True)
    argparser.add_argument("--task_ids", type=str, nargs='+', help="List of task IDs to process.", choices=["a", "b", "c"], required=True)
    argparser.add_argument("--sample_rate", type=float, default=1.0, help="Sampling rate for dataset sampling.")
    argparser.add_argument("--disable_overwrite", action="store_true", help="Disable overwrite confirmation for existing sampled datasets.")
    args = argparser.parse_args()

    src_root = args.src_root
    dst_root = args.dst_root

    print("Starting dataset sampling... Using sample rate:", args.sample_rate)
    for task_id in args.task_ids:
        for dataset_path in args.dataset_paths:
            sample_dataset(task_id, dataset_path, args.sample_rate, overwrite=not args.disable_overwrite)
