import json
import argparse
from pathlib import Path
from core.data.reader import FastLerobotVLReader
from pathlib import Path
import json
import pandas as pd
import numpy as np

def flatten_parquet(pf_path: str, backup_dir: str):
    """
    Fill missing sub_task_index (-1) in a LeRobot parquet file with
    the last seen valid sub_task_index, and save a JSON backup of the
    original sub_task_index mapping.

    Args:
        pf_path: path to the parquet file
        backup_dir: directory where the backup JSON will be stored
    """
    pf_path = Path(pf_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    json_path = backup_dir / f"{pf_path.stem}.json"

    print(f"[*] Loading {pf_path} ...")
    df = pd.read_parquet(pf_path, engine="pyarrow")

    if 'frame_index' not in df.columns or 'sub_task_index' not in df.columns:
        raise KeyError(f"Missing required columns in {pf_path}")

    frame_index = df['frame_index'].astype(int).to_numpy()
    sub_task_index = df['sub_task_index'].astype(int).to_numpy()

    # record original valid sub_task_index values
    valid_mask = sub_task_index >= 0
    original_stidx_dict = {int(f): int(st) for f, st in zip(frame_index[valid_mask], sub_task_index[valid_mask])}

    # save JSON backup
    with open(json_path, "w") as f:
        json.dump(original_stidx_dict, f, indent=2)
    print(f"[*] Backup saved to {json_path}")

    # fill forward the last seen valid sub_task_index
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        print(f"[!] No valid sub_task_index found in {pf_path}")
        return

    last_valid = sub_task_index[valid_indices[0]]
    for i in range(len(sub_task_index)):
        if sub_task_index[i] >= 0:
            last_valid = sub_task_index[i]
        else:
            sub_task_index[i] = last_valid

    df['sub_task_index'] = sub_task_index

    # write safely to disk
    tmp_path = pf_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")

    # atomic replace
    tmp_path.replace(pf_path)
    print(f"[*] Updated parquet saved to {pf_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args = args.parse_args()
    datasets = args.dataset_paths
    for df in datasets:
        dataset = FastLerobotVLReader(root=df)
        parquet_paths = dataset.all_episode_paths
        for pf in parquet_paths:
            flatten_parquet(pf, Path(df) / 'sub_task_index_backup')
            # break