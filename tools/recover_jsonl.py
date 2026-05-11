"""
Copy all '.<suffix>.jsonl' files from every sub-directory in 'src_dir' to the same-named sub-directory in 'dst_dir'.
If the destination file already exists, skip copying.
"""

# suffix = 'sub_task'
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("--src_dir", type=str, required=True, help="Source directory containing dataset sub-directories.")
args.add_argument("--dst_dir", type=str, required=True, help="Destination directory to copy files to.")
args.add_argument("--suffix", type=str, required=True, help="Suffix of the jsonl files to copy (e.g., 'sub_task', 'bbox').")
args = args.parse_args()

src_dir = args.src_dir
dst_dir = args.dst_dir
suffix = args.suffix

# skip_list = [
#     'real_world_tabletop_tasks_1013',
#     'simulation_tabletop_tasks_1013',
#     'libero_v21',
# ]

src_path = Path(src_dir)
dst_path = Path(dst_dir)

for ds_dir in tqdm(list(src_path.iterdir()), desc="Datasets"):
    if not ds_dir.is_dir():
        continue
    dsname = ds_dir.name

    # if dsname in skip_list:
    #     print(f"Skipping dataset {dsname}")
    #     continue

    dst_ds_dir = dst_path / dsname
    if not dst_ds_dir.exists():
        dst_ds_dir.mkdir(parents=True, exist_ok=True)

    for pf in ds_dir.glob(f'*.{suffix}.jsonl'):
        dest_file = dst_ds_dir / pf.name
        if dest_file.exists():
            print(f"Skipping existing file {dest_file}")
        else:
            shutil.copy(pf, dest_file)
            print(f"Copied {pf} to {dest_file}")