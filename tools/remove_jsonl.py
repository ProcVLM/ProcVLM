"""
python tools/remove_jsonl.py --src /path/to/jsonl_files --category plan --dataset_paths /path/to/dataset1 /path/to/dataset2
"""

import os
import shutil
import argparse
import datetime
from dotenv import load_dotenv
# from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
load_dotenv()

current_run_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

#
argparser = argparse.ArgumentParser()
argparser.add_argument("--src", type=str, required=True, help="Source root directory containing jsonl files to be archived")
argparser.add_argument("--category", type=str, required=True, default="plan", choices=["plan", "subtask", "bbox", "cot", "all", "subfolder", "borders"], help="Which type of files to archive")
argparser.add_argument("--copy_only", action="store_true", help="Whether to copy instead of move files.", default=False)
argparser.add_argument("--disable_override", action="store_true", help="Whether to override existing files.", default=False)
argparser.add_argument("--dataset_paths", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
args = argparser.parse_args()
global_override = not args.disable_override
if not global_override:
    print("Warning: Not overriding existing files. Existing files will be skipped.")
    print("If some files already exist, your source files will still be deleted but not archived. This may lead to unrecoverable data loss.")
    print("Are you sure you want to continue? (y/n)")
    ans = input()
    if ans.lower() != 'y':
        print("Aborting.")
        exit(0)
global_copy_only = args.copy_only
if global_copy_only:
    print("Warning: Copying files instead of moving. Source files will not be deleted.")
_raw_datasets = args.dataset_paths

print("Datasets to process:")
for ds in _raw_datasets:
    print(f" - {ds}")
print(f"Total datasets: {len(_raw_datasets)}, are you sure to proceed? (y/n)")
ans = input()
if ans.lower() != 'y':
    print("Aborting.")
    exit(0)

src_dir = Path(args.src)

archive_root_dir = Path(os.getenv("ARCHIVE_JSONL_DIR") + "/deprecated") / src_dir.name
if not archive_root_dir.exists():
    archive_root_dir.mkdir(parents=True, exist_ok=True)

print(f"Archiving files from {src_dir} to {archive_root_dir}" + (" by copying." if global_copy_only else " by moving."))
print("Proceed? (y/n)")
ans = input()
if ans.lower() != 'y':
    print("Aborting.")
    exit(0)

def archive_plan(datasets: list, plan_root_dir: str):
    plan_root = Path(plan_root_dir)
    assert plan_root.exists(), f"Plan root directory {plan_root} does not exist."
    for ds in datasets:
        dsname = Path(ds).name
        ds_plan_dir = plan_root / dsname
        if not ds_plan_dir.exists():
            print(f"Plan directory for dataset {dsname} does not exist, skipping.")
            continue
        # move all .plan.jsonl files to archive_root_dir/dsname/
        archive_ds_dir = archive_root_dir / dsname
        if not archive_ds_dir.exists():
            archive_ds_dir.mkdir(parents=True, exist_ok=True)
        plan_files = list(ds_plan_dir.glob("*.plan.jsonl"))
        for pf in plan_files:
            dest_file = archive_ds_dir / pf.name
            if dest_file.exists():
                if global_override:
                    print(f"Overriding existing file {dest_file}")
                else:
                    if not global_copy_only:
                        # just unlink the source file
                        print(f"File {dest_file} already exists, skipping and deleting source file {pf}")
                        pf.unlink()
                    continue
            if global_copy_only:
                shutil.copy(pf, dest_file)
                print(f"Copied {pf} to {dest_file}")
            else:
                # move to archive
                pf.rename(dest_file)
                print(f"Archived {pf} to {dest_file}")

def archive_subtask(datasets: list, subtask_root_dir: str):
    subtask_root = Path(subtask_root_dir)
    assert subtask_root.exists(), f"Subtask root directory {subtask_root} does not exist."
    for ds in datasets:
        dsname = Path(ds).name
        ds_subtask_dir = subtask_root / dsname
        if not ds_subtask_dir.exists():
            print(f"Subtask directory for dataset {dsname} does not exist, skipping.")
            continue
        # move all .sub_task.jsonl files to archive_root_dir/dsname/
        archive_ds_dir = archive_root_dir / dsname
        if not archive_ds_dir.exists():
            archive_ds_dir.mkdir(parents=True, exist_ok=True)
        subtask_files = list(ds_subtask_dir.glob("*.sub_task.jsonl"))
        for sf in subtask_files:
            dest_file = archive_ds_dir / sf.name
            if dest_file.exists():
                if global_override:
                    print(f"Overriding existing file {dest_file}")
                else:
                    if not global_copy_only:
                        # just unlink the source file
                        print(f"File {dest_file} already exists, skipping and deleting source file {sf}")
                        sf.unlink()
                    continue
            if global_copy_only:
                shutil.copy(sf, dest_file)
                print(f"Copied {sf} to {dest_file}")
            else:
                # move to archive
                sf.rename(dest_file)
                print(f"Archived {sf} to {dest_file}")

def remove_subfolder(datasets: list, subfolder_name: str):
    for ds in datasets:
        dsname = Path(ds).name
        ds_subfolder = Path(ds) / subfolder_name
        if ds_subfolder.exists() and ds_subfolder.is_dir():
            try:
                for item in ds_subfolder.iterdir():
                    if item.is_dir():
                        remove_subfolder([item], "")  # Recursively remove subdirectories
                    else:
                        item.unlink()  # Remove file
                ds_subfolder.rmdir()  # Remove the now-empty directory
                print(f"Removed subfolder {ds_subfolder}")
            except Exception as e:
                print(f"Error removing subfolder {ds_subfolder}: {e}")
        else:
            print(f"Subfolder {ds_subfolder} does not exist, skipping.")

def remove_borders_file(datasets: list, borders_root_dir: str):
    for ds in datasets:
        dsname = Path(ds).name
        ds_borders_dir = Path(borders_root_dir) / dsname
        borders_file = ds_borders_dir / "borders.pkl"
        if borders_file.exists():
            borders_file.unlink()
            print(f"Removed borders file {borders_file}")
        else:
            print(f"Borders file {borders_file} does not exist, skipping.")

def archive_bbox(datasets: list, bbox_root_dir: str):
    bbox_root = Path(bbox_root_dir)
    assert bbox_root.exists(), f"BBox root directory {bbox_root} does not exist."
    for ds in datasets:
        dsname = Path(ds).name
        ds_bbox_dir = bbox_root / dsname
        if not ds_bbox_dir.exists():
            print(f"BBox directory for dataset {dsname} does not exist, skipping.")
            continue
        # move all .bbox.jsonl files to archive_root_dir/dsname/
        archive_ds_dir = archive_root_dir / dsname
        if not archive_ds_dir.exists():
            archive_ds_dir.mkdir(parents=True, exist_ok=True)
        bbox_files = list(ds_bbox_dir.glob("*.bbox.jsonl"))
        for bf in bbox_files:
            dest_file = archive_ds_dir / bf.name
            if dest_file.exists():
                if global_override:
                    print(f"Overriding existing file {dest_file}")
                else:
                    if not global_copy_only:
                        # just unlink the source file
                        print(f"File {dest_file} already exists, skipping and deleting source file {bf}")
                        bf.unlink()
                    continue
            if global_copy_only:
                shutil.copy(bf, dest_file)
                print(f"Copied {bf} to {dest_file}")
            else:
                # move to archive
                bf.rename(dest_file)
                print(f"Archived {bf} to {dest_file}")

def archive_cot(datasets: list, cot_root_dir: str):
    cot_root = Path(cot_root_dir)
    assert cot_root.exists(), f"COT root directory {cot_root} does not exist."
    for ds in datasets:
        dsname = Path(ds).name
        ds_cot_dir = cot_root / dsname
        if not ds_cot_dir.exists():
            print(f"COT directory for dataset {dsname} does not exist, skipping.")
            continue
        # move all .cot.jsonl files to archive_root_dir/dsname/
        archive_ds_dir = archive_root_dir / dsname
        if not archive_ds_dir.exists():
            archive_ds_dir.mkdir(parents=True, exist_ok=True)
        cot_files = list(ds_cot_dir.glob("*.cot.jsonl"))
        for cf in cot_files:
            dest_file = archive_ds_dir / cf.name
            if dest_file.exists():
                if global_override:
                    print(f"Overriding existing file {dest_file}")
                else:
                    if not global_copy_only:
                        # just unlink the source file
                        print(f"File {dest_file} already exists, skipping and deleting source file {cf}")
                        cf.unlink()
                    continue
            if global_copy_only:
                shutil.copy(cf, dest_file)
                print(f"Copied {cf} to {dest_file}")
            else:
                # move to archive
                cf.rename(dest_file)
                print(f"Archived {cf} to {dest_file}")

if __name__ == "__main__":
    if args.category == "plan":
        archive_plan(_raw_datasets, src_dir)
    elif args.category == "subtask":
        archive_subtask(_raw_datasets, src_dir)
    elif args.category == "all":
        archive_plan(_raw_datasets, src_dir)
        archive_subtask(_raw_datasets, src_dir)
        archive_bbox(_raw_datasets, src_dir)
        archive_cot(_raw_datasets, src_dir)
    elif args.category == "subfolder":
        remove_subfolder(_raw_datasets, "sub_task_jsonl")
    elif args.category == "bbox":
        archive_bbox(_raw_datasets, src_dir)
    elif args.category == "cot":
        archive_cot(_raw_datasets, src_dir)
    elif args.category == "borders":
        remove_borders_file(_raw_datasets, src_dir)
    else:
        print(f"Not implemented category {args.category}, exiting.")
        exit(1)