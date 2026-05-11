import argparse
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from PIL import Image
from moviepy.editor import VideoFileClip
from pathlib import Path
from core.data.generals import EXO_CAMERA_MAP, EGO_CAMERA_MAP
from datasets import Dataset, Features, Value, Sequence
from datasets import Image as datasets_Image

def get_dataset_camera_keys(dataset_path: str):
    dataset_name = Path(dataset_path).name
    third_cam_key = EXO_CAMERA_MAP.get(dataset_name, 'observation.images.cam')
    # third_cam_key = EXO_CAMERA_MAP.get(dataset_name, 'observation.images.cam_0')
    ego_cam_key = EGO_CAMERA_MAP.get(dataset_name, None)
    return third_cam_key, ego_cam_key

def repair_video(video_path: str, backup_dir: str):
    """
    修复视频的红蓝通道错误（交换R和B维度），使用MoviePy实现。
    自动将原视频移动到备份目录，并在原位置生成修复后的视频。

    Args:
        video_path (str): 原视频路径 (.mp4)
        backup_dir (str): 备份目录
    """
    video_path = Path(video_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    temp_output = video_path.parent / f"{video_path.stem}_fixed{video_path.suffix}"

    print(f"🧩 修复视频通道: {video_path}")

    # 使用MoviePy加载视频
    clip = VideoFileClip(str(video_path))

    # 对每帧进行R/B通道交换（RGB→BGR）
    def swap_rb(frame):
        # frame shape: (H, W, 3), dtype=np.uint8
        return frame[..., ::-1].copy()

    repaired = clip.fl_image(swap_rb)

    # 写出修复视频（编码器使用libx264，兼容性最高）
    repaired.write_videofile(
        str(temp_output),
        codec='libx264',
        audio=False,
        verbose=False,
        logger=None
    )

    clip.close()
    repaired.close()

    # 移动原文件到备份目录
    backup_path = backup_dir / video_path.name
    shutil.move(str(video_path), str(backup_path))

    # 将修复后的视频改名为原文件名
    temp_output.rename(video_path)

    print(f"✅ 修复完成: {video_path}")
    print(f"📦 原视频已备份至: {backup_path}")

def repair_video_root(dataset_path: str, cam_key: str):
    video_root = Path(dataset_path) / "videos"
    sub_dirs = [d for d in video_root.iterdir() if d.is_dir()]
    for sub_dir in sub_dirs: # chunk-00x/
        video_dir = sub_dir / cam_key
        if not video_dir.exists():
            print(f"Video dir {video_dir} does not exist, skipping.")
            continue
        video_files = [f for f in video_dir.iterdir() if f.suffix == ".mp4"]
        backup_dir = Path(dataset_path) / "video_backups" / sub_dir.name / cam_key
        for video_file in video_files:
            repair_video(str(video_file), str(backup_dir))


# parquet repair
def backup_parquet(dataset_path: str):
    dataset_root = Path(dataset_path)
    backup_dir = dataset_root / "data_backups"
    print(f"🗂️ Backing up parquet files in dataset: {backup_dir}")
    if backup_dir.exists():
        print(f"Backup dir {backup_dir} already exists, skipping backup.")
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / "data"
    # cp -r [target_dir/*] to [backup_dir/]
    for item in tqdm(target_dir.iterdir(), desc="Backing up parquet files"):
        dest = backup_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"✅ Backed up data from {target_dir} to {backup_dir}")

features_rh20t = Features({
    "observation.images.cam_0": datasets_Image(),
    "observation.state": Sequence(
        feature=Value(dtype="float32"),
        length=-1  # -1 represents variable length
    ),
    "action": Sequence(
        feature=Value(dtype="float32"),
        length=-1
    ),
    "timestamp": Value(dtype="float32"),
    "frame_index": Value(dtype="int64"),
    "episode_index": Value(dtype="int64"),
    "index": Value(dtype="int64"),
    "task_index": Value(dtype="int64"),
    "next.done": Value(dtype="bool"),
    "embodiment_id": Value(dtype="int64")
})

features_berkeley = Features({
    "observation.images.cam": datasets_Image(),
    "observation.images.cam1": datasets_Image(),
    "observation.state": Sequence(
        feature=Value(dtype="float32"),
        length=-1  # -1 represents variable length
    ),
    "action": Sequence(
        feature=Value(dtype="float32"),
        length=-1
    ),
    "timestamp": Value(dtype="float32"),
    "next.done": Value(dtype="bool"),
    "frame_index": Value(dtype="int64"),
    "episode_index": Value(dtype="int64"),
    "index": Value(dtype="int64"),
    "task_index": Value(dtype="int64"),
    "embodiment_id": Value(dtype="int64"),
    "sub_task_index": Value(dtype="int64"),
    "cot_index": Value(dtype="int64"),
    "bbox_index": Value(dtype="int64"),
})

features_stanford = features_berkeley
features_utaustin = features_berkeley

def fix_image_rb(image: datasets_Image) -> datasets_Image:
    """Fix the R and B channels of a datasets.Image by swapping them."""
    # image: <class 'PIL.PngImagePlugin.PngImageFile'>
    img_array = np.array(image)  # Convert to numpy array
    if img_array.shape[2] != 3:
        raise ValueError("Image does not have 3 channels.")
    # Swap R and B channels
    img_array[..., [0, 2]] = img_array[..., [2, 0]]
    fixed_image = Image.fromarray(img_array)

    # save to tmp/hhmmss_rb_fixed.png for debug
    import time, os, random
    if random.random() < 0.001:  # save only 0.1% of images for debug
        debug_dir = "tmp"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime("%H%M%S", time.localtime())
        debug_path = os.path.join(debug_dir, f"{timestamp}_rb_fixed.png")
        fixed_image.save(debug_path)
        print(f"Saved fixed image for debug: {debug_path}")

    return fixed_image

def safe_dataset_from_parquet(df: pd.DataFrame, features):
    # remove columns not in features
    df = df[[c for c in df.columns if c in features]]
    # repair dtype dismatch from features
    for col, feat in features.items():
        if isinstance(feat, Sequence) and col in df.columns:
            df[col] = df[col].apply(lambda x: [float(v) for v in x])
        elif isinstance(feat, Value):
            dtype = feat.dtype
            if dtype == "float32":
                df[col] = df[col].astype("float32", errors="ignore")
            elif dtype == "int64":
                df[col] = df[col].astype("int64", errors="ignore")
    # # repair image columns 
    # for cam_key in df.columns:
    #     if cam_key.startswith("observation.images."):
    #         if isinstance(df[cam_key].iloc[0], dict):
    #             df[cam_key] = df[cam_key].apply(lambda x: x.get("path", None))
    # create Dataset
    return Dataset.from_pandas(df, features=features)

def repair_parquet(parquet_path: str, cam_key: str, dataset_name: str):
    df = pd.read_parquet(parquet_path)
    print(f"🧩 Repairing parquet: {parquet_path} for {cam_key}")
    if dataset_name.startswith("RH20T"):
        dataset = safe_dataset_from_parquet(df, features_rh20t)
    elif dataset_name.startswith("berkeley"):
        dataset = safe_dataset_from_parquet(df, features_berkeley)
    elif dataset_name.startswith("utaustin"):
        dataset = safe_dataset_from_parquet(df, features_utaustin)
    elif dataset_name.startswith("stanford_hydra"):
        dataset = safe_dataset_from_parquet(df, features_stanford)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported for parquet repair.")
    if cam_key not in dataset.column_names:
        print(f"Camera key {cam_key} not in parquet columns, skipping {parquet_path}.")
        return
    # fix every image in the cam_key column, swap R and B channels
    dataset = dataset.map(
        lambda example: {cam_key: fix_image_rb(example[cam_key])},
        num_proc=32,
        batched=False
    )
    # save back to parquet
    dataset.to_parquet(parquet_path)
    print(f"✅ Repaired parquet saved: {parquet_path}")

def repair_parquet_root(dataset_path: str, cam_key: str):
    backup_parquet(dataset_path)
    dataset_root = Path(dataset_path)
    data_dir = dataset_root / "data"
    sub_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    for sub_dir in sub_dirs: # chunk-00x/
        parquet_files = [f for f in sub_dir.iterdir() if f.suffix == ".parquet"]
        for parquet_file in tqdm(parquet_files):
            repair_parquet(str(parquet_file), cam_key, dataset_root.name)


#
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--third_problem_datasets", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args.add_argument("--ego_problem_datasets", type=str, nargs='*', required=True, help="List of dataset paths to process. Must be specified.")
    args.add_argument("--third_problem_datasets_video", type=str, nargs='*', required=True, help="List of dataset paths with video to process. Must be specified.")
    args.add_argument("--ego_problem_datasets_video", type=str, nargs='*', required=True, help="List of dataset paths with video to process. Must be specified.")
    args = args.parse_args()

    PROBLEM_DATASETS_EGO = args.ego_problem_datasets
    PROBLEM_DATASETS_THIRD = args.third_problem_datasets
    PROBLEM_DATASETS_THIRD_VIDEO = args.third_problem_datasets_video
    PROBLEM_DATASETS_EGO_VIDEO = args.ego_problem_datasets_video

    for dataset_path in PROBLEM_DATASETS_THIRD_VIDEO:
        third_cam_key, _ = get_dataset_camera_keys(dataset_path)
        if third_cam_key is None:
            print(f"No third camera key for dataset {dataset_path}, skipping.")
            continue
        repair_video_root(dataset_path, third_cam_key)

    for dataset_path in PROBLEM_DATASETS_EGO_VIDEO:
        _, ego_cam_key = get_dataset_camera_keys(dataset_path)
        if ego_cam_key is None:
            print(f"No ego camera key for dataset {dataset_path}, skipping.")
            continue
        repair_video_root(dataset_path, ego_cam_key)

    for dataset_path in PROBLEM_DATASETS_THIRD:
        third_cam_key, _ = get_dataset_camera_keys(dataset_path)
        if third_cam_key is None:
            print(f"No third camera key for dataset {dataset_path}, skipping.")
            continue
        repair_parquet_root(dataset_path, third_cam_key)

    for dataset_path in PROBLEM_DATASETS_EGO:
        _, ego_cam_key = get_dataset_camera_keys(dataset_path)
        if ego_cam_key is None:
            print(f"No ego camera key for dataset {dataset_path}, skipping.")
            continue
        repair_parquet_root(dataset_path, ego_cam_key)

    pass
