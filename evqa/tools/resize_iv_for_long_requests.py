import argparse
import cv2
import os
from PIL import Image
from dotenv import load_dotenv
from typing import List
from pathlib import Path
from tqdm import tqdm
from core.utils.common import load_jsonlines, write_jsonlines
from core.utils.runner_utils import auto_split_datasets_entirely
from evqa.tools.functions import iter_episode_files
load_dotenv()


data_base_path = os.path.join(os.getenv("EVQA_DATASET_ROOT"), "data")

def check_and_maybe_resize_iv(cluster: str, ds_name: str, jl: List[dict], max_tokens: int):
    updated = False
    target_res = (320, 320)
    
    for item in jl:
        total_tokens = item['num_tokens']
        if total_tokens <= max_tokens:
            continue

        img_meta = [] # (abs_path, rel_path, w, h)
        if "image" in item:
            if isinstance(item["image"], str):
                item["image"] = [item["image"]]
            for rel_path in item["image"]:
                abs_path = os.path.join(data_base_path, rel_path)
                try:
                    with Image.open(abs_path) as img:
                        w, h = img.size
                    img_meta.append((abs_path, rel_path, w, h))
                except Exception as e:
                    print(f"Warning: Failed to read image {abs_path}: {e}")

        vid_meta = None # (abs_path, rel_path, w, h, duration, fps)
        if "video" in item:
            rel_path = item["video"]
            abs_path = os.path.join(data_base_path, rel_path)
            try:
                cap = cv2.VideoCapture(abs_path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_cnt / fps if fps > 0 else 0
                    cap.release()
                    vid_meta = (abs_path, rel_path, w, h, duration, fps)
            except Exception as e:
                print(f"Warning: Failed to read video {abs_path}: {e}")

        # 4. 超过阈值，执行Resize操作
        if total_tokens > max_tokens:
            print(f"Resizing IV for [{cluster}]{ds_name}: total_tokens={total_tokens} > {max_tokens}")
            if "image" in item:
                img0_w, img0_h = img_meta[0][2], img_meta[0][3]
                print(f"  Original image numbers: {len(item['image'])}, size: {img0_w}x{img0_h}")
            if "video" in item:
                print(f"  Original video present.")
            updated = True

            # --- check if already resized (with _320 suffix), skip ---
            already_resized = False
            if img_meta:
                for _, rel_p, _, _ in img_meta:
                    if rel_p.endswith("_320.jpg") or rel_p.endswith("_320.png"):
                        already_resized = True
                        break
            if vid_meta and not already_resized:
                _, rel_p, _, _, _, _ = vid_meta
                if rel_p.endswith("_320.mp4") or rel_p.endswith("_320.mov"):
                    already_resized = True
            if already_resized:
                print(f"  Already resized, skipping.")
                continue
            
            # --- 处理图像 ---
            if img_meta:
                new_rel_paths = []
                for abs_p, rel_p, w, h in img_meta:
                    path_parts = os.path.splitext(rel_p)
                    new_rel = f"{path_parts[0]}_320{path_parts[1]}"
                    new_abs = os.path.join(data_base_path, new_rel)
                    
                    if not os.path.exists(new_abs):
                        try:
                            with Image.open(abs_p) as img:
                                img = img.resize(target_res)
                                img.save(new_abs)
                        except Exception as e:
                            print(f"Error resizing image {abs_p}: {e}")
                    new_rel_paths.append(new_rel)
                item["image"] = new_rel_paths # 更新路径

            # --- 处理视频 ---
            if vid_meta:
                abs_p, rel_p, w, h, dur, fps = vid_meta
                path_parts = os.path.splitext(rel_p)
                new_rel = f"{path_parts[0]}_320{path_parts[1]}"
                new_abs = os.path.join(data_base_path, new_rel)
                
                if not os.path.exists(new_abs):
                    try:
                        cap = cv2.VideoCapture(abs_p)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(new_abs, fourcc, fps, target_res)
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame = cv2.resize(frame, target_res)
                            out.write(frame)
                        cap.release()
                        out.release()
                    except Exception as e:
                        print(f"Error resizing video {abs_p}: {e}")
                item["video"] = new_rel # 更新路径

    return updated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs="+", required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    DATASETS = auto_split_datasets_entirely(args.dataset_paths, args.world_size, args.rank)

    # -----------------------------
    # Configurations
    # -----------------------------
    ds_names = [Path(p).name for p in DATASETS]
    working_dir = Path(args.work_root)

    for cluster, ds_name, ds_jsonl_path in tqdm(iter_episode_files(working_dir), desc=f"Resizing IV [{args.rank}/{args.world_size}]"):
        if ds_name not in ds_names:
            continue
        jl = load_jsonlines(ds_jsonl_path)
        if check_and_maybe_resize_iv(cluster, ds_name, jl, max_tokens=args.max_tokens):
            # breakpoint()
            write_jsonlines(jl, ds_jsonl_path)

if __name__ == "__main__":
    main()
