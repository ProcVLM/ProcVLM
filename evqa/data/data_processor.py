import json
import mmap
import os
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    # start_time = time.time()
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    # print(f"Processing time: {time.time() - start_time:.2f} seconds")

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # Extract progress value from raw assistant text (avoids tokenizer decode overhead)
    progress_val = -1.0  # sentinel: no progress tag
    for turn in source["conversations"]:
        if turn["from"] != "human":
            m = re.search(r"<progress>(.*?)</progress>", turn["value"])
            if m:
                try:
                    progress_val = float(m.group(1).replace("%", "").strip())
                except Exception:
                    progress_val = -1.0
                break
    full_result["progress_gt"] = torch.tensor([progress_val], dtype=torch.float32)

    return full_result


class MmapJsonFile:
    """Memory-mapped indexed access to a JSON array or JSONL file.

    Builds a byte-offset index on first use and caches it as .mmap_idx.npy
    next to the source file.  The index is validated via file size + mtime.
    The mmap handle is created lazily so that forked DataLoader workers each
    get their own handle while sharing OS page-cache backed physical pages.
    """

    def __init__(self, file_path: str, data_path: str):
        self.file_path = str(file_path)
        self.data_path = data_path
        self._mmap = None  # created lazily per-process (must be set before _build_or_load_index)
        self._offsets = self._build_or_load_index()  # numpy (N, 2) int64

    # ---- index management ------------------------------------------------

    def _idx_paths(self):
        return self.file_path + '.mmap_idx.npy', self.file_path + '.mmap_meta.json'

    def _build_or_load_index(self):
        idx_path, meta_path = self._idx_paths()
        st = os.stat(self.file_path)

        def _cache_valid():
            if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
                return False
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                return (meta.get('size') == st.st_size
                        and meta.get('mtime') == st.st_mtime_ns)
            except Exception:
                return False

        if _cache_valid():
            rank0_print(f"Reusing mmap index: {idx_path}")
            return np.load(idx_path)

        # Index needs to be (re)built
        distributed = torch.distributed.is_initialized()
        is_main = (not distributed) or (torch.distributed.get_rank() == 0)

        build_error = None
        offsets = None
        if is_main:
            try:
                rank0_print(f"Building mmap index for {self.file_path} ...")
                if self.file_path.endswith('.jsonl'):
                    raw = self._scan_jsonl()
                else:
                    raw = self._scan_json_array()
                offsets = np.array(raw, dtype=np.int64)
                np.save(idx_path, offsets)
                with open(meta_path, 'w') as f:
                    json.dump({'size': st.st_size, 'mtime': st.st_mtime_ns}, f)
                rank0_print(f"Built index: {len(offsets)} records -> {idx_path}")
            except Exception as e:
                build_error = e

        if distributed:
            rank0_print("[CRITICAL] Initializing index in a distributed setting may cause hangs if some rank has no bonded devices."
                        "Avoid this by building the index on a single process first, using test/test_dataloader.py or similar.")
            torch.distributed.barrier()

        # rank 0: re-raise build error after barrier so other ranks are unblocked
        if is_main:
            if build_error is not None:
                raise build_error
            return offsets

        # non-main ranks: retry loading to handle NFS/shared-FS propagation delay
        for _retry in range(10):
            if os.path.exists(idx_path):
                return np.load(idx_path)
            time.sleep(0.5)
        raise FileNotFoundError(
            f"Index file {idx_path} not found after waiting. "
            f"Rank 0 may have failed to build the index for {self.file_path}"
        )

    # ---- byte-offset scanners -------------------------------------------

    def _scan_jsonl(self):
        """Record (byte_offset, byte_length) of each non-empty line."""
        offsets = []
        with open(self.file_path, 'rb') as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.rstrip(b'\r\n')
                if stripped:
                    offsets.append((start, len(stripped)))
        return offsets

    def _scan_json_array(self):
        """Record (byte_offset, byte_length) of each top-level element
        inside a JSON array, using chunked mmap reads."""
        offsets = []
        with open(self.file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            size = mm.size()

            # skip to opening '['
            pos = 0
            while pos < size and mm[pos] != 91:  # ord('[')
                pos += 1
            pos += 1

            depth = 0
            in_string = False
            escape_next = False
            elem_start = -1

            CHUNK = 8 * 1024 * 1024  # 8 MB
            while pos < size:
                end = min(pos + CHUNK, size)
                chunk = mm[pos:end]
                chunk_len = len(chunk)
                i = 0
                done = False

                while i < chunk_len:
                    c = chunk[i]

                    if escape_next:
                        escape_next = False
                        i += 1
                        continue

                    if in_string:
                        if c == 92:   # ord('\\')
                            escape_next = True
                        elif c == 34: # ord('"')
                            in_string = False
                        i += 1
                        continue

                    if c == 34:       # '"'
                        in_string = True
                        if depth == 0 and elem_start < 0:
                            elem_start = pos + i
                    elif c == 123 or c == 91:   # '{' or '['
                        if depth == 0 and elem_start < 0:
                            elem_start = pos + i
                        depth += 1
                    elif c == 125 or c == 93:   # '}' or ']'
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and elem_start >= 0:
                                offsets.append(
                                    (elem_start, pos + i - elem_start + 1)
                                )
                                elem_start = -1
                        else:
                            # closing ']' of the outer array
                            done = True
                            break

                    i += 1

                if done:
                    break
                pos = end

            mm.close()
        return offsets

    # ---- data access -----------------------------------------------------

    def _ensure_mmap(self):
        if self._mmap is None:
            fd = os.open(self.file_path, os.O_RDONLY)
            self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, i):
        self._ensure_mmap()
        offset, length = int(self._offsets[i, 0]), int(self._offsets[i, 1])
        raw = self._mmap[offset:offset + length]
        record = json.loads(raw)
        # inject data_path (same as the original per-annotation loop)
        if isinstance(record, list):
            for sub in record:
                if isinstance(sub, dict):
                    sub['data_path'] = self.data_path
        elif isinstance(record, dict):
            record['data_path'] = self.data_path
        return record

    def __del__(self):
        if self._mmap is not None:
            self._mmap.close()

    # pickling support (for spawn-mode DataLoader)
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mmap'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class MmapDataList:
    """Virtual list backed by multiple MmapJsonFile objects.

    All index data lives in a single numpy array (C-allocated, CoW-safe
    across fork), so forked DataLoader workers share physical memory.
    """

    def __init__(self, files, indices):
        """
        files:   list[MmapJsonFile]
        indices: numpy (N, 2) int64 — each row is [file_idx, local_idx]
        """
        self._files = files
        self._indices = indices  # numpy array, CoW-safe

    def shuffle(self):
        np.random.shuffle(self._indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        file_idx, local_idx = self._indices[i]
        return self._files[int(file_idx)][int(local_idx)]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args, shuffle_data=True):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        mmap_files = []
        all_indices = []

        for data in dataset_list:
            mmap_file = MmapJsonFile(data["annotation_path"], data["data_path"])
            file_idx = len(mmap_files)
            mmap_files.append(mmap_file)

            n = len(mmap_file)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                k = int(n * sampling_rate)
                selected = np.random.choice(n, k, replace=False)
                rank0_print(f"sampling {k} examples from dataset {data}")
            else:
                selected = np.arange(n, dtype=np.int64)
                rank0_print(f"dataset name: {data}")

            file_indices = np.column_stack([
                np.full(len(selected), file_idx, dtype=np.int64),
                selected.astype(np.int64),
            ])
            all_indices.append(file_indices)

        list_data_dict = MmapDataList(
            mmap_files, np.concatenate(all_indices, axis=0)
        )
        rank0_print(f"Total training samples: {len(list_data_dict)}")

        if shuffle_data:
            list_data_dict.shuffle()

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            sources = [sources]
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            # Collect per-sub-sequence progress values
            progress_gts = torch.cat(
                [d["progress_gt"] for d in data_list if "progress_gt" in d], dim=0
            )

            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
                "progress_gt": progress_gts,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        # progress ground truth passthrough
        if "progress_gt" in instances[0]:
            batch["progress_gt"] = torch.cat(
                [inst["progress_gt"] for inst in instances], dim=0
            )

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        # progress ground truth passthrough (flat concat for packed sequences)
        if "progress_gt" in instances[0]:
            batch["progress_gt"] = torch.cat(
                [inst["progress_gt"] for inst in instances], dim=0
            )

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning with Validation support."""
    print("Loading Training Dataset...")
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)


    # --- Update: Add eval dataset if specified ---
    eval_dataset = {}
    if hasattr(data_args, "val_dataset_use") and data_args.val_dataset_use:
        # --- [backup arguments] ---
        original_dataset_use = data_args.dataset_use
        original_data_packing = data_args.data_packing
        
        # --- [prepare environment] ---
        # data packing should be disabled for eval datasets, to avoid mixing samples
        data_args.data_packing = False 
        val_configs = data_args.val_dataset_use.split(",") # "indomain:dsname%20,outdomain:dsname%20"
        for val_conf in val_configs:
            # parse name:path
            if ":" in val_conf:
                ds_name, ds_path = val_conf.split(":", 1)
            else:
                ds_name = "validation"
                ds_path = val_conf
            print(f"Loading Validation Dataset [{ds_name}] from: {ds_path}")
            # --- [swap arguments] ---
            data_args.dataset_use = ds_path
            try:
                # Instantiate validation dataset (it reads ds_path at this point)
                val_ds = LazySupervisedDataset(processor, data_args=data_args, shuffle_data=False)
                eval_dataset[ds_name] = val_ds
            except Exception as e:
                print(f"Error loading validation dataset {ds_name}: {e}")
        # --- [restore arguments] ---
        data_args.dataset_use = original_dataset_use
        data_args.data_packing = original_data_packing
    if len(eval_dataset) == 0:
        eval_dataset = None
    # --- End of eval dataset preparation ---
    print("Evaluation Dataset(s): ", eval_dataset.keys() if eval_dataset else "None")

    standard_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    if data_args.data_flatten or data_args.data_packing:
        train_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    else:
        train_collator = standard_collator
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=train_collator,
        eval_data_collator=standard_collator # data packing MUST be disabled during evaluation, if --predict_with_generate is enabled
    )


if __name__ == "__main__":
    pass
