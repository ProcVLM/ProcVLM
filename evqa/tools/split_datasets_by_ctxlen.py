import argparse
import random
import os
import json
from torchcodec.decoders import VideoDecoder
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

def check_video_duration(data: Tuple[int, str]) -> Tuple[int, Optional[float]]:
    idx, abs_path = data
    try:
        decoder = VideoDecoder(abs_path)
        duration = decoder.metadata.duration_seconds
        assert isinstance(duration, (float, int))
        return idx, duration
    except Exception as e:
        print(f"Warning: Failed to read video at {abs_path}. Marking duration as -1.0. Error: {e}")
        return idx, -1.0

def process_split(
    jsonl_path: Path,
    args: argparse.Namespace
):
    print(f"Loading {jsonl_path} into memory...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_len = len(dataset)
    video_data = []
    for idx, item in enumerate(tqdm(dataset, desc="Preparing video paths", total=total_len)):
        if 'video' in item:
            rel_path = item['video']
            abs_path = os.path.join(args.data_base_path, rel_path)
            video_data.append((idx, abs_path))
    print(f"Total videos to validate: {len(video_data)}")
    durations = []
    for data in tqdm(video_data, total=len(video_data), desc="Validating Videos"):
        durations.append(check_video_duration(data))
    print("Merging durations back to dataset...")
    duration_dict = {idx: duration for idx, duration in durations if duration is not None}
    durations_full = [duration_dict.get(i, None) for i in range(total_len)]

    long_path = Path(args.long_context_output_path)
    short_path = Path(args.short_context_output_path)
    long_path.parent.mkdir(parents=True, exist_ok=True)
    short_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {long_path} and {short_path}...")
    
    with open(long_path, 'w', encoding='utf-8', buffering=64*1024) as f_long, \
         open(short_path, 'w', encoding='utf-8', buffering=64*1024) as f_short:
        
        for item, duration in tqdm(zip(dataset, durations_full), total=total_len, desc="Splitting"):
            if duration is not None and duration < 1.0:
                continue  # Skip invalid or too short videos
            
            line = json.dumps(item, ensure_ascii=False) + '\n'
            total_tokens = item.get('num_tokens', 0)

            if total_tokens >= args.long_context_min_length:
                if total_tokens <= args.long_context_max_length:
                    f_long.write(line)
            else:
                f_short.write(line)
                if random.random() < args.short_context_mix_ratio:
                    f_long.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--long_context_min_length", type=int, default=8192, help="The context length threshold to split datasets.")
    parser.add_argument("--long_context_max_length", type=int, default=16384, help="The maximum context length for the long context dataset. Samples exceeding this will be filtered out.")
    parser.add_argument("--short_context_mix_ratio", type=float, default=0.01, help="The ratio of shorter context samples mixed into the long context dataset (replay). Recommend 70% replay and 30% new samples.")
    parser.add_argument("--data_base_path", required=True, help="The base path to resolve relative image/video paths.")
    parser.add_argument("--long_context_output_path", required=True, help="The output path for the long context dataset.")
    parser.add_argument("--short_context_output_path", required=True, help="The output path for the short context dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    src_path = Path(args.src)
    
    if src_path.is_file():
        process_split(src_path, args)
    elif src_path.is_dir():
        for jsonl_name in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            jsonl_path = src_path / jsonl_name
            if jsonl_path.exists():
                process_split(jsonl_path, args)

if __name__ == "__main__":
    main()