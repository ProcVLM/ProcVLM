import argparse
import json
import os
import random
from typing import List
from pathlib import Path
from tqdm import tqdm
from evqa.tools.functions import iter_episode_files

def write_jsonl_batch(dst_file, batch: List[str]):
    """
    Safely write a batch of raw JSONL lines:
    - parse JSON to ensure correctness
    - dump JSON to normalize unicode
    - flush as one large write
    """
    out_lines = []
    for raw in batch:
        try:
            obj = json.loads(raw)     # ensure valid JSON
            safe_line = json.dumps(obj, ensure_ascii=False)
            out_lines.append(safe_line + "\n")
        except Exception as e:
            raise ValueError(f"Invalid JSON line encountered: {raw[:200]}") from e

    dst_file.write("".join(out_lines))
    dst_file.flush()

def load_lines(path):
    with open(path) as f:
        for line in f:
            yield line

def load_selected_test(path):
    """
    Load selected test episodes from a jsonl file.
    Each line in the file is a JSON object like:
    {"dataset_name": "...", "episode_name": "..."}
    """
    selected = set()
    for line in load_lines(path):
        j = json.loads(line)
        selected.add((j["dataset_name"], j["episode_name"]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs="+", required=True)
    parser.add_argument("--valid_datasets", nargs="*", default=[])
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--selected_test_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disabled_clusters", nargs="*", default=[], help="List of cluster names to exclude (e.g., 'cluster_1 cluster_2')")
    args = parser.parse_args()

    random.seed(args.seed)

    # -----------------------------
    # Configurations
    # -----------------------------
    ds_names = [Path(p).name for p in args.dataset_paths]
    fixed_valid = set(Path(p).name for p in args.valid_datasets)

    # If selected_test_file is specified, test_ratio must be 0
    selected_test = None
    if args.selected_test_file:
        selected_test = load_selected_test(args.selected_test_file)
    if args.valid_datasets and args.valid_ratio > 0.0:
        raise ValueError("valid_ratio must be 0.0 when valid_datasets is specified, as those datasets are directly used as validation set, and other datasets are split between train and test only.")
    if args.train_ratio + args.test_ratio + args.valid_ratio != 1.0:
        raise ValueError("train_ratio + test_ratio + valid_ratio must equal to 1.0")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_f = open(output_root / "train.jsonl", "w")
    test_f  = open(output_root / "test.jsonl", "w")
    valid_f = open(output_root / "valid.jsonl", "w")

    # -----------------------------
    # Batch buffers
    # -----------------------------
    BATCH_SIZE = 20000
    buf_train, buf_test, buf_valid = [], [], []

    def flush(buf, f):
        if buf:
            write_jsonl_batch(f, buf)
            buf.clear()

    # -----------------------------
    # Main Loop
    # All Valid = fixed valid (if any) / random split (if valid_datasets not specified)
    # All Test = random select from selected_test at a higher priority (if specified) + random split, the random split always happens
    # -----------------------------
    for cluster, ds_name, ep in tqdm(iter_episode_files(input_root), desc="Applying Train-Test-Valid Split"):

        if cluster in args.disabled_clusters:
            continue

        if ds_name not in ds_names:
            continue

        # fixed valid dataset directly goes to valid
        if ds_name in fixed_valid:
            for line in load_lines(ep):
                buf_valid.append(line)
                if len(buf_valid) >= BATCH_SIZE:
                    flush(buf_valid, valid_f)
            continue

        ep_name = ep.stem

        if selected_test is not None and (ds_name, ep_name) in selected_test:
            # -----------------------------
            # Case 1: selected_test_file specified test set
            # -----------------------------
            if random.random() < 0.3:   # chosen as test at fix ratio 0.3
                dst_buf = buf_test
                dst_f = test_f
            else:   # otherwise split between train & valid
                r = random.random()
                if r < args.train_ratio / (args.train_ratio + args.valid_ratio):
                    dst_buf = buf_train
                    dst_f = train_f
                else:
                    dst_buf = buf_valid
                    dst_f = valid_f
        else:
            # -----------------------------
            # Case 2: Normal random split
            # -----------------------------
            r = random.random()
            if r < args.train_ratio:
                dst_buf = buf_train
                dst_f = train_f
            elif r < args.train_ratio + args.test_ratio:
                dst_buf = buf_test
                dst_f = test_f
            else:
                dst_buf = buf_valid
                dst_f = valid_f

        # -----------------------------
        # Load episode lines
        # -----------------------------
        for line in load_lines(ep):
            dst_buf.append(line)
            if len(dst_buf) >= BATCH_SIZE:
                flush(dst_buf, dst_f)

    # -----------------------------
    # Final flush
    # -----------------------------
    flush(buf_train, train_f)
    flush(buf_test, test_f)
    flush(buf_valid, valid_f)

    train_f.close()
    test_f.close()
    valid_f.close()

    # auto remove empty files
    for f in [train_f, test_f, valid_f]:
        if os.path.getsize(f.name) == 0:
            os.remove(f.name)

if __name__ == "__main__":
    main()
