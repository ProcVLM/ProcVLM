import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

# Define placeholders for dataset paths
data_dict = {
    "train_s1_8k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/stage1/trun8192_pack.json"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "train_s1_16k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/stage1/ext16384_pack.json"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "train_s2_8k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/stage2/trun8192_pack.json"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "train_s2_16k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/stage2/ext16384_pack.json"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "test_8k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/test/trun8192_200.jsonl"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "test_full": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/test/full.jsonl"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "valid_8k": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/valid/trun8192_200.jsonl"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "valid_full": {
        "annotation_path": os.path.join(os.getenv("EVQA_DATASET_ROOT"), "jsons/valid/full.jsonl"),
        "data_path": os.getenv("EVQA_DATASET_ROOT"),
    },
    "train_close_oven_oneshot": { # use relative path
        "annotation_path": "assets/demo/lora_data/qa_pairs.jsonl",
        "data_path": "assets/demo/lora_data/"
    }
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    if dataset_names[0] == '*':
        dataset_names = list(data_dict.keys())
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name) # remove sampling rate suffix
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
