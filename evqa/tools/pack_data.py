"""
python evqa/tools/pack_data.py
"""

import json
import torch
import binpacking
from pathlib import Path
from tqdm import tqdm
import time

def tensor_to_list(item):
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            item[key] = value.tolist()
    return item


def read_data(file_path):
    """Read JSON or JSONL file"""
    if file_path.endswith(('.json', '.jsonl')):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError('Please provide a .json or .jsonl file')


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=4)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item) + '\n')


def pack_data(data_list, pack_length):
    # Extract the length of each data item
    lengths = [data["num_tokens"] for data in data_list]
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)),  # Explicitly convert to list
        pack_length,
        weight_pos=1
    )
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            new_data = data_list[index].copy()
            # new_data.pop("num_tokens", None)
            group_data.append(tensor_to_list(new_data))
        packed_data.append(group_data)
    return packed_data

from evqa.data import data_dict as datasets

for dataset_name, config in tqdm(datasets.items(), desc="Packing Datasets"):
    if dataset_name in ['dummy_short']:
        continue
    if not 'train' in dataset_name.lower():
        print(f'Skipping non-training dataset: {dataset_name}')
        continue
    
    print(f'\n--- Processing dataset: {dataset_name} ---')
    annotation_path = config['annotation_path']
    if annotation_path.endswith('_pack.json'):
        print(f'Packed file already exists at: {annotation_path}, skipping packing.')
        continue

    pack_output_path = annotation_path.replace('.jsonl', '_pack.json')
    if Path(pack_output_path).exists():
        print(f'Packed file already exists at: {pack_output_path}, skipping packing.')
        continue

    data_with_tokens = read_data(annotation_path)

    # Assume the packing length is 4096
    if '8k' in dataset_name.lower():
        pack_length = 8000
    elif '16k' in dataset_name.lower():
        pack_length = 16000
    else:
        raise NotImplementedError(f'Please specify pack length for dataset: {dataset_name}')
    
    # Define the batch size
    batch_size = 4096
    all_packed_results = []

    # Record the start time of binpacking
    start_time = time.time()
    for i in tqdm(range(0, len(data_with_tokens), batch_size), desc="Binpacking Batches"):
        batch_data = data_with_tokens[i: i + batch_size]
        batch_packed_result = pack_data(batch_data, pack_length)
        all_packed_results.extend(batch_packed_result)
    # Record the end time of binpacking
    end_time = time.time()

    # Calculate the time spent on binpacking
    binpack_time = end_time - start_time
    print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

    # Save the packed results as a JSON file
    with open(pack_output_path, 'w', encoding='utf-8') as file:
        json.dump(all_packed_results, file, indent=2)
    print(f"Packed results saved to: {pack_output_path}")