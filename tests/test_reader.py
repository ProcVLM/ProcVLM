"""
python grounding/tests/test_reader.py
"""

import time
import random
from pathlib import Path
from PIL import Image
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP
from core.utils.common import hash_image

datasets = [
    '/pretrain_data/oxe_lerobot/dobbe/',
    '/pretrain_data/rh20t/RH20T_cfg1_lerobot',
    '/pretrain_data/droid_rlds/droid_1.0.1_lerobot',
]
args = [
    {},
    {
        'image_resize': (504, 504),
    },
    {
        'load_all_camera_keys': True,
        'return_record_meta': True,
        'image_resize': (504, 504),
    }
]

def test_fast_lerobot_vl_reader(dr: str, args: dict):
    start_time = time.time()
    dn = Path(dr).name
    dataset = FastLerobotVLReader(
        root=dr,
        ego_name=EGO_CAMERA_MAP.get(dn, None),
        exo_name=EXO_CAMERA_MAP.get(dn, None),
        **args
    )
    print(f"Testing dataset at {dr}, dataset has {len(dataset)} samples.")
    print(f"Time taken to initialize dataset: {time.time() - start_time:.2f} seconds")
    print(f"Camera keys: {dataset.camera_keys}")

    # sequential read test
    upbound = min(50000, len(dataset))
    start_time = time.time()
    for i, data in enumerate(dataset):
        if i >= upbound:
            break
        # hsh = hash_image(data['exo_image'])
        # hsh1 = hash_image(data['ego_image'])
        # all_hsh = (hsh, hsh1)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            throughput = (i + 1) / elapsed
            print(f"Sequentially read {i + 1}/{upbound} samples. Throughput: {throughput:.2f} samples/sec")
    total_time = time.time() - start_time
    total_throughput = upbound / total_time
    print(f"Sequential read of {upbound} samples took {total_time:.2f} seconds.")
    print(f"Average throughput: {total_throughput:.2f} samples/sec")

    # random access test (estimated 40x slower than sequential)
    # dataset.prefetch_num = 0
    # upbound = min(1000, len(dataset))
    # random_indices = random.sample(range(len(dataset)), upbound)
    # start_time = time.time()
    # for count, idx in enumerate(random_indices):
    #     _ = dataset[idx]
    #     if (count + 1) % 100 == 0:
    #         elapsed = time.time() - start_time
    #         throughput = (count + 1) / elapsed
    #         print(f"Randomly accessed {count + 1}/{upbound} samples. Throughput: {throughput:.2f} samples/sec")
    # total_time = time.time() - start_time
    # total_throughput = upbound / total_time
    # print(f"Random access of {upbound} samples took {total_time:.2f} seconds.")
    # print(f"Average random access throughput: {total_throughput:.2f} samples/sec")

    # verify that all camera keys are present and valid images
    # start_time = time.time()
    # for i, data in enumerate(dataset):
    #     for cam_key in dataset.camera_keys:
    #         assert cam_key in data, f"Camera key '{cam_key}' missing in data sample."
    #         img_data = data[cam_key]
    #         assert isinstance(img_data, Image.Image), f"Data for camera key '{cam_key}' is not a PIL Image."
    #     if i >= upbound:
    #         break
    # print(f"Integrity check passed for {upbound} samples in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    for dr, args in zip(datasets, args):
        test_fast_lerobot_vl_reader(dr, args)