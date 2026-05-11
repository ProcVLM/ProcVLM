import time
from pathlib import Path

from core.data.reader import FastLerobotVLReader
from core.utils.common import (
    draw_bboxes,
    json_response_to_bboxes,
    hash_image,
    check_data_sim,
    load_jsonlines,
)
# thershold = 0.995
dr = "/pretrain_data/oxe_lerobot/language_table"
save_dir = Path('/pretrain_data/sim_sampled') / Path(dr).name
if save_dir.exists():
    for f in save_dir.iterdir():
        if f.is_file():
            f.unlink()
save_dir.mkdir(parents=True, exist_ok=True)

def match(obj1, obj2, eps=1e-2):
    if len(obj1) != len(obj2):
        return False
    boxes1 = sorted([tuple(b['bbox_2d']) for b in obj1])
    boxes2 = sorted([tuple(b['bbox_2d']) for b in obj2])
    for b1, b2 in zip(boxes1, boxes2):
        for v1, v2 in zip(b1, b2):
            if abs(v1 - v2) > eps:
                return False
    return True

if __name__ == "__main__":
    start_time = time.time()
    dataset = FastLerobotVLReader(
        root=dr,
        load_all_camera_keys=False,
        return_record_meta=True,
    )
    bbox_jsl_dir = Path('/pretrain_data/bbox_jsonl') / Path(dr).name
    print(f"Testing dataset at {dr}, dataset has {len(dataset)} samples.")
    print(f"Time taken to initialize dataset: {time.time() - start_time:.2f} seconds")
    print(f"Camera keys: {dataset.camera_keys}")

    upbound = min(10000, len(dataset))
    start_time = time.time()
    last_id = None
    skip = 0
    perfect_skip = 0
    possible_ep_jsonl = ''
    bbox_dict = {}
    for i, data in enumerate(dataset[:upbound]):
        frame_id = data['frame_index']
        ep_stem = data['episode_name']
        task = data['task']
        
        # --- cache for .bbox.jsonl ---
        pej = bbox_jsl_dir / f"{ep_stem}.bbox.jsonl"
        if pej != possible_ep_jsonl:
            possible_ep_jsonl = pej
            if pej.exists():
                _jf = load_jsonlines(pej)
                bbox_dict = {x['img_hash']: x['bboxes'] for x in _jf}
            else:
                bbox_dict = {}
        
        ckey = 'exo_image'
        if last_id and check_data_sim(data, dataset[last_id], img_key=ckey):
            skip += 1
            # if i < 200:
            #     jr = bbox_dict[hash_image(data[ckey])]
            #     image_with_bbox = draw_bboxes(
            #         data[ckey],
            #         json_response_to_bboxes(jr),
            #         show_object_name=True
            #     )
            #     image_with_bbox.save(save_dir / f"{ep_stem}_{frame_id:04d}_skipped.png")
            #     dataset[i][ckey].save(save_dir / f"{ep_stem}_{frame_id:04d}_skipped.png")
            # obj = bbox_dict[hash_image(data[ckey])]
            # lobj = bbox_dict[hash_image(dataset[last_id][ckey])]
            # if match(obj, lobj):
            #     perfect_skip += 1
        else:
            last_id = i
            # if i < 200:
            #     dataset[i][ckey].save(save_dir / f"{ep_stem}_{frame_id:04d}.png")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            throughput = (i + 1) / elapsed
            print(f"Processed {i + 1}/{upbound} samples. Throughput: {throughput:.2f} samples/sec")
    total_time = time.time() - start_time
    total_throughput = upbound / total_time
    print(f"Finished processing {upbound} samples in {total_time:.2f} seconds. Average throughput: {total_throughput:.2f} samples/sec") 

    print(f"finally sampled {upbound - skip} images, skip rate: {skip / upbound:.2%}, perfect skip rate: {perfect_skip / skip:.2%}")