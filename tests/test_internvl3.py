"""
CUDA_VISIBLE_DEVICES=4,5 python grounding/tests/test_internvl3.py


1 node       0.6m per day, at least 4m grounding data per week
comparison qwen72b_once: 2.4m per day   

AgiBotAlpha  6.2m
AgiBotBeta   16.5m
rh20t        0.32m
robomind     0.28m
bc_z         4.2m
bridge_v2    2m
robot_net    2.5m

real(ours)   0.37m
"""

import json
import time
import os
import ast
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any


from core.utils.common import (
    json_response_to_bboxes,
    draw_bboxes,
    batch_infer_pipeline,
    prepare_grounding_questions, 
    prepare_grounding_questions_with_target_objects,
    prepare_multiturn_grounding_questions,
)
from grounding.utils.qwenvl_utils import prepare_target_objects_questions
from core.models.internvl import (
    batch_generate_with_lmd,
    batch_multiturn_generate_with_lmd,
)
from core.models.qwen import (batch_generate)

TEST_NAME = 'ivl35_2step'
VLM_PATH = os.getenv('VLM_PATH_INTERN_INFER')
TENSOR_PARALLEL_SIZE = 2

# logging.basicConfig(
#     level=logging.INFO,  # 设置最低级别
#     format="%(asctime)s - %(levelname)s - %(message)s"  # 可选，输出更清晰
# )

def test_a_path(
    test_path: str,
    dump_grounded_images: bool = True,
) -> int:
    desc_json_path = test_path + "sub_tasks.json"
    with open(desc_json_path, 'r') as f:
        desc_json = json.load(f)
    
    images = []
    task_descriptions = []
    for k, v in desc_json.items():
        img_path = test_path + k
        task_desc = v['sub_task']
        pil_img = Image.open(img_path)
        images.append(pil_img)
        task_descriptions.append(task_desc)

    timecost = time.time()
    if 'twice' in TEST_NAME:
        target_objects_questions = prepare_target_objects_questions(task_descriptions)
        target_objects = [
            ast.literal_eval(x) for x in batch_generate(target_objects_questions, model_path=os.getenv('LLM_PATH_QWEN_INFER'), max_tokens=40)
        ]
        images, questions = prepare_grounding_questions_with_target_objects(images, target_objects)
    elif 'once' in TEST_NAME:
        images, questions = prepare_grounding_questions(images, task_descriptions)
    elif '2step' in TEST_NAME:
        images, questions = prepare_multiturn_grounding_questions(images, task_descriptions)
    else:
        raise ValueError("TEST_NAME must contain any of 'twice', 'once' or '2step' to determine the grounding method.")
    
    _bi_func = batch_generate_with_lmd
    if '2step' in TEST_NAME:
        _bi_func = batch_multiturn_generate_with_lmd

    json_responses = batch_infer_pipeline(
        images, questions, VLM_PATH,
        batch_infer_func=_bi_func,
        tp=TENSOR_PARALLEL_SIZE,
        use_tqdm=True,
    )

    timecost = time.time() - timecost
    print(f"Time cost for {TEST_NAME} is {timecost:.2f} seconds.")

    if dump_grounded_images:
        output_images_path = test_path + TEST_NAME + "/"
        os.makedirs(output_images_path, exist_ok=True)
        for i, (image, json_response) in enumerate(zip(images, json_responses)):
            image_with_bbox = draw_bboxes(
                image,
                json_response_to_bboxes(json_response),
                show_object_name=True
            )
            output_image_path = f"{output_images_path}{i}.png"
            # try to concat reference image on the right side
            ref_image_possible_path = test_path + f"gemini/{i}.png"
            if os.path.exists(ref_image_possible_path):
                ref_image = Image.open(ref_image_possible_path).convert("RGB")
                # resize ref_image to the same height as image_with_bbox
                ref_image = ref_image.resize((int(ref_image.width * image_with_bbox.height / ref_image.height), image_with_bbox.height))
                new_width = image_with_bbox.width + ref_image.width
                new_image = Image.new('RGB', (new_width, image_with_bbox.height))
                new_image.paste(image_with_bbox, (0, 0))
                new_image.paste(ref_image, (image_with_bbox.width, 0))
                image_with_bbox = new_image
            # task_desc = task_descriptions[i]
            # # draw task_desc on the bottom of image_with_bbox
            # draw = Image.Draw.Draw(image_with_bbox)
            # font = Image.Font.load_default()
            # text_position = (10, image_with_bbox.height - 20)
            # draw.text(text_position, task_desc, fill=(255, 0, 0), font=font)
            image_with_bbox.save(output_image_path)
            # print(f"Saved result image to {output_image_path}")
    
    return timecost


# def test_speed(test_path: str) -> None:
#     # warmup
#     rough_timecost = test_a_path(test_path, dump_grounded_images=False)
    
#     total_timecost = 0
#     total_runs = 10
#     for i in range(total_runs):
#         timecost = test_a_path(test_path, dump_grounded_images=False)
#         total_timecost += timecost
#         logging.info(f"Test {i+1} time cost: {timecost:.2f} seconds.")
#     average_timecost = total_timecost / total_runs
#     logging.info(f"Average time cost for {total_runs} runs: {average_timecost:.2f} seconds.")
#     logging.info(f"Estimated warmup overhead: {(rough_timecost - average_timecost):.2f} seconds.")


def main(
    test_paths: List[str],
):
    for test_path in test_paths:
        test_a_path(test_path)
    # test_speed(test_paths[0])



if __name__ == "__main__":
    main(test_paths=[
        "grounding/generate/badcases/",
        "grounding/generate/randomcases/", 
        "grounding/generate/agicases/",
        "grounding/generate/egocases/",
        "grounding/generate/oxecases/",
    ])

"""
78-2step-1retry (default)
WARNING:root:Batch infer pipeline: 0 failed out of 11, rate: 0.00%
Time cost for ivl78b_2step is 256.28 seconds.
WARNING:root:Batch infer pipeline: 0 failed out of 30, rate: 0.00%
Time cost for ivl78b_2step is 6.67 seconds.
WARNING:root:Batch infer pipeline: 0 failed out of 30, rate: 0.00%
Time cost for ivl78b_2step is 40.89 seconds.
WARNING:root:Batch infer pipeline: 5 failed out of 30, rate: 16.67%
Time cost for ivl78b_2step is 7.69 seconds.
WARNING:root:Batch infer pipeline: 4 failed out of 60, rate: 6.67%
Time cost for ivl78b_2step is 13.56 seconds.
"""


"""
78-2step-1retry
WARNING:root:Batch infer pipeline: 0 failed out of 11, rate: 0.00%
Time cost for ivl78b_2step is 118.78 seconds.
WARNING:root:Batch infer pipeline: 0 failed out of 30, rate: 0.00%
Time cost for ivl78b_2step is 6.67 seconds.
WARNING:root:Batch infer pipeline: 5 failed out of 30, rate: 16.67%
Time cost for ivl78b_2step is 7.20 seconds.
WARNING:root:Batch infer pipeline: 4 failed out of 60, rate: 6.67%
Time cost for ivl78b_2step is 13.73 seconds.

78-2step-2retry
WARNING:root:Batch infer pipeline: 0 failed out of 11, rate: 0.00%
Time cost for ivl78b_2step is 121.00 seconds.
WARNING:root:Batch infer pipeline: 0 failed out of 30, rate: 0.00%
Time cost for ivl78b_2step is 7.98 seconds.
WARNING:root:Batch infer pipeline: 5 failed out of 30, rate: 16.67%
Time cost for ivl78b_2step is 8.23 seconds.
WARNING:root:Batch infer pipeline: 3 failed out of 60, rate: 5.00%
Time cost for ivl78b_2step is 14.76 seconds.

78-2step-3retry
WARNING:root:Batch infer pipeline: 0 failed out of 11, rate: 0.00%
Time cost for ivl78b_2step is 112.97 seconds.
WARNING:root:Batch infer pipeline: 0 failed out of 30, rate: 0.00%
Time cost for ivl78b_2step is 8.00 seconds.
WARNING:root:Batch infer pipeline: 5 failed out of 30, rate: 16.67%
Time cost for ivl78b_2step is 9.33 seconds.
WARNING:root:Batch infer pipeline: 4 failed out of 60, rate: 6.67%
Time cost for ivl78b_2step is 16.99 seconds.

38-2step-2retry
WARNING:root:Batch infer pipeline: 2 failed out of 11, rate: 18.18%
Time cost for ivl38b_2step is 151.74 seconds.
WARNING:root:Batch infer pipeline: 14 failed out of 30, rate: 46.67%
Time cost for ivl38b_2step is 10.55 seconds.
WARNING:root:Batch infer pipeline: 8 failed out of 30, rate: 26.67%
Time cost for ivl38b_2step is 7.64 seconds.
WARNING:root:Batch infer pipeline: 12 failed out of 60, rate: 20.00%
Time cost for ivl38b_2step is 19.70 seconds.
"""