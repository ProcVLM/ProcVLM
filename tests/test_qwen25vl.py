"""
CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python grounding/tests/test_qwen25vl.py
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
from core.models.qwenvl import (
    batch_generate_with_vllm,
    batch_multiturn_generate_with_vllm,
)
from core.models.qwen import batch_generate

TEST_NAME = 'qwen72b_once'
VLM_PATH = os.getenv('VLM_PATH_QWEN_25_INFER')

def test_a_path(
    test_path: str,
) -> List[Dict[str, Any]]:
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
        image, questions = prepare_grounding_questions(images, task_descriptions)
    elif '2step' in TEST_NAME:
        image, questions = prepare_multiturn_grounding_questions(images, task_descriptions)
    else:
        raise ValueError("TEST_NAME must contain 'twice', 'once' or '2step' to determine the grounding method.")
    
    bi_func = batch_generate_with_vllm 
    if '2step' in TEST_NAME:
        bi_func = batch_multiturn_generate_with_vllm

    json_responses = batch_infer_pipeline(
        images, questions, VLM_PATH,
        batch_infer_func=bi_func
    )
        
    timecost = time.time() - timecost
    print(f"Time cost for {TEST_NAME} is {timecost:.2f} seconds.")

    output_images_path = test_path + TEST_NAME + "/"
    os.makedirs(output_images_path, exist_ok=True)
    for i, (image, json_response) in enumerate(zip(images, json_responses)):
        image_with_bbox = draw_bboxes(
            image,
            json_response_to_bboxes(json_response),
            show_object_name=True
        )
        output_image_path = f"{output_images_path}{i}.jpg"
        # try to concat reference image on the right side
        ref_image_possible_path = test_path + f"ivl78b_2step/{i}.png"
        if os.path.exists(ref_image_possible_path):
            ref_image = Image.open(ref_image_possible_path).convert("RGB")
            # resize ref_image to the same height as image_with_bbox
            ref_image = ref_image.resize((int(ref_image.width * image_with_bbox.height / ref_image.height), image_with_bbox.height))
            new_width = image_with_bbox.width + ref_image.width
            new_image = Image.new('RGB', (new_width, image_with_bbox.height))
            new_image.paste(image_with_bbox, (0, 0))
            new_image.paste(ref_image, (image_with_bbox.width, 0))
            image_with_bbox = new_image
        image_with_bbox.save(output_image_path)
        # print(f"Saved result image to {output_image_path}")



def main(
    test_paths: List[str],
):
    for test_path in test_paths:
        test_a_path(test_path)



if __name__ == "__main__":
    main(test_paths=[
        "grouning/generate/badcases/",

        "grouning/generate/randomcases/", 
        # Time cost for qwen72b_once is 22.87 seconds.
        
        "grouning/generate/agicases/",
        
        "grouning/generate/egocases/",

        "grouning/generate/oxecases/",
    ])