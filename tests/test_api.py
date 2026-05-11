"""
export http_proxy=http://httpproxy.glm.ai:3128
"""

import json
import time
import os
from dotenv import load_dotenv
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any


from core.utils.common import (
    _pil_to_base64,
    json_response_to_bboxes,
    draw_bboxes,
)
from core.backends.api import (
    generated_texts_to_json_responses,
    __multi_turn,
    __single_turn,
)

TEST_NAME = 'glm45v_once'
load_dotenv()
API_URL = os.getenv('API_URL')

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
    
    if '2step' in TEST_NAME:
        infer_func = __multi_turn
    else:
        infer_func = __single_turn
    json_responses = []
    for img, task_desc in zip(images, task_descriptions):
        generated_text = infer_func(img, task_desc, API_URL)
        if infer_func == __multi_turn:
            _, generated_text = generated_text
        json_response, _ = generated_texts_to_json_responses([generated_text], [img])
        json_responses.extend(json_response)

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


def main(
    test_paths: List[str],
):
    for test_path in test_paths:
        test_a_path(test_path)


if __name__ == "__main__":
    main(test_paths=[
        "grouning/generate/badcases/",
        "grouning/generate/randomcases/", 
        "grouning/generate/agicases/",
        "grouning/generate/egocases/",
        "grouning/generate/oxecases/",
    ])
