import requests
import io, re, os
import base64
import json
import logging
import bisect
import numbers
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Dict, Any, Tuple
from core.utils.common import (
    improved_json_parser,
    check_normalized,
    normalize_bbox,
    filter_robot_from_json_response,
    pack_timestamp_image_content_for_qwen3,
    __multi_turn_template_turn1,
    __multi_turn_template_turn2,
)

load_dotenv()

### Question Templates ###
__single_turn_template = (
    "Locate all objects in the image that is used in this task."
    "Do not include the robotic arm itself.\n"
    "Task: {task_desc}.\n"
    "output its bbox coordinates using JSON format. No extra text.\n"
    'Output Format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}]'
)

### API Helpers ###
def general_api_query(
    mm_input: Image.Image | Any, 
    question: str, 
    base_url: str, 
    model_name: str,
    api_key_namespace: str = "API_KEY",
) -> str:
    if isinstance(mm_input, Image.Image):
        raise NotImplementedError("Image input not supported in this function.")
    else:
        # video: List[Tuple[int, Image.Image]]
        content = pack_timestamp_image_content_for_qwen3(mm_input, question)
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
    client = OpenAI(
        api_key=os.getenv(api_key_namespace),
        base_url=base_url
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return completion.choices[0].message.content

def api_base64_image(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    prefix = "data:image/png;base64,"
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    b64_with_prefix = prefix + b64_str
    return b64_with_prefix

def api_query(image: Image.Image, question: str, url: str, max_tokens: int=4096, temperature: float=1.0) -> Dict:
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    # {"type": "image_url", "image_url": {"url": api_base64_image(image)}}
                ] + [
                    {"type": "image_url", "image_url": {"url": api_base64_image(img)}}
                    for img in image
                ]
            }
        ],
        "top_p": 1.0,
        "temperature": temperature,
        "repetition_penalty": 1.0,
        "top_k": -1,
        "n": 1,
        "stop_token_ids": [151329, 151336, 151338],
        "skip_special_tokens":False,
        "include_stop_str_in_output":True,
        "max_tokens": max_tokens
    }
    headers = {'Content-Type': 'application/json'}
    session = requests.Session()
    retry_strategy = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"] # 确保POST在重试方法列表中
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    try:
        response = session.post(url, headers=headers, json=data, verify=False, timeout=10)
        response.raise_for_status()
        output = response.json()
        generated_text = output.get("choices", [{}])[0].get("message", {}).get("content", "")
        # remove between '<think>' and '</think>'
        _think_content = re.search(r'<think>.*?</think>', generated_text, flags=re.DOTALL)
        if _think_content:
            _think_content = _think_content.group(0)
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
        # remove '<|user|>'
        generated_text = re.sub(r'<\|user\|>', '', generated_text).strip()
        # remove '\n' at the beginning and end
        generated_text = generated_text.strip('\n')
        # print(f"Generated Text: {generated_text}")
        return generated_text
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return 'None'
    except json.JSONDecodeError as e:
        logging.error(f"Could not decode JSON response: {e}")
        return 'None'
    except KeyError as e:
        logging.error(f"Expected key not found in response: {e}")
        return 'None'
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return 'None'


        # slice between '<|begin_of_box|>' and '<|end_of_box|>'

def api_query_offline(image_or_videobase64: Image.Image | List[Image.Image] | str, question: str, model_key: str) -> Dict:
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url="https://api.chatglm.cn/v1"
    )
    if isinstance(image_or_videobase64, str):
        stream = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "video_url", "video_url": {"url": image_or_videobase64}}
                    ]
                }
            ],
            model=model_key,
            stream=True,
            max_tokens=4096
        )
    else:
        if isinstance(image_or_videobase64, Image.Image):
            image_or_videobase64 = [image_or_videobase64]
        stream = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        # {"type": "image_url", "image_url": {"url": api_base64_image(image_or_videobase64)}}
                    ] + [
                        {"type": "image_url", "image_url": {"url": api_base64_image(img)}}
                        for img in image_or_videobase64
                    ]
                }
            ],
            model=model_key,
            stream=True,
            max_tokens=4096
        )
    # response = stream.choices[0].message.content
    response = ""
    thinking_response = ""
    for part in stream:
        if hasattr(part.choices[0].delta, "content"):
            response += part.choices[0].delta.content
        if hasattr(part.choices[0].delta, "reasoning_content"):
            thinking_response += part.choices[0].delta.reasoning_content
    final_answer = f"<think>{thinking_response}</think>{response}"
    return response

### Infer / Batch Infer Helpers ###
def __single_turn(
    image: Image.Image, task_desc: str, url: str,
    question_template: str = __single_turn_template
) -> List[str]:
    return api_query(image, question_template.format(task_desc=task_desc), url)
    
def __multi_turn(
    image: Image.Image, task_desc: str, url: str,
    question_template_turn1: str = __multi_turn_template_turn1,
    question_template_turn2: str = __multi_turn_template_turn2,
) -> List[str]:
    question_turn1 = question_template_turn1.format(task_desc=task_desc)
    answer_turn1 = api_query(image, question_turn1, url)
    if not answer_turn1:
        logging.warning(f"Turn 1 failed for task: {task_desc}. Using fallback.")
        answer_turn1 = "object"
    question_turn2 = question_template_turn2.replace("<list>", answer_turn1)
    answer_turn2 = api_query(image, question_turn2, url)
    return answer_turn1, answer_turn2

def batch_generate(
    images: List[Image.Image] | str, task_descriptions: List[str], url: str,
    question_template: str = __single_turn_template,
    max_workers: int = 32,
    **kwargs,
) -> List[str]:
    """
    Run api_query(image, question, url) for each (image, description) concurrently.
    Results are returned in the same order as inputs.
    """
    questions = [question_template.format(task_desc=td) for td in task_descriptions]

    async def _run():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, api_query_offline, img, q, url, **kwargs)
                for img, q in zip(images, questions)
            ]
            return await asyncio.gather(*tasks)  # 保持顺序

    return asyncio.run(_run())


### Processing Model Outputs Helpers ###
def generated_texts_to_json_responses(
    texts: List[str],
    images: List[Image.Image]
):
    all_json_responses = []
    failed_indices = []
    for text_response, image in zip(texts, images):
        try:
            # slice between '<|begin_of_box|>' and '<|end_of_box|>'
            clean_text = text_response
            match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', clean_text, flags=re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
            # parse json
            raw_json_response = improved_json_parser(clean_text)
            if check_normalized(raw_json_response):
                json_response = raw_json_response
            else:
                json_response = normalize_bbox(raw_json_response)
            json_response = filter_robot_from_json_response(json_response)
        except (json.JSONDecodeError, ValueError) as e:
            # fall back to find_boxes_all
            boxes = find_boxes_all(text_response)
            raw_json_response = []
            for box in boxes:
                while isinstance(box, list) and len(box) == 1:
                    box = box[0]
                if isinstance(box, list) and len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box):
                    raw_json_response.append({"bbox_2d": box, "label": "object"})
            if check_normalized(raw_json_response):
                json_response = raw_json_response
            else:
                json_response = normalize_bbox(raw_json_response)
            json_response = filter_robot_from_json_response(json_response)
            logging.warning(f"Failed to parse JSON from {text_response}. Used fallback method. Result: {json_response}")
        all_json_responses.append(json_response)
    return all_json_responses, failed_indices


### bbox Helpers ###
from dataclasses import dataclass
from enum import Enum, auto

class BracketsStyle(Enum):
    """Enum for different styles of brackets."""
    SQUARE = {'left': '[', 'right': ']'}  # []
    CURLY = {'left': '{', 'right': '}'}   # {}
    ANGLE = {'left': '<', 'right': '>'}   # <>
    PARENTHESES = {'left': '(', 'right': ')'}  # ()
    

class NestedStyle(Enum):
    """Enum for different styles of nested brackets."""
    DOUBLE = auto()   #[[]] or {{}}
    SINGLE = auto()  # [] or {}

def is_single_list(obj):
    return isinstance(obj, list) and all(not isinstance(i, list) for i in obj)

def is_nested_list(obj):
    return isinstance(obj, list) and all(isinstance(i, list) for i in obj)

def find_boxes(text, brackets_style=BracketsStyle.SQUARE, nested_style=NestedStyle.SINGLE, return_matches=False):
    """ Find boxes of each object in a sentence (e.g., `Two apples {{1, 2, 3, 4}, {5,6,7,8}} and an orange {{9,10,11,12}} in basket.`) into a list of boxes `[[[1,2,3,4],[5,6,7,8]], [[9,10,11,12]]]`.
    """
    LB, RB = brackets_style.value['left'], brackets_style.value['right']
    LB_p, RB_p = (f'\{LB}', f'\{RB}') if brackets_style in [BracketsStyle.SQUARE, BracketsStyle.CURLY, BracketsStyle.PARENTHESES] else (LB, RB)
    if nested_style == NestedStyle.DOUBLE:
        pattern = '(%s\s*%s[\d\.,\s\[\]]+%s\s*%s)' % (LB_p,LB_p, RB_p, RB_p)
    else:
        pattern = '(%s\s*[\d\.,\s]+\s*%s)' % (LB_p, RB_p)
    res_boxes, res_strs, res_spans = [], [], []
    for item in re.finditer(pattern, text):
        try:
            boxes_lst = item.group(1).replace(LB, '[').replace(RB, ']')
            boxes = eval(boxes_lst)
            if is_nested_list(boxes) and all([len(bbx)==4 for bbx in boxes]):
                boxes = [[_ if isinstance(_,numbers.Number) else eval(_) for _ in _b] for _b in boxes]
            elif is_single_list(boxes) and len(boxes) ==4:
                boxes = [_ if isinstance(_,numbers.Number) else eval(_) for _ in boxes]
            else:
                continue
            res_boxes.append(boxes)
            res_strs.append(item.group(1))
            res_spans.append(item.span())
        except:
            pass
    if return_matches:
        return res_boxes, res_strs, res_spans
    else:
        return res_boxes


def find_boxes_all(text):
    found, ps_l, ps_r = [], [], []
    for nested_style in NestedStyle: # Double first
        for brackets_style in BracketsStyle:
            for bbx, bbx_str, bbx_span in zip(*find_boxes(text, brackets_style, nested_style=nested_style, return_matches=True)):
                ls = bisect.bisect_right(ps_l, bbx_span[0])
                if ls >0 and bbx_span[1] <= ps_r[ls-1]:
                    continue # fall within existing span
                found.append((bbx, bbx_str, bbx_span))
                ps_l.insert(ls, bbx_span[0]); ps_r.insert(ls, bbx_span[1])
    # ordering
    found_sort = []
    for bbx_str_span in  sorted(found, key=lambda x: x[2][0]):
        found_sort.append(bbx_str_span[0])
    return found_sort


