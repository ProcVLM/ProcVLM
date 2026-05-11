import os
import json_repair
import random
import time
import logging
from pathlib import Path
from core.utils.common import append_jsonlines
from core.models.qwen import generate_with_vllm as generate, batch_generate_with_vllm as batch_generate
from typing import List, Optional, Union, Tuple, Dict, Any


model_path = os.getenv('LLM_PATH_QWEN_INFER')
tp = 2

# query llm to remove useless boxes
# useless objects:
# 1. background objects (wall, floor, table etc.)
# 2. objects have nothing to do with the task
# 3. office supplies (if the task is not about office), such as chair, computer, keyboard, mouse, monitor, phone, charger, power outlet, etc.

system_prompt = """You are an assistant that filters objects based on task relevance."""

user_prompt = """You are given an ACTION and a list of object descriptions. Find the useless objects that are unlikely to be interacted with during the ACTION.

Rules:
1. Background objects (e.g., wall, floor, table) are always useless, report them. Controlling person and other office workers are considered background and should be reported.
2. Office supplies (e.g., chair, computer, keyboard, mouse, monitor, phone, charger, power outlet) are generally from the experiment environment, so they are useless and should be reported, unless the ACTION is about using them.
3. If an object, based on its description, is unlikely to be interacted with during the ACTION, consider it useless and report it.
4. The robotic arm, gripper tool or end-effector are generally useless, unless the ACTION is about using them.
5. Only evaluate the objects listed in `OBJECTS`. Ignore objects not in the list.

Output requirements:
- Return a Python list of integer indices (0-based) corresponding to the useless objects.
- Indices must match the numbering in the `OBJECTS`.  
- If all objects are useful, return exactly: `[]`.  

Input:
ACTION: "{task_desc}"  
list of object descriptions: {label_list}  

Output:
The list of indices of useless objects. Do not return anything else besides the list."""

date = time.strftime("%Y%m%d", time.gmtime()) # make it more concise
tmp_output_path = f'tmp/bbox_unused_{date}.jsonl'
if not Path(tmp_output_path).parent.exists():
    Path(tmp_output_path).parent.mkdir(parents=True, exist_ok=True)

def generated_texts_to_list(gen_text: str, bound: range) -> List:
    try:
        lst = json_repair.loads(gen_text)
        if isinstance(lst, list):
            if not lst: # empty list
                return []
            if isinstance(lst[0], list):
                lst = lst[0] # sometimes llm returns [[1,2,3]]
            if all(isinstance(i, int) for i in lst) and all(i in bound for i in lst):
                return lst
            else:
                logging.error("Some indices in LLM output are out of range: %s", gen_text)
                return [x for x in lst if isinstance(x, int) and x in bound]
        else:
            logging.error("LLM output is not a list of integers: %s", gen_text)
            return []
    except Exception as e:
        logging.error("Failed to parse LLM output: %s; error: %s", gen_text, str(e))
        return []

def llm_filter(bboxes, task_desc) -> List:
    """
    Use LLM to filter out useless boxes.
    Args:
        bboxes: list of dict, each dict has keys: 'label', 'bbox_2d'
        task_desc: str, description of the task
    Returns:
        filtered_bboxes: list of dict, filtered bboxes
    """
    if len(bboxes) < 1:
        return bboxes, []
    label_list = [bbox['label'] for bbox in bboxes]
    label_str = '\n'.join([f'{idx}. "{label}"' for idx, label in enumerate(label_list)])
    prompt = user_prompt.format(task_desc=task_desc, label_list=label_str)
    gen_text = generate(
        prompt, 
        model_path=model_path, 
        system_prompt=system_prompt, 
        max_tokens=32768, 
        temperature=0,
        tp=tp,
    )
    unused_indices = generated_texts_to_list(gen_text, range(len(bboxes)))
    filtered_bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx not in unused_indices]
    deprecated_bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx in unused_indices]  
    if random.random() < 1:
        jd = {
            "task": task_desc,
            "unused": [bbox['label'] for idx, bbox in enumerate(bboxes) if idx in unused_indices],
        }
        append_jsonlines(jd, tmp_output_path)
    return filtered_bboxes, deprecated_bboxes

def llm_filter_batch(ins: List[Tuple[List, str]]) -> List[Tuple[List, List]]:
    batch_ids = []
    batch_input = []
    outs = [None] * len(ins)
    for i, (bboxes, task_desc) in enumerate(ins):
        if len(bboxes) < 1:
            outs[i] = (bboxes, [])
            continue
        label_list = [bbox['label'] for bbox in bboxes]
        id_range_str = f"[0, {len(bboxes)-1}]"
        label_str = '\n'.join([f'{idx}. "{label}"' for idx, label in enumerate(label_list)])
        prompt = user_prompt.format(task_desc=task_desc, label_list=label_str, id_range_str=id_range_str)
        batch_ids.append(i)
        batch_input.append(prompt)
    batch_output = []
    if len(batch_input) > 0:
        batch_output = batch_generate(
            batch_input, 
            model_path=model_path, 
            system_prompt=system_prompt, 
            max_tokens=32768, 
            temperature=0,
            tp=tp,
        )
    for bid, gen_text in zip(batch_ids, batch_output):
        bboxes, task_desc = ins[bid]
        unused_indices = generated_texts_to_list(gen_text, range(len(bboxes)))
        filtered_bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx not in unused_indices]
        deprecated_bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx in unused_indices]
        if random.random() < 1:
            jd = {
                "task": task_desc,
                "unused": [bbox['label'] for idx, bbox in enumerate(bboxes) if idx in unused_indices],
            }
            append_jsonlines(jd, tmp_output_path)
        outs[bid] = (filtered_bboxes, deprecated_bboxes)
    return outs
    
def rulebased_filter(bboxes, task_desc) -> List:
    """
    Use rule-based method to filter out useless boxes.
    Args:
        bboxes: list of dict, each dict has keys: 'label', 'bbox_2d'
        task_desc: str, description of the task
    Returns:
        filtered_bboxes: list of dict, filtered bboxes
    """
    if len(bboxes) < 1:
        return bboxes, []
    _useless_keywords = [
        "computer", "keyboard", "mouse", "monitor", "charger", "power outlet", "laptop", "person", "power strip", "powerbank", "powerstrip", "office", "sofa"
    ]
    _useless_label = [
        "chair", "table", "desk", "desktop",
    ]
    task_desc_lower = task_desc.lower()
    filtered_bboxes = []
    deprecated_bboxes = []
    for bbox in bboxes:
        label_lower = bbox['label'].lower()
        useless_keywords = [kw for kw in _useless_keywords if kw not in task_desc_lower]
        useless_label = [ul for ul in _useless_label if ul not in task_desc_lower]
        if any(kw in label_lower for kw in useless_keywords) or any(ul == label_lower for ul in useless_label):
            deprecated_bboxes.append(bbox)
        else:
            filtered_bboxes.append(bbox)
    return filtered_bboxes, deprecated_bboxes

def rulebased_filter_batch(ins: List[Tuple[List, str]]) -> List[Tuple[List, List]]:
    outs = []
    for bboxes, task_desc in ins:
        filtered_bboxes, deprecated_bboxes = rulebased_filter(bboxes, task_desc)
        if len(deprecated_bboxes) > 0 and random.random() < 1:
            jd = {
                "task": task_desc,
                "unused": [bbox['label'] for bbox in deprecated_bboxes],
            }
            append_jsonlines(jd, tmp_output_path)
        outs.append((filtered_bboxes, deprecated_bboxes))
    return outs