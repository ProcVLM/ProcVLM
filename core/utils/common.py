import logging
import json, jsonlines
import io, re
import inspect
import base64
import random
import hashlib, zlib, imagehash
import numpy as np
from itertools import chain
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Union, Dict, Any, Tuple



### Question Templates ###
__single_turn_template = (
    "Task: {task_desc}.\n"
    "Locate the target object for the above task in the image. "
    "A target object should be the destination object that the robot is manipulating towards. It must be mentioned in the task description. "
    "For example, target objects for 'place the red block on the green block' is the green block. 'hang the cup on the rack' is the rack. 'press the button on the panel' is button on the panel. "
    "output its bbox coordinates using JSON format. No other text.\n"
    'Output Format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "name of the object"}}]'
)
__single_turn_template_no_task = (
    "Locate the objects in the image. "
    "Do not include the robotic arm itself."
    "output its bbox coordinates using JSON format. No other text.\n"
    'Output Format: [{"bbox_2d": [x1, y1, x2, y2], "label": "object"}]'
)
__single_turn_template_given_objs = (
    "Locate {objs} in the image. "
    "output its bbox coordinates using JSON format. No other text.\n"
    'Output Format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}]'
)

__multi_turn_template_turn1 = (
    "Which objects are used in the following task? "
    "Do not include the robotic arm itself."
    "Task: {task_desc}.\n"
    'Output Format: \n'
    '1. precise description of object1\n'
    '2. ...'
)
__multi_turn_template_turn2 = (
    "Locate and label the following objects in the image: \n"
    "<list>\n"
    "Do not include objects that are not in the list."
    "output its bbox coordinates using JSON format. No other text.\n"
    'Output Format: [{"bbox_2d": [x1, y1, x2, y2], "label": "object"}]'
)
__multi_turn_template_turn1_no_task = (
    "Which objects are available for manipulation in the image? "
    "Do not include the robotic arm itself.\n"
    'Output Format: \n'
    '1. precise description of object1\n'
    '2. ...'
)
__multi_turn_template_turn2_no_task = __multi_turn_template_turn2

def prepare_grounding_questions(
    images: List[Image.Image], task_descriptions: List[str],
    question_template_with_task: str = __single_turn_template,
    question_template_without_task: str = __single_turn_template_no_task,
) -> Union[List[Image.Image], List[str]]: 
    assert isinstance(images, list) or isinstance(images, Image.Image), \
        "Input images must be a list of PIL Image objects or a single PIL Image object."
    assert isinstance(task_descriptions, list) or isinstance(task_descriptions, str), \
        "Task descriptions must be a list of strings or a single string."
    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions]
    assert len(images) == len(task_descriptions), \
        "The number of images must match the number of task descriptions."
    questions = []
    for task_desc in task_descriptions:
        if task_desc:
            question = question_template_with_task.format(task_desc=task_desc)
        else:
            question = question_template_without_task
        questions.append(question)
    return images, questions

def prepare_grounding_questions_with_target_objects(
    images: List[Image.Image], target_objects: List[List[str]],
    question_template_with_task: str = __single_turn_template_given_objs,
    question_template_without_task: str = __single_turn_template_no_task,
) -> Union[List[Image.Image], List[str]]:
    assert isinstance(images, list) or isinstance(images, Image.Image), \
        "Input images must be a list of PIL Image objects or a single PIL Image object."
    assert isinstance(target_objects, list) and all(isinstance(obj, list) for obj in target_objects), \
        "Target objects must be a list of lists, where each inner list contains object names."
    if isinstance(images, Image.Image):
        images = [images]
    assert len(images) == len(target_objects), \
        "The number of images must match the number of target objects."
    questions = []
    for objs in target_objects:
        if len(objs) > 0:
            # objs = [f"'{obj}'" for obj in objs]
            question = question_template_with_task.format(objs=", ".join(map(str, objs)))
        else:
            question = question_template_without_task
        questions.append(question)
    return images, questions

def prepare_multiturn_grounding_questions(
    images: List[Image.Image],
    task_descriptions: List[str],
    question_template_with_task: List[str] = [__multi_turn_template_turn1, __multi_turn_template_turn2],
    question_template_without_task: List[str] = [__multi_turn_template_turn1_no_task, __multi_turn_template_turn2_no_task],
) -> Union[List[Image.Image], List[List[str]]]:
    assert isinstance(images, list) or isinstance(images, Image.Image), \
        "Input images must be a list of PIL Image objects or a single PIL Image object."
    assert isinstance(task_descriptions, list) or isinstance(task_descriptions, str), \
        "Task descriptions must be a list of strings or a single string."
    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions]
    assert len(images) == len(task_descriptions), \
        "The number of images must match the number of task descriptions."
    questions = []
    for task_desc in task_descriptions:
        if task_desc:
            question = [
                question_template_with_task[0].format(task_desc=task_desc),
                question_template_with_task[1]
            ]
        else:
            question = question_template_without_task
        questions.append(question)
    return images, questions

def pack_timestamp_image_content_for_qwen3(video: List[Tuple[int, Image.Image]], prompt: str, use_base64: bool=True) -> List[Dict]:
    """
    Pack a list of (timestamp integer, PIL Image) into the required format for API input.

    Args:
        Frame list of a video, each frame is (timestamp number, PIL Image)
    
    Returns:
        A list of interleaved dictionaries like:
        {
            "type": "text",
            "text": "<7 seconds>"
        },
        {
            "type": "image",
            "image": <Image.Image object>
        },
    """
    if use_base64:
        content = []
        for ts, img in video:
            base64_image = _pil_to_base64(img)
            content.append({
                "type": "text",
                "text": f"<{ts}.0 seconds>"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt
        })
    else:
        interleaved_pairs_iterator = (
            ({"type": "text", "text": f"<{ts}.0 seconds>"}, {"type": "image", "image": img})
            for ts, img in video
        )
        content = list(chain.from_iterable(interleaved_pairs_iterator))
        content.append({"type": "text", "text": prompt})
    return content



### Processing Model Outputs Helpers ###
def generated_texts_to_json_responses(
    texts: List[str],
    images: List[Image.Image]
):
    all_json_responses = []
    failed_indices = []
    for text_response, image in zip(texts, images):
        try:
            raw_json_response = improved_json_parser(text_response)
            if check_normalized(raw_json_response):
                json_response = raw_json_response
            else:
                W, H = image.size
                json_response = normalize_bbox(raw_json_response)
            # json_response = filter_robot_from_json_response(json_response)
        except (json.JSONDecodeError, ValueError) as e:
            logging.debug(f"Failed to parse JSON from text: {text_response}\nError: {e}")
            json_response = []
            failed_indices.append(len(all_json_responses))
        all_json_responses.append(json_response)
    return all_json_responses, failed_indices

def filter_robot_from_json_response(json_response: List[Dict]):
    for obj in json_response:
        if 'gripper' in obj.get('label', '').lower():
            json_response.remove(obj)
    return json_response



### Grounding Pipeline Helpers ###
def batch_infer_pipeline(
    images: List[Image.Image],
    questions: List[Any],
    model_path: str,
    batch_infer_func: callable,
    process_text_response_func: callable = generated_texts_to_json_responses,
    max_tokens: int = 256,
    temperature: float = 0,
    max_retries: int = 1,
    disable_logging: bool = False,
    **kwargs,
) -> List[List[Dict]]: # returns list of json_response
    json_responses = [[] for _ in range(len(images))]
    visited = [False for _ in range(len(images))]
    sig = inspect.signature(batch_infer_func)
    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    supports_temperature = ("temperature" in params) or accepts_var_kwargs
    supports_sampling_kwargs = ("sampling_kwargs" in params) or accepts_var_kwargs
    for _ in range(max_retries):
        currently_unvisited_indices = [i for i, v in enumerate(visited) if not v]

        infer_kwargs = {
            "model_path": model_path,
            "max_tokens": max_tokens,
        }
        infer_kwargs.update(kwargs)
        if supports_temperature:
            infer_kwargs["temperature"] = temperature
        elif supports_sampling_kwargs:
            sampling_kwargs = dict(infer_kwargs.pop("sampling_kwargs", {}) or {})
            sampling_kwargs.setdefault("temperature", temperature)
            infer_kwargs["sampling_kwargs"] = sampling_kwargs

        generated_texts = batch_infer_func(
            [images[i] for i in currently_unvisited_indices],
            [questions[i] for i in currently_unvisited_indices],
            **infer_kwargs,
        )
        _json_responses, _failed_indices = process_text_response_func(generated_texts, images)
        _failed_indices = [currently_unvisited_indices[i] for i in _failed_indices]

        for local_i, _jr in enumerate(_json_responses):
            global_i = currently_unvisited_indices[local_i]
            if global_i not in _failed_indices:
                json_responses[global_i] = _jr
                visited[global_i] = True

        if all(visited):
            break

    if not disable_logging:
        logging.warning(f"Batch infer pipeline: {(len(visited)-sum(visited))} failed out of {len(visited)}, rate: {(len(visited)-sum(visited))/len(visited):.2%}")
        logging.info(f"binery map: {''.join(['1' if v else '0' for v in visited])}")
    return json_responses

def batch_multiturn_generate(
    images: List[Image.Image],
    questions: List[List[str]],  # 每个样本的问题列表，例如 [["desc", "Q1", "Q2"], ["desc", "Q1"]]
    model_path: str,
    batch_generate_func: callable,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    __debug_mode: bool = False,
    **kwargs,
):
    q1_list = [q[0] for q in questions]
    q1_answers = batch_generate_func(
        images,
        q1_list,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    q2_list = []
    for (generated_text, q) in zip(q1_answers, questions):
        if __debug_mode: print(10*"=" + "Generated object descriptions" + 10*"=" + "\n" + generated_text)
        q2 = q[1].replace("<list>", generated_text.strip())
        if __debug_mode: print(10*"=" + "Final Question" + 10*"=" + "\n" + q2)
        q2_list.append(q2)
    
    all_responses = batch_generate_func(
        images,
        q2_list,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return all_responses

def fast_batch_infer_pipeline(
    images: List[Image.Image],
    texts: List[str],
    model_path: str,
    batch_infer_func: callable,
    process_text_response_func: callable = generated_texts_to_json_responses,
    **kwargs
) -> List[List[Dict]]: # returns list of json_response
    """
    This function is the Non-Retry version of batch_infer_pipeline.
    """
    generated_texts = batch_infer_func(
        images,
        texts,
        model_path=model_path,
        **kwargs
    )
    json_responses, _failed_indices = process_text_response_func(generated_texts, images)
    failed_binary_map = ''.join(['1' if i not in _failed_indices else '0' for i in range(len(images))])
    logging.debug(f"Fast batch infer pipeline: {len(_failed_indices)} failed out of {len(images)}, rate: {len(_failed_indices)/len(images):.2%}")
    logging.debug(f"binery map: {failed_binary_map}")
    return json_responses, failed_binary_map



### File I/O Helpers ###
def load_json(fpath: Path | str) -> Any:
    with open(fpath) as f:
        return json.load(f)

def load_jsonlines(fpath: Path | str) -> list[Any]:
    """Loads a JSON Lines file and returns a list of dictionaries."""
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)

def write_jsonlines(data: dict, fpath: Path | str) -> None:
    """Writes a list of dictionaries to a JSON Lines file."""
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)

def append_jsonlines(data: dict | List[dict], fpath: Path | str) -> None:
    """Append a single JSON line to a file."""
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    fpath.parent.mkdir(exist_ok=True, parents=True)
    if not isinstance(data, list):
        with jsonlines.open(fpath, "a") as writer:
            writer.write(data)
    else:
        with jsonlines.open(fpath, "a") as writer:
            writer.write_all(data)

def episode2num_token(ep_stem: str, num_tokens_factor: int=None) -> int:
    """Convert episode string like 'episode_000114' to a token number '11', which is episode index // 10, or modulo num_tokens_factor if provided."""
    match = re.search(r'episode_(\d+)', ep_stem)
    if match:
        if not num_tokens_factor:
            return int(match.group(1)) // 10
        return (int(match.group(1)) // 10) % num_tokens_factor
    else:
        raise ValueError(f"Invalid episode string format: {ep_stem}")



### Image and Video Processing Helpers ###
def hash_str(s: str) -> int:
    """
    Uniform hash using multiplicative folding to remove 2^k bias.
    """
    x = zlib.crc32(s.encode("utf-8")) & 0xffffffff
    # 64-bit expand and mix
    x ^= (x >> 16)
    x *= 0x9e3779b97f4a7c15  # golden ratio prime multiplier
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 32)
    return x

def hash_image(image: Image.Image, algo="sha256", max_len=40) -> str:
    # img = image.convert("RGB")
    data = image.tobytes() + str(image.size).encode()
    h = hashlib.new(algo)
    h.update(data)
    digest = h.hexdigest()
    return digest[:max_len]

def _pil_to_base64(image: Image.Image, format="PNG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images, which reports the similarity between two images.
    A score of 1.0 means identical images, while a score of 0.0 means completely different images.
    The images must be the same size.
    """
    from skimage.metrics import structural_similarity as ssim
    if img1.size != img2.size:
        return 1.0
    arr1 = np.asarray(img1.convert("L"))
    arr2 = np.asarray(img2.convert("L"))
    score, _ = ssim(arr1, arr2, full=True)
    return score

def image_phash_sim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute perceptual hash similarity between two images, which reports the similarity between two images.
    A score of 1.0 means identical images, while a score of 0.0 means completely different images.
    The images must be the same size.
    NOTE: Because phash uses discrete hamming distance, the similarity score can only take discrete values such as 1.0, 0.984375(63/64), 0.96875(62/64), etc.
    """
    if img1.size != img2.size:
        return 1.0
    h1 = imagehash.phash(img1)
    h2 = imagehash.phash(img2)
    return 1 - (h1 - h2) / len(h1.hash) ** 2

def check_data_sim(data: Dict, last_data: Dict, img_key: str, thershold: int=0.99, sim_func: callable=image_phash_sim) -> bool:
    """ 
    Check if two data samples are similar based on task, sub_task, and image similarity. 
    Return True if similar, False otherwise.
    For thershold, because hamming distance only takes discrete values, you can choose <0.984375 to allow 1-bit difference, <0.96875 to allow 2-bit difference, etc. The default 0.99 means only exact match is considered similar.
    """
    if data['task_description'] != last_data['task_description']:
        return False
    sim = sim_func(data[img_key], last_data[img_key])
    if sim > thershold:
        return True
    return False

def get_draw_object(
    img: Image.Image, 
    pos_for_append: str = None, 
    append_height: int = 0
) -> Tuple[Image.Image, Image.Image, ImageDraw.ImageDraw]:
    """
    Prepares an image for drawing by creating an RGBA version, handling canvas expansion, 
    and creating an overlay and a Draw object.

    Args:
        img (Image.Image): The base PIL Image object.
        pos_for_append (str, optional): If 'append-bottom', the canvas will be expanded.
        append_height (int, optional): The height to add to the canvas if expanding.

    Returns:
        Tuple[Image.Image, Image.Image, ImageDraw.ImageDraw]: 
            - The final RGBA base image (possibly expanded).
            - The transparent overlay image.
            - The ImageDraw object linked to the overlay.
    """
    W, H = img.size
    final_img = img

    # Handle canvas expansion for 'append-bottom'
    if pos_for_append == "append-bottom":
        # Create a new, larger canvas with a black background
        final_img = Image.new("RGBA", (W, H + append_height + 16), (0, 0, 0, 255))
        final_img.paste(img, (0, 0))

    final_img_rgba = final_img.convert("RGBA")
    overlay = Image.new("RGBA", final_img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    return final_img_rgba, overlay, draw

def draw_bboxes_on_overlay(
    draw: ImageDraw.ImageDraw, 
    size: Tuple[int, int], 
    bbox_list: List[Dict], 
    show_object_name: bool = False, 
    line_width: int = 3, 
    font: ImageFont.FreeTypeFont = None
):
    """
    Draws bounding boxes and their labels directly onto a given ImageDraw object.
    This function does not create images or handle compositing.
    """
    W, H = size
    if font is None and show_object_name:
        font = ImageFont.load_default()

    for item in bbox_list:
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box") or None
        label = item.get("label") or item.get("name") or ""
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H
        
        # Clip coordinates to be within image bounds
        x1 = int(np.clip(x1, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y2 = int(np.clip(y2, 0, H - 1))

        # Stable color map based on label
        clean_label = re.sub(r'[^a-zA-Z0-9]+', '', label).lower() # extract only a-z, A-Z, 0-9
        h = hashlib.md5(clean_label.encode("utf-8")).hexdigest()
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = 80 + r // 2, 80 + g // 2, 80 + b // 2
        color = (r, g, b, 255)

        # Box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label text and background
        if label and show_object_name:
            text_bbox = font.getbbox(label)
            # Use getlength for a more accurate width in modern Pillow
            tw = font.getlength(label)
            th = text_bbox[3] - text_bbox[1]
            pad = 4

            bg_x1 = x1
            bg_y1 = y1 - th - pad * 2
            bg_x2 = x1 + tw + pad * 2
            bg_y2 = y1
            
            if bg_y1 < 0:
                bg_y1 = y1
                bg_y2 = y1 + th + pad * 2
            
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(r, g, b, 200))
            # Use anchor="lt" for simpler and more accurate text positioning
            draw.text((bg_x1 + pad, bg_y1 + pad), label, font=font, fill=(255, 255, 255, 255), anchor="lt")

def _calculate_text_block_layout(text, font, max_w, margin, line_spacing):
    """Internal helper to calculate text layout properties."""
    lines = []
    for para in str(text).split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            test = cur + " " + word if cur else word
            if font.getlength(test) <= max_w - 2 * margin:
                cur = test
            else:
                if cur: lines.append(cur)
                cur = word
        if cur: lines.append(cur)

    if not lines: return [], [], [], 0, 0
    
    line_sizes = [font.getbbox(line) for line in lines]
    heights = [box[3] - box[1] for box in line_sizes]
    widths = [font.getlength(line) for line in lines]
    
    text_h = sum(heights) + line_spacing * (len(lines) - 1)
    text_w = min(max(widths, default=0), max_w - 2 * margin)
    
    box_w = text_w + 2 * margin
    box_h = text_h + 2 * margin

    return lines, line_sizes, heights, box_w, box_h

def draw_text_block_on_overlay(draw: ImageDraw.ImageDraw, size: Tuple[int, int], text: str,
                               bg_transparency: float = 0.8, pos: str = "top-left", font: ImageFont.FreeTypeFont = None, 
                               color: Tuple[int, int, int] = (255, 255, 255), max_width_ratio: float = 0.7):
    """
    Draws a multi-line text block directly onto a given ImageDraw object.
    This function does not create images or handle compositing.
    """
    if font is None:
        font = ImageFont.load_default()

    margin = int(font.size / 2 + 0.5)
    line_spacing = int(font.size / 2.5 + 0.5)

    W, H = size
    max_w = int(W * max_width_ratio)

    lines, line_sizes, heights, box_w, box_h = _calculate_text_block_layout(
        text, font, max_w, margin, line_spacing
    )
    if not lines: return

    # Calculate top-left corner (x0, y0) of the text block
    original_H = H
    if pos == "append-bottom":
        # In the refactored logic, H is the *new* canvas height. We need the original.
        original_H = H - box_h - 16
        x0, y0 = 8, original_H + 8
    elif pos == "top-left":
        x0, y0 = 8, 8
    elif pos == "bottom-left":
        x0, y0 = 8, H - 8 - box_h
    else:
        raise ValueError(f"Invalid pos value: {pos}")

    bg_alpha = int(255 * bg_transparency)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, bg_alpha))

    current_y = y0 + margin
    for line, line_size, h in zip(lines, line_sizes, heights):
        # Use anchor="lt" for robust vertical alignment
        draw.text((x0 + margin, current_y), line, font=font, fill=(*color, 255), anchor="lt")
        current_y += h + line_spacing

def merge_layers(img_rgba: Image.Image, overlay: Image.Image) -> Image.Image:
    """
    Composites an overlay onto an RGBA base image and converts the result to RGB.
    """
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")

def draw_bboxes(img: Image.Image, bbox_list: List[Dict], show_object_name=False, line_width=3, font=None):
    """
    Draw bounding boxes on the image. 

    Args:
        img (Image.Image): PIL Image object
        bbox_list (List[Dict]): List of bounding box dictionaries, each containing:
            - "bbox_2d" or "bbox" or "box": [x1, y1, x2, y2]
            - "label" or "name": object label
        show_object_name (bool): Whether to display the object name on the box
        line_width (int): Width of the bounding box lines
        font (ImageFont.FreeTypeFont): Font for the object names, which size is configured when creating the font object.
    
    Returns:
        Image.Image: PIL Image object with bounding boxes drawn
    """
    if not bbox_list:
        return img
    # 1. Get drawing tools
    img_rgba, overlay, draw = get_draw_object(img)
    # 2. Draw on the overlay
    draw_bboxes_on_overlay(
        draw=draw, 
        size=img_rgba.size, 
        bbox_list=bbox_list, 
        show_object_name=show_object_name, 
        line_width=line_width, 
        font=font
    )
    # 3. Merge layers and return
    return merge_layers(img_rgba, overlay)   

def draw_text_block(img: Image.Image, text: str, bg_transparency=0.8, pos="top-left",
                    font=None, color=(255,255,255), max_width_ratio=0.7) -> Image.Image:
    """
    Draw multi-line text block on the entire image (with semi-transparent background), pos in {"top-left","bottom-left","append-bottom"}.
    
    Args:
        img (Image.Image): PIL Image object
        text (str): Text to draw
        font (ImageFont.FreeTypeFont, optional): Font object. If None, default font is used. Font size is configured when creating the font object.
        color (tuple, optional): Color of text (R, G, B)
        bg_transparency (float, optional): Background transparency in [0.0, 1.0], 0.0 is fully transparent, 1.0 is fully opaque
        pos (str, optional): Position of the text block, "top-left" or "bottom-left" or "append-bottom"
        max_width_ratio (float, optional): Maximum width ratio of the text block to the image width
    
    Returns:
        Image.Image: PIL Image object with text block drawn
    """
    if not text:
        return img
    if font is None:
        font = ImageFont.load_default()
    # Pre-calculate layout to determine if canvas expansion is needed
    margin = int(font.size / 2 + 0.5)
    line_spacing = int(font.size / 2.5 + 0.5)
    max_w = int(img.width * max_width_ratio)
    _, _, _, _, box_h = _calculate_text_block_layout(
        text, font, max_w, margin, line_spacing
    )
    # 1. Get drawing tools (handles canvas expansion if needed)
    final_img_rgba, overlay, draw = get_draw_object(
        img, 
        pos_for_append=pos, 
        append_height=box_h if pos == "append-bottom" else 0
    )
    # 2. Draw on the overlay
    draw_text_block_on_overlay(
        draw=draw,
        size=final_img_rgba.size,
        text=text,
        bg_transparency=bg_transparency,
        pos=pos,
        font=font,
        color=color,
        max_width_ratio=max_width_ratio
    )
    # 3. Merge layers and return
    return merge_layers(final_img_rgba, overlay)

def annotate_frames(frames: List[Image.Image], bbox_list: List[List[Dict]], tasks: Optional[List[str]] = None, sub_tasks: Optional[List[str]] = None, cots: Optional[List[str]] = None, 
                    show_plans: bool = True, show_object_name: bool=True, font=None, min_width: int=512) -> List[Image.Image]:
    assert len(frames) == len(bbox_list), "Number of frames must match number of bbox lists."
    if tasks is not None:
        assert len(frames) == len(tasks), "Number of frames must match number of tasks."
    if sub_tasks is not None:
        assert len(frames) == len(sub_tasks), "Number of frames must match number of sub-tasks."
    if cots is not None:
        assert len(frames) == len(cots), "Number of frames must match number of cots."
    # Prepare plans if needed
    if sub_tasks is not None and show_plans:
        plans = []
        cur_plan = []
        for i in range(len(frames)-1, -1, -1):
            if sub_tasks[i] and sub_tasks[i] not in ['done', 'not done'] and sub_tasks[i] not in cur_plan:
                cur_plan.insert(0, sub_tasks[i])
            plans.append(cur_plan.copy())
        plans = plans[::-1]
    else:
        plans = None
    # load font if not provided
    if font is None:
        font = ImageFont.load_default()
    # Annotate each frame
    annotated_frames = []
    for i, (frame, bboxes) in enumerate(zip(frames, bbox_list)):
        frame = frame.resize((min_width, int(frame.height * min_width / frame.width)), resample=Image.Resampling.LANCZOS)
        # prepare task
        task_str = ""
        if tasks is not None:
            task_str = f"Input: {tasks[i]}".replace("\n", " ").strip()
        # prepare sub-task
        sub_task_str = ""
        if sub_tasks is not None:
            sub_task_str = sub_tasks[i].replace("\n", " ").strip()
        # prepare appended text
        app_text = ""
        if cots is not None and cots[i]:
            app_text += f"Reasoning: {cots[i]}\n\n"
        if plans is not None and plans[i]:
            app_text += "Planned Actions: " + " > ".join(plans[i])
        app_text = app_text.strip()
        # precalculate appended text height if needed
        append_height = 0
        if app_text:
            margin = int(font.size / 2 + 0.5)
            line_spacing = int(font.size / 2.5 + 0.5)
            # max_width_ratio for appended text is 1.0
            max_w = int(frame.width * 1.0) 
            _, _, _, _, box_h = _calculate_text_block_layout(app_text, font, max_w, margin, line_spacing)
            append_height = box_h
        # get canvas and draw object within single effort
        final_img_rgba, overlay, draw = get_draw_object(
            frame,
            pos_for_append="append-bottom" if app_text else None,
            append_height=append_height
        )
        # compose everything on the overlay
        draw_bboxes_on_overlay(draw, frame.size, bboxes, show_object_name, font=font)
        if task_str:
            draw_text_block_on_overlay(draw, frame.size, task_str, bg_transparency=0.5, pos="top-left", font=font)
        if sub_task_str:
            draw_text_block_on_overlay(draw, frame.size, sub_task_str, bg_transparency=0.5, pos="bottom-left", font=font)
        if app_text: # use full size for appended text
            draw_text_block_on_overlay(draw, final_img_rgba.size, app_text, bg_transparency=0, pos="append-bottom", font=font, max_width_ratio=1.0)
        # merge layers and append
        final_frame = merge_layers(final_img_rgba, overlay)
        annotated_frames.append(final_frame)
    return annotated_frames

def adapt_frames_size(frames: List[Image.Image]) -> List[Image.Image]:
    """
    Adapt all frames to have the same size (the max width and height among all frames).
    Extra areas are filled with black. (Optimized Version)
    """
    if not frames:
        return []
    max_w, max_h = map(max, zip(*[frame.size for frame in frames]))
    # Create a single, reusable black canvas
    black_canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))    
    adapted_frames = []
    for frame in frames:
        if frame.width == max_w and frame.height == max_h:
            adapted_frames.append(frame)
        else:
            # .copy() is significantly faster than Image.new() in a loop.
            new_frame = black_canvas.copy()
            new_frame.paste(frame, (0, 0))
            adapted_frames.append(new_frame)
    return adapted_frames

def images_to_video(images: List[Image.Image], output_path: Path | str, fps: int=25, show_log: bool=True):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    # check if need adapt size
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        images = adapt_frames_size(images)
        logging.warning("Input images have different sizes, adapted to the same size for video generation.")
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)
    frames = [np.array(img) for img in images]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False, logger=('bar' if show_log else None))
    
def images_to_video_base64(images: List[Image.Image], fps: int=25):
    tmp_path = Path("tmp") / f"{hash_image(images[0])}.mp4"
    images_to_video(images, tmp_path, fps=fps)
    with open(tmp_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    tmp_path.unlink(missing_ok=True)
    return video_base64

def save_video_with_bboxes(frames: List[Image.Image], bbox_list: List[List[Dict]], save_path: Path | str, tasks: Optional[List[str]] = None, fps: int=25, show_object_name: bool=True):
    assert len(frames) == len(bbox_list), "Number of frames must match number of bbox lists."
    if tasks is not None:
        assert len(frames) == len(tasks), "Number of frames must match number of tasks."
    frames_with_bboxes = []
    for i, (frame, bboxes) in enumerate(zip(frames, bbox_list)):
        if frame.width < 512:
            frame = frame.resize((512, int(frame.height * 512 / frame.width)), resample=Image.Resampling.LANCZOS)
        frame_with_bboxes = draw_bboxes(frame, bboxes, show_object_name=show_object_name)
        if tasks is not None:
            frame_with_bboxes = draw_text_block(frame_with_bboxes, tasks[i])
        frames_with_bboxes.append(frame_with_bboxes)
    images_to_video(frames_with_bboxes, save_path, fps=fps)



### Legacy Utils ###
def improved_json_parser(text_response: str):
    if not isinstance(text_response, str):
        raise TypeError("Input must be a string.")

    def remove_json_comments(json_str: str) -> str:
        """Remove // comments from JSON-like string (ignore http://, https://)."""
        lines = []
        for line in json_str.splitlines():
            line = re.split(r'(?<!http:)(?<!https:)//', line)[0]
            lines.append(line.rstrip())
        return '\n'.join(lines)

    def try_parse(candidate: str):
        """Try to parse and validate JSON list of dicts, with repair for malformed items."""
        try:
            js = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(js, list):
            return None
        repaired_list = []
        for item in js:
            if not isinstance(item, dict):
                continue
            if ("bbox_2d" in item and "label" in item 
            and isinstance(item["bbox_2d"], list) and len(item["bbox_2d"]) == 4 and all(isinstance(coord, (int, float)) for coord in item["bbox_2d"]) 
            and isinstance(item["label"], str)):
                repaired_list.append(item)
                continue
            # try to repair keys or remove invalid items
            repaired = {}
            for _, v in item.items():
                if isinstance(v, list) and len(v) == 4 and all(isinstance(coord, (int, float)) for coord in v):
                    repaired["bbox_2d"] = v
                elif isinstance(v, str):
                    repaired["label"] = v
            if "bbox_2d" in repaired and "label" in repaired:
                repaired_list.append(repaired)
        return repaired_list if repaired_list else None

    text_response = text_response.strip()

    # --- 1. Try full text directly ---
    js = try_parse(text_response)
    if js is not None:
        return js

    # --- 2. Try extracting from code blocks ---
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            cleaned = remove_json_comments(match.group(1).strip())
            js = try_parse(cleaned)
            if js is not None:
                return js

    # --- 3. Fallback: longest bracketed JSON-like substring ---
    for start, end in [('[', ']'), ('{', '}')]:
        s_idx, e_idx = text_response.find(start), text_response.rfind(end)
        if s_idx != -1 and e_idx > s_idx:
            cleaned = remove_json_comments(text_response[s_idx:e_idx+1])
            js = try_parse(cleaned)
            if js is not None:
                return js

    raise ValueError("Could not extract valid JSON list of dicts from the input string")

def check_normalized(json_response: List[Dict]):
    for obj in json_response:
        bbox = obj.get('bbox_2d')
        if not bbox:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
            return False
    return True

def normalize_bbox(json_response: List[Dict], image_width=1000, image_height=1000):
    """
    Transforms the API response to the required format.
    - Unifies keys ('box'/'name' to 'bbox_2d'/'label').
    - Converts coordinates from API's [y1, x1, y2, x2] in [0, 1000] range
      to normalized [x1, y1, x2, y2] in [0, 1] range.
    The width and height parameters are ignored to maintain signature compatibility,
    as normalization is based on the API's fixed 1000x1000 grid.
    """
    processed_json_response = []
    for obj in json_response:
        # Unify keys for bounding box and label
        bbox = obj.get('bbox_2d') or obj.get('box_2d') or obj.get('box') or obj.get('bbox')
        label = obj.get('label') or obj.get('name')
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (ValueError, TypeError):
            continue # Skip if coordinates are not numbers

        # Normalize to [0, 1] and reorder to [x1, y1, x2, y2]
        norm_x1 = max(0.0, min(x1 / image_width, 1.0))
        norm_y1 = max(0.0, min(y1 / image_height, 1.0))
        norm_x2 = max(0.0, min(x2 / image_width, 1.0))
        norm_y2 = max(0.0, min(y2 / image_height, 1.0))

        if not (norm_x1 < norm_x2 and norm_y1 < norm_y2): continue

        processed_json_response.append({
            'bbox_2d': [norm_x1, norm_y1, norm_x2, norm_y2],
            'label': label
        })
    return processed_json_response

def denormalize_bbox(json_response: List[Dict], image_width=1000, image_height=1000) -> List[Dict]:
    denormalized_json_response = []
    for obj in json_response:
        bbox = obj.get('bbox_2d')
        label = obj.get('label')
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (ValueError, TypeError):
            continue # Skip if coordinates are not numbers

        denorm_x1 = int(x1 * image_width)
        denorm_y1 = int(y1 * image_height)
        denorm_x2 = int(x2 * image_width)
        denorm_y2 = int(y2 * image_height)

        denormalized_json_response.append({
            'bbox_2d': [denorm_x1, denorm_y1, denorm_x2, denorm_y2],
            'label': label
        })
    return denormalized_json_response

def json_response_to_bboxes(json_response: List[Dict], disable_shuffle: bool=False) -> List[Dict]:
    bboxes = []
    colors = ["red", "lime", "blue", "yellow", "cyan", "magenta", "deeppink", "orange", "purple", "lightgreen"]
    if not disable_shuffle:
        random.shuffle(colors)
    else:
        colors = ['red']
    json_response = sorted(json_response, key=lambda obj: (obj['bbox_2d'][2]-obj['bbox_2d'][0])*(obj['bbox_2d'][3]-obj['bbox_2d'][1]))
    for i, obj in enumerate(json_response):
        bboxes.append({
            "box": obj['bbox_2d'],
            "name": obj['label'],
            "color": colors[i % len(colors)]
        })
    return bboxes

def load_parquet(dataset_path) -> List[str]:
    parquet_paths = []
    if type(dataset_path) is str:
        dataset_path = Path(dataset_path)
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist."
    data_root = dataset_path / "data"
    assert data_root.exists(), f"Data root {data_root} does not exist."
    # find 'chunk-*' folders
    chunk_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    assert chunk_folders, f"No chunk folders found in {data_root}."
    for chunk_folder in chunk_folders:
        # find 'part-*.parquet' files
        parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if not parquet_files:
            logging.error(f"No parquet files found in {chunk_folder}.")
            continue
        parquet_paths.extend(parquet_files)
    return parquet_paths
