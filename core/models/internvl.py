import math
import re
import torch, torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any, Union
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig
from torchvision.transforms.functional import InterpolationMode
from core.utils.common import (
    __multi_turn_template_turn1,
    __multi_turn_template_turn2,
)


### use LMDeploy ###
def batch_generate_with_lmd(
    images: List[Image.Image],
    questions: List[str],
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    tp: int = 8, dp: int = 1,
    use_pytorch: bool = False,
    **kwargs,
) -> List[str]:
    """
    Generate text for a batch of images and questions using the LMDeploy backend.

    This function initializes and caches the LMDeploy pipeline for efficiency,
    making subsequent calls much faster. It is designed as a drop-in,
    out-of-the-box solution for batch inference with InternVL models.

    Args:
        images (List[Image.Image]): A list of PIL Image objects.
        questions (List[str]): A list of corresponding text prompts.
        model_path (str): The path or name of the model to use (e.g., 'OpenGVLab/InternVL2-8B').
        max_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature for sampling. Set to 0 for greedy decoding.
        tp (int): Tensor parallelism degree.
        dp (int): Data parallelism degree.
        use_pytorch (bool): Whether to use the PyTorch backend. If False, uses Turbomind backend.
        **kwargs: Additional keyword arguments to pass to the pipeline.

    Returns:
        List[str]: A list of generated text responses.
    """
    from core.backends.lmdeploy import get_lmdeploy_pipeline
    lmdeploy_pipeline = get_lmdeploy_pipeline(model_path, tp=tp, dp=dp, use_pytorch=use_pytorch)

    # LMDeploy's pipeline expects a list of (prompt, image) tuples for multimodal input
    prompts = list(zip(questions, images))
    responses = lmdeploy_pipeline(
        prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        **kwargs
    )

    # The pipeline returns a list of GenerationOutput objects. Extract the text from each.
    if not isinstance(responses, list):
        responses = [responses] # Ensure it's a list for single-item batches
        
    generated_texts = [res.text for res in responses]    
    return generated_texts

def batch_multiturn_generate_with_lmd(
    images: List[Image.Image], task_descriptions: List[str], model_path: str,
    question_template_turn1: str = __multi_turn_template_turn1,
    question_template_turn2: str = __multi_turn_template_turn2,
    return_all_turns: bool = False,
    **kwargs,
) -> List[str]:
    questions_turn1 = [question_template_turn1.format(task_desc=td) for td in task_descriptions]
    answers_turn1 = batch_generate_with_lmd(images, questions_turn1, model_path, **kwargs)
    questions_turn2 = [question_template_turn2.replace("<list>", ans.strip()) for ans in answers_turn1]
    answers_turn2 = batch_generate_with_lmd(images, questions_turn2, model_path, **kwargs)
    if return_all_turns:
        return answers_turn1, answers_turn2
    return answers_turn2

# for eval
def sample_video_frames(video_path: str, num_segments: int = 8) -> List[Image.Image]:
    """ Sample video frames uniformly from the video using decord. Copied from LMDeploy examples. """
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    seg_size = max(float(max_frame + 1) / num_segments, 1)
    frame_indices = [
        int(seg_size / 2 + seg_size * i) for i in range(num_segments)
    ]
    imgs = []
    for frame_index in frame_indices:
        if frame_index > max_frame: # safety check
            break
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs

def batch_chat_with_lmd(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    tp: int = 1,
    use_pytorch: bool = False,
    other_kwargs: Optional[Dict[str, Any]] = {},
) -> List[str]:
    """
    Unified Multimodal Generation for InternVL using LMDeploy backend.
    Supports mixed batch of: Text-only, Single-Image, Multi-Image, Video.
    Args:
        batch_items (List[Dict]): List of data items, each containing 'image' (optional), 'video' (optional), and 'conversations' (required, List[Dict[str, Any]], conversations[0]['value'] is the question).
        model_path (str): Path to the model.
        tp (int): Tensor parallel size.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
    """
    from core.backends.lmdeploy import GenerationConfig, get_lmdeploy_pipeline, IMAGE_TOKEN
    lmdeploy_pipeline = get_lmdeploy_pipeline(model_path, tp=tp, use_pytorch=use_pytorch)
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=1 if temperature == 0 else 50,
        **other_kwargs
    )
    # --- prepare inputs ---
    results = []
    for item in batch_items:
        # prompt construction
        raw_prompt = item["conversations"][0]["value"]
        clean_prompt = raw_prompt.replace("<image>", "").replace("<video>", "").strip() # let lmdeploy handle the rest
        text = ''
        # images
        images = item.get("image", [])
        if isinstance(images, str): images = [images]
        image_pool = []
        for i, img in enumerate(images):
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            image_pool.append({'type': 'image_data', 'image_data': {'data': img}, 'max_dynamic_patch': 12})
            text += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # videos
        raw_videos = item.get("video", [])
        if isinstance(raw_videos, str): raw_videos = [raw_videos]
        video_pool = []
        for vid_path in raw_videos:
            frames = sample_video_frames(vid_path)
            video_pool.append([
                {"type": "image_data", "image_data": {"data": frame}, "max_dynamic_patch": 1}
                for frame in frames
            ])
            for i in range(len(frames)):
                text += f'Frame{i+1}: {IMAGE_TOKEN}\n'
        # construct messages
        text += clean_prompt
        content = [dict(type='text', text=text)]
        content.extend(image_pool)
        for vid_frames in video_pool:
            content.extend(vid_frames)
        messages = [dict(role='user', content=content)]
        # inference
        res = lmdeploy_pipeline(messages, generation_config=gen_config)
        results.append(res)
    # extract text
    generated_texts = [res.text for res in results]
    return generated_texts



### use Hugging Face Transformers ###
def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_image(image: Image, input_size: int = 448, use_thumbnail: bool = True, max_num: int = 12) -> torch.Tensor:
    """
    Preprocess the image for model input.
    Args:
        image (Image): Input image.
        input_size (int): Size to resize the image to.
        use_thumbnail (bool): Whether to use a thumbnail image.
        max_num (int): Maximum number of blocks to split the image into.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = build_transform(input_size=input_size)
    patched_images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in patched_images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

_cached_model = None
_cached_tokenizer = None
def get_model(model_path: str) -> tuple:
    """
    Return cached tokenizer and model. Load only once.
    """
    global _cached_model, _cached_tokenizer
    if _cached_model is None or _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device_map = split_model(model_path)
        _cached_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
    return _cached_tokenizer, _cached_model

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video_frames(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    """
    Load video frames and preprocess them using dynamic_preprocess (usually max_num=1 for video).
    Returns:
        pixel_values (Tensor): [N_frames, C, H, W] (if max_num=1) or stacked tiles
        num_patches_list (List[int]): patches per frame
    """
    from decord import VideoReader, cpu
    if VideoReader is None:
        raise ImportError("decord not installed.")
        
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list = []
    num_patches_list = []
    
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        # Video frames usually use max_num=1 to reduce token count
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        
        frame_pixel_values = [transform(tile) for tile in img_tiles]
        frame_pixel_values = torch.stack(frame_pixel_values) # [n_tiles, 3, H, W]
        
        num_patches_list.append(frame_pixel_values.shape[0])
        pixel_values_list.append(frame_pixel_values)
        
    pixel_values = torch.cat(pixel_values_list) # [total_tiles_all_frames, 3, H, W]
    return pixel_values, num_patches_list

def batch_generate(
    images: List[Union[Image.Image, List[Image.Image]]],
    questions: List[str],
    model_path: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> List[str]:
    """
    Generate text for a batch of questions using Hugging Face model (PyTorch backend).
    """
    tokenizer, model = get_model(model_path)
    generation_config = dict(max_new_tokens=max_tokens, temperature=temperature, do_sample=(temperature > 0))
    
    pixel_values_list = []
    for image in images:
        if isinstance(image, list):
            pixel_values = [process_image(img) for img in image]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = process_image(image)
    num_patches_list = [pixel_values.shape[0] for pixel_values in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)

    text_responses = model.batch_chat(
        tokenizer, pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config
    )
    return text_responses

def batch_chat(
    batch_items: List[Dict[str, Any]], 
    model_path: str,
    device: str = "cuda", 
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    **kwargs
) -> List[str]:
    """
    Multimodal Generation for InternVL (Sequential implementation for stability).
    Iterates through items one by one to avoid batch tensor alignment issues.
    """
    tokenizer, model = get_model(model_path)
    responses = []
    generation_config = dict(
        max_new_tokens=max_new_tokens, 
        do_sample=(temperature > 0),
        temperature=temperature
    )
    if temperature > 0:
        generation_config["top_p"] = kwargs.get("top_p", 0.9)

    # simple loop implementation
    for item in batch_items:
        images = item.get("image", [])
        if isinstance(images, str): images = [images]
        videos = item.get("video", [])
        if isinstance(videos, str): videos = [videos]
        raw_question = item["conversations"][0]["value"]
        clean_question = raw_question.replace("<image>", "").replace("<video>", "").strip()
        pixel_tensors = []
        prompt_prefix = ""
        current_num_patches_list = []
        if len(images) > 0:
            for img_path in images:
                pil_img = Image.open(img_path).convert("RGB")
                pixel_v = process_image(pil_img, max_num=12).to(torch.bfloat16).to(device)
                pixel_tensors.append(pixel_v)
                current_num_patches_list.append(pixel_v.size(0))
                prompt_prefix += "<image>\n"
        if len(videos) > 0:
            for vid_path in videos:
                vid_pixel_v, vid_patches_list = load_video_frames(
                    vid_path, max_num=1, num_segments=8
                )
                vid_pixel_v = vid_pixel_v.to(torch.bfloat16).to(device)
                pixel_tensors.append(vid_pixel_v)
                current_num_patches_list.extend(vid_patches_list)
                for i in range(len(vid_patches_list)):
                    prompt_prefix += f"Frame-{i+1}: <image>\n"
        if len(pixel_tensors) > 0:
            pixel_values = torch.cat(pixel_tensors, dim=0)
        else:
            pixel_values = None
            current_num_patches_list = None
        question = prompt_prefix + clean_question
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer, 
                pixel_values, 
                question, 
                generation_config,
                num_patches_list=current_num_patches_list,
                history=None, 
                return_history=True
            )
        responses.append(response)
    return responses