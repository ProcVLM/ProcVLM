import torch
import re
import uuid
import atexit
import asyncio
import threading
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import List, Optional, Union, Tuple, Dict, Any
from qwen_vl_utils import process_vision_info
from core.utils.common import (
    batch_multiturn_generate as __batch_multiturn_generate,
    pack_timestamp_image_content_for_qwen3,
)


### use sglang backend ###
def batch_generate_with_sgllm(
    images,
    questions,
    system_prompt="You are a helpful assistant",
    model_path="Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens=1024,
    temperature=0.1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    from core.backends.sglang import get_sgllm
    runtime_kwargs = engine_kwargs or {}
    rt, processor = get_sgllm(model_path, runtime_kwargs)
    batch_prompts = []
    batch_image_data = []
    for image, question in zip(images, questions):
        if not isinstance(image, list):
            image_list = [image]
        else:
            image_list = image
        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": question}]
                + [{"type": "image", "image": img} for img in image_list]
            },
        ]
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch_prompts.append(prompt_str)
        batch_image_data.append(image_list)
    outputs = rt.generate(
        prompt=batch_prompts,
        image_data=batch_image_data,
        sampling_params=dict(
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    )
    results = []
    for out in outputs:
        text = out["text"]
        if "</think>" in text:
            text = text.split("</think>")[1].strip()
        results.append(text)
    return results

def batch_multiturn_generate_with_sgllm(
    images,
    questions,
    model_path="Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens=1024,
    temperature=0.1,
):
    return __batch_multiturn_generate(
        images, questions,
        model_path=model_path,
        batch_generate_func=batch_generate_with_sgllm,
        max_tokens=max_tokens,
        temperature=temperature,
        __debug_mode=True,
    )

def batch_video_generate_with_sgllm(
    video,
    questions,
    system_prompt="You are a helpful assistant",
    model_path="Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens=1024,
    temperature=0.1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    from core.backends.sglang import get_sgllm
    runtime_kwargs = engine_kwargs or {}
    rt, processor = get_sgllm(model_path, runtime_kwargs)
    batch_prompts = []
    batch_video_data = []
    for (video_path, max_pixel, min_pixel, total_pixel, fps), question in zip(video, questions):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": max_pixel,
                        "min_pixels": min_pixel,
                        "total_pixels": total_pixel,
                        "fps": fps,
                    }
                ],
            },
        ]
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch_prompts.append(prompt_str)
        batch_video_data.append([video_path])
    outputs = rt.generate(
        prompt=batch_prompts,
        video_data=batch_video_data,
        sampling_params=dict(
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    )
    results = []
    for out in outputs:
        text = out["text"]
        if "</think>" in text:
            text = text.split("</think>")[1].strip()
        results.append(text)
    return results

def process_batch_frame_list_sgllm(
    videos,
    questions,
    model_path,
):
    from core.backends.sglang import get_sgllm_processor
    processor = get_sgllm_processor(model_path)
    llm_inputs_batch = []
    for video, question in zip(videos, questions):
        content = pack_timestamp_image_content_for_qwen3(video, question, use_base64=False)
        messages = [{"role": "user", "content": content}]
        raw_images = [
            item["image"] 
            for item in content 
            if item.get("type") == "image"
        ]
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        llm_inputs_batch.append(
            {
                "prompt": prompt_str,
                "images": raw_images, # List[PIL.Image]
            }
        )
    return llm_inputs_batch

def run_batch_frame_list_sgllm(
    llm_inputs_batch,
    model_path,
    max_tokens=8192,
    temperature=0.1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    from core.backends.sglang import get_sgllm_instance
    runtime_kwargs = engine_kwargs or {}
    rt = get_sgllm_instance(model_path, runtime_kwargs)
    batch_prompts = [entry["prompt"] for entry in llm_inputs_batch]     # prompt: List[str]
    batch_image_data = [entry["images"] for entry in llm_inputs_batch]  # image_data: List[List[PIL.Image]]
    outputs = rt.generate(
        prompt=batch_prompts,
        image_data=batch_image_data,
        sampling_params=dict(
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    )
    results = []
    for out in outputs:
        text = out["text"]
        if "</think>" in text:
            text = text.split("</think>")[1].strip()
        results.append(text)
    return results

def batch_frame_list_generate_with_sgllm(
    video,
    questions,
    model_path,
    max_tokens=8192,
    temperature=0.1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    llm_inputs_batch = process_batch_frame_list_sgllm(video, questions, model_path)
    return run_batch_frame_list_sgllm(
        llm_inputs_batch,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        engine_kwargs=engine_kwargs,
    )



### use vLLM backend ###
def batch_generate_with_vllm(
    images: List[Union[List[Image.Image], Image.Image]],
    questions: List[str],
    system_prompt: str = "You are a helpful assistant",
    model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens: int = 1024,
    tp: int = 1,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    """
    from core.backends.vllm import SamplingParams, get_vllm, run_batch_async
    assert len(images) == len(questions), "Number of images must match number of questions."
    runtime_kwargs = engine_kwargs or {}
    llm, processor = get_vllm(model_path, tp, runtime_kwargs)
    sampling_config = {
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if sampling_kwargs:
        sampling_config.update(sampling_kwargs)
    sampling_params = SamplingParams(**sampling_config)
    inputs_to_submit = []
    for image, question in zip(images, questions):
        # Construct messages in the format expected by Qwen-VL processor and process_vision_info
        if not isinstance(image, list):
            image = [image]
        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ] + [
                    {
                        "type": "image",
                        "image": img,
                        # Optional: min_pixels/max_pixels if your process_vision_info requires/supports them
                        # "min_pixels": 224 * 224,
                        # "max_pixels": 1280 * 28 * 28, # Example values
                    } for img in image
                ],
            },
        ]
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, image_patch_size=16, return_video_kwargs=True)
        inputs_to_submit.append({
            "prompt": prompt_str,
            "mm_data": {"image": image_inputs} if image_inputs is not None else {},
            "mm_kwargs": video_kwargs or {},
            "sampling_params": sampling_params
        })
    return run_batch_async(llm, inputs_to_submit)

def batch_multiturn_generate_with_vllm(
    images: List[Image.Image],
    questions: List[List[str]],
    model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> List[str]:
    def _batch_generate_with_sampling(images, questions, model_path, max_tokens, temperature=0.1, **kwargs):
        sampling_kwargs = dict(kwargs.pop("sampling_kwargs", {}) or {})
        sampling_kwargs.setdefault("temperature", temperature)
        return batch_generate_with_vllm(
            images,
            questions,
            model_path=model_path,
            max_tokens=max_tokens,
            sampling_kwargs=sampling_kwargs,
            **kwargs,
        )

    return __batch_multiturn_generate(
        images, questions,
        model_path=model_path,
        batch_generate_func=_batch_generate_with_sampling,
        max_tokens=max_tokens,
        temperature=temperature,
        __debug_mode=True,
    )

def batch_video_generate_with_vllm(
    video: List[Tuple[str, int, int, int, int]],
    questions: List[str],
    system_prompt: str = "You are a helpful assistant",
    model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    max_tokens: int = 1024,
    tp: int = 1,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    video: List of tuples (video_path, max_pixel, min_pixel, total_pixel, fps)
    """
    from core.backends.vllm import SamplingParams, get_vllm, run_batch_async
    assert len(video) == len(questions), "Number of videos must match number of questions."
    runtime_kwargs = engine_kwargs or {}
    llm, processor = get_vllm(model_path, tp, runtime_kwargs)
    sampling_config = {
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if sampling_kwargs:
        sampling_config.update(sampling_kwargs)
    sampling_params = SamplingParams(**sampling_config)
    inputs_to_submit = []
    for (video_path, max_pixel, min_pixel, total_pixel, fps), question in zip(video, questions):
        # Construct messages in the format expected by Qwen-VL processor and process_vision_info
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": max_pixel,
                        "min_pixels": min_pixel,
                        "total_pixels": total_pixel,
                        "fps": fps,
                    }
                ],
            },
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs_to_submit.append({
            "prompt": prompt_str,
            "mm_data": mm_data,
            "mm_kwargs": video_kwargs or {},
            "sampling_params": sampling_params
        })
    return run_batch_async(llm, inputs_to_submit)

# qwen3-vl features a more precise video input interface, using the interleaved frame-timestamp list.
# The following functions are adapted for that interface.
def process_batch_frame_list_vllm(
    videos: List[List[Tuple[int, Image.Image]]],
    questions: List[str],
    model_path: str,
    max_tokens: int = 8192,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
        videos: List of video, list of Tuples (timestamp, Image)
        example of interleaved frame-timestamp list for one video:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<0.0 seconds>"
                    },
                    {
                        "type": "image_url",
                        "image_url":
                        {
                            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000000.jpg"
                        }
                    },
                    {
                        "type": "text",
                        "text": "<1.0 seconds>"
                    },
                    {
                        "type": "image_url",
                        "image_url":
                        {
                            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000030.jpg"
                        }
                    },
                    ...,
                    {
                        "type": "text",
                        "text": "Given the query \"there is a moving bicycle towards an adult in black in a path.\", for each frame, detect and localize the visual content described by the given textual query in JSON format. If the visual content does not exist in a frame, skip that frame. Output Format: [{\"time\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}, {\"time\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}, ...]."
                    }
                ]
            }
        ]
    """
    from core.backends.vllm import SamplingParams, get_vllm_processor
    assert len(videos) == len(questions), "Number of videos must match number of questions."
    processor = get_vllm_processor(model_path)
    sampling_config = {
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if sampling_kwargs:
        sampling_config.update(sampling_kwargs)
    sampling_params = SamplingParams(**sampling_config)
    inputs_to_submit = []
    for video, question in zip(videos, questions):
        content = pack_timestamp_image_content_for_qwen3(video, question, use_base64=False)
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs_to_submit.append({
            "prompt": prompt_str,
            "mm_data": mm_data,
            "mm_kwargs": video_kwargs or {},
            "sampling_params": sampling_params,
        })
    return inputs_to_submit

def run_batch_frame_list_vllm(
    llm_inputs_batch: List[Dict[str, Any]],
    model_path: str,
    tp: int = 1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    video: List of tuples (video_path, max_pixel, min_pixel, total_pixel, fps)
    """
    from core.backends.vllm import get_vllm_instance, run_batch_async
    runtime_kwargs = engine_kwargs or {}
    llm = get_vllm_instance(model_path, tp, runtime_kwargs)
    return run_batch_async(llm, llm_inputs_batch)

def batch_frame_list_generate_with_vllm(
    video: List[List[Tuple[int, Image.Image]]],
    questions: List[str],
    model_path: str,
    max_tokens: int = 8192,
    tp: int = 1,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    video: List of tuples (video_path, max_pixel, min_pixel, total_pixel, fps)
    """
    llm_inputs_batch = process_batch_frame_list_vllm(
        video,
        questions,
        model_path,
        max_tokens,
        sampling_kwargs,
    )
    return run_batch_frame_list_vllm(
        llm_inputs_batch,
        model_path,
        tp=tp,
        engine_kwargs=engine_kwargs,
    )


def process_batch_chat_vllm(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build vLLM-ready batched inputs for universal chat usage."""
    from core.backends.vllm import SamplingParams, get_vllm_processor

    processor = get_vllm_processor(model_path)
    sampling_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if sampling_kwargs:
        sampling_config.update(sampling_kwargs)
    sampling_params = SamplingParams(**sampling_config)

    inputs_to_submit = []
    for item in batch_items:
        # --- build messages adpated from Qwen-VL data processor ---
        images = item.get("image") or []
        if isinstance(images, str):
            images = [images]
        videos = item.get("video") or []
        if isinstance(videos, str):
            videos = [videos]
        image_pool = [
            {"type": "image", "image": img} for img in images
        ]
        video_pool = [
            {"type": "video", "video": vid} for vid in videos
        ]
        messages = []
        for turn in item["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            text: str = turn["value"]
            if role == "user":
                content = []
                # Split text by <image> or <video> placeholders while keeping delimiters
                text_parts = re.split(r"(<image>|<video>)", text)
                for seg in text_parts:
                    if seg == "<image>":
                        if not image_pool:
                            raise ValueError(
                                "Number of <image> placeholders exceeds the number of provided images"
                            )
                        content.append(image_pool.pop(0))
                    elif seg == "<video>":
                        if not video_pool:
                            raise ValueError(
                                "Number of <video> placeholders exceeds the number of provided videos"
                            )
                        content.append(video_pool.pop(0))
                    elif seg.strip():
                        content.append({"type": "text", "text": seg.strip()})
                if image_pool or video_pool:
                    # Unused images/videos are appended before the last text segment.
                    content.extend(image_pool)
                    content.extend(video_pool)
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": text})

        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        inputs_to_submit.append({
            "prompt": prompt_str,
            "mm_data": mm_data,
            "mm_kwargs": video_kwargs or {},
            "sampling_params": sampling_params,
        })

    return inputs_to_submit


def run_batch_chat_vllm(
    llm_inputs_batch: List[Dict[str, Any]],
    model_path: str,
    tp: int = 1,
    disable_async: bool = False,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run preprocessed vLLM chat inputs and return decoded texts."""
    from core.backends.vllm import get_vllm_instance, run_batch_async

    runtime_kwargs = engine_kwargs or {}
    llm = get_vllm_instance(model_path, tp, runtime_kwargs, disable_async=disable_async)

    if disable_async:
        if not llm_inputs_batch:
            return []
        batch_inputs = []
        for entry in llm_inputs_batch:
            prompt = entry["prompt"]
            mm_data = entry["mm_data"]
            mm_kwargs = entry.get("mm_kwargs", {})
            vllm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            if mm_kwargs:
                vllm_inputs["mm_processor_kwargs"] = mm_kwargs
            batch_inputs.append(vllm_inputs)

        # All items from process_batch_chat_vllm share one sampling config.
        sampling_params = llm_inputs_batch[0]["sampling_params"]
        batch_outputs = llm.generate(batch_inputs, sampling_params)
        texts = []
        for final_output in batch_outputs:
            text = final_output.outputs[0].text
            if '</think>' in text:
                answer = text.split('</think>')[1].strip()
            else:
                answer = text.strip()
            texts.append(answer)
        return texts

    return run_batch_async(llm, llm_inputs_batch)

# for eval
def batch_chat_with_vllm(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    tp: int = 1,
    disable_async: bool = False,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Universal Batch Chat for Qwen2-VL / Qwen3-VL using vLLM backend.
    Supports mixed batch of: Text-only, Single-Image, Multi-Image, Video.
    Args:
        batch_items (List[Dict]): List of data items, each containing 'image' (optional), 'video' (optional), and 'conversations' (required, List[Dict[str, Any]], each conversation is a dict like {"from": "human" or "assistant", "value": str}).
        model_path (str): Path to the model.
        tp (int): Tensor parallel size.
        max_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for sampling.
        sampling_kwargs (Optional[Dict[str, Any]]): Extra decoding params for vLLM SamplingParams, e.g. {"temperature": 1.0, "top_p": 0.95, "top_k": 20}.
    """
    llm_inputs_batch = process_batch_chat_vllm(
        batch_items=batch_items,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        sampling_kwargs=sampling_kwargs,
    )
    return run_batch_chat_vllm(
        llm_inputs_batch=llm_inputs_batch,
        model_path=model_path,
        tp=tp,
        disable_async=disable_async,
        engine_kwargs=engine_kwargs,
    )



### use transformers backend ###
# NOTICE: not stable yet, use vLLM version for better results.
_cached_model = None
_cached_processor = None

def get_transformers(model_path: str, device_map="auto"):
    global _cached_model, _cached_processor
    if _cached_model is None or _cached_processor is None:
        _cached_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            _cached_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype="bfloat16",
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=False, # for FP8 models
                attn_implementation="flash_attention_2" # Recommended by docs for video/multi-image
            )
        except Exception as e:
            # Fallback for systems without flash_attn
            print(f"[Info] Loading with SDPA/Default attn due to: {e}")
            _cached_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device_map,
                low_cpu_mem_usage=False, # for FP8 models
                trust_remote_code=True
            )            
    return _cached_model, _cached_processor

def batch_frame_list_generate(
    video: List[List[Tuple[int, Image.Image]]],
    questions: List[str],
    model_path: str,
    max_tokens: int = 8192,
) -> List[str]:
    """
    Generate text for a batch of questions using transformers backend.
    video: List of tuples (video_path, max_pixel, min_pixel, total_pixel, fps)
    """
    import torch
    assert len(video) == len(questions), "Number of videos must match number of questions."
    model, processor = get_transformers(model_path)
    results = []
    for video, question in zip(video, questions):
        content = pack_timestamp_image_content_for_qwen3(video, question, use_base64=False)
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        response = output_text[0]
        results.append(response)
    return results

def batch_chat(
    batch_items: List[Dict[str, Any]], 
    model_path: str,
    device: str = "cuda", 
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    **kwargs
) -> List[str]:
    """
    Universal Batch Generation for Qwen2-VL / Qwen3-VL.
    Supports mixed batch of: Text-only, Single-Image, Multi-Image, Video.
    Args:
        batch_items (List[Dict]): List of data items, each containing 'image' (optional), 'video' (optional), and 'conversations' (required, List[Dict[str, Any]], conversations[0]['value'] is the question).
        model_path (str): Path to the model.
        device (str): Device to run the model on.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
    """
    model, processor = get_transformers(model_path)

    # Qwen-VL batch generation requires left padding
    if processor.tokenizer.padding_side != 'left':
        processor.tokenizer.padding_side = 'left'

    # 1. Construct Messages List
    messages_list = []
    for item in batch_items:
        content = []
        
        # -- Process Images --
        images = item.get("image", [])
        if isinstance(images, str): images = [images]
        
        for img_path in images:
            content.append({
                "type": "image", 
                "image": img_path
            })
            
        # -- Process Videos --
        videos = item.get("video", [])
        if isinstance(videos, str): videos = [videos]
        
        for vid_path in videos:
            content.append({
                "type": "video", 
                "video": vid_path
            })
            
        # -- Process Text --
        # Qwen's apply_chat_template handles prompt formatting,
        # so we just need the raw user query text.
        # We clean <image>/<video> tags as they are implicit in Qwen's message struct.
        raw_query = item["conversations"][0]["value"]
        clean_query = raw_query.replace("<image>", "").replace("<video>", "").strip()
        content.append({
            "type": "text", 
            "text": clean_query
        })
        messages_list.append([
            {"role": "user", "content": content}
        ])

    # 2. Prepare Inputs
    # apply_chat_template processes text formatting
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]
    
    # process_vision_info extracts and processes images/videos from the messages structure
    # This is CRITICAL for Qwen-VL to handle mixed inputs correctly
    image_inputs, video_inputs = process_vision_info(messages_list)
    
    # 3. Tokenize & Pad
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True, # Critical for batch
        return_tensors="pt",
    )
    # Move to device (model.device handles device_map dispatch)
    inputs = inputs.to(model.device)

    # 4. Generate
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        generation_config["temperature"] = temperature
        generation_config["top_p"] = kwargs.get("top_p", 0.9)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)

    # 5. Decode
    # Trim input tokens to get only new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return output_texts
