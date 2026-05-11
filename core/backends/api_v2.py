import os
import io
import re
import json
import base64
import logging
import asyncio
import cv2
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def encode_image_to_base64(image_input: Union[str, Image.Image]) -> str:
    """Helper to convert path or PIL image to base64 string."""
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            binary_data = image_file.read()
    elif isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="PNG")
        binary_data = buffered.getvalue()
    else:
        raise ValueError("Unsupported image type")
    
    return base64.b64encode(binary_data).decode("utf-8")

def extract_video_frames(video_path: str, num_frames: int = 8) -> List[str]:
    """Extract frames from a video path and return as base64 list."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames_b64 = []
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if idx in indices:
            _, buffer = cv2.imencode(".jpg", frame)
            frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames_b64

def clean_model_output(text: str) -> str:
    """Remove thought tags and extra markers."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|user\|>', '', text)
    return text.strip()

def api_query(
    messages: List[Dict[str, Any]],
    model_name: str,
    base_url: Optional[str] = None,
    api_key_namespace: str = "API_KEY",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> str:
    """Standard API query for OpenAI-compatible interfaces."""
    client = OpenAI(
        api_key=os.getenv(api_key_namespace),
        base_url=base_url
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        content = completion.choices[0].message.content
        return clean_model_output(content)
    except Exception as e:
        logging.error(f"API Request Failed: {e}")
        return ""

def format_item_to_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses the demo data format into OpenAI message format.
    Supports 'image' (list or str) and 'video' (path).
    """
    content = []
    # 1. Handle Visual Input
    if "video" in item:
        frames = extract_video_frames(item["video"])
        for f in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
    elif "image" in item:
        imgs = item["image"] if isinstance(item["image"], list) else [item["image"]]
        for img_path in imgs:
            b64 = encode_image_to_base64(img_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    # 2. Handle Text Input (Assuming first turn is human)
    human_text = item["conversations"][0]["value"]
    # Remove the <image> placeholder if model doesn't need it explicitly in text
    human_text = human_text.replace("<image>\n", "").replace("\n<image>", "")
    content.append({"type": "text", "text": human_text})

    return [{"role": "user", "content": content}]

def batch_generate(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    base_url: Optional[str] = None,
    api_key_namespace: str = "API_KEY",
    max_workers: int = 8,
    **kwargs
) -> List[str]:
    """
    Parallel generation for a batch of items.
    model_path is used as model_name for the API.
    """
    def _single_task(item):
        messages = format_item_to_messages(item)
        return api_query(
            messages=messages,
            model_name=model_path,
            base_url=base_url,
            api_key_namespace=api_key_namespace,
            **kwargs
        )

    async def _run_async():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [loop.run_in_executor(executor, _single_task, item) for item in batch_items]
            return await asyncio.gather(*tasks)

    # If running in a script with an existing loop, use this logic
    try:
        return asyncio.run(_run_async())
    except RuntimeError:
        # Fallback for nested event loops (e.g. Jupyter or specific environments)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_single_task, batch_items))

# Usage Example for partial:
# from functools import partial
# custom_batch_gen = partial(batch_generate, base_url="https://api.openai.com/v1", api_key_namespace="GPT4_KEY")