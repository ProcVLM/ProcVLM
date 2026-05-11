import os
import torch
from dotenv import load_dotenv
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModel
load_dotenv()

model_path = os.getenv('SIGLIP2_CKPT')

# ===== Siglip Helpers =====
# def init_vl_model():
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained(model_path)
    # return model, processor

# def _box_match_score(image: Image.Image, label: str) -> float:
#     # follows the pipeline prompt template to get same results
#     texts = [f'This is a photo of {label}.']
#     # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
#     inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     probs = torch.sigmoid(logits_per_image)
#     # probs = logits_per_image.softmax(dim=1).cpu().numpy()
#     return float(probs[0][0])

def box_match_score(images: List[Image.Image], labels: List[str]) -> List[float]:
    assert len(images) == len(labels), "Number of images and labels must be the same"
    if len(images) == 0:
        return []
    texts = [f'This photo contains a {label}.' for label in labels]
    texts = [t[:100] for t in texts]  # truncate to 100 chars
    # texts = ['The photo shows a background scene or enviroment.']
    inputs = processor(text=texts, images=images, padding="max_length", max_length=64, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    # return [1 - float(prob) for prob in probs[:, 0]]
    return [float(prob) for prob in probs.diagonal()]

def padding_box(box: List[int], padding_size: float=0.025) -> List[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - padding_size),
        max(0, y1 - padding_size),
        min(1, x2 + padding_size),
        min(1, y2 + padding_size),
    ]

def crop_box(image: Image.Image, box: List[int]) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = box
    return image.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))

def process_a_frame(image: Image.Image, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a single frame and return boxes with match scores.
    Args:
        image: PIL Image of the frame
        boxes: List of dicts, each with keys 'label' and 'bbox_2d' (bbox is [x1, y1, x2, y2])
    Returns:
        List of dicts, each with keys 'label', 'bbox_2d', and 'score'
    """
    if not boxes:
        return []
    imgs = [crop_box(image, padding_box(box['bbox_2d'])) for box in boxes]
    labels = [box['label'] for box in boxes]
    scores = box_match_score(imgs, labels)
    for box, score in zip(boxes, scores):
        box['score'] = score
    # viz_results(imgs, boxes)
    return boxes
