import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any
from core.utils.common import denormalize_bbox
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

load_dotenv()
sam_ckpt = os.getenv('SAM2_CKPT')
sam_config = 'configs/sam2.1/sam2.1_hiera_b+.yaml'

_cached_predictor = None
def get_sam2_predictor():
    global _cached_predictor
    if _cached_predictor is None:
        sam2_model = build_sam2(sam_config, sam_ckpt)
        _cached_predictor = SAM2ImagePredictor(sam2_model)
    return _cached_predictor

def set_image_for_sam2(image: Image.Image):
    predictor = get_sam2_predictor()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    predictor.set_image(np.array(image))

def segment_by_box(box: List[Tuple[int, int, int, int]]):
    """
    Args:
        box: [x1, y1, x2, y2], absolute coordinates, 0 <= x1 < x2 <= image width, 0 <= y1 < y2 <= image height. All coordinates are integers
    Returns:
        masks: np.ndarray of shape (N, H, W), binary masks for each detected object. Each digit is float32 type 0 or 1.
        scores: np.ndarray of shape (N,), confidence scores for each mask.
    """
    box = np.array(box).reshape(1, 4)
    predictor = get_sam2_predictor()
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    return masks, scores

def segment_by_point(point: List[Tuple[int, int]]):
    """
    Args:
        point: [x, y], absolute coordinates, 0 <= x < image width, 0 <= y < image height. All coordinates are integers
    Returns:
        mask: np.ndarray of shape (H, W), binary mask for the detected object. Each digit is float32 type 0 or 1.
    """
    keypoint = np.array(point).reshape(1, 2)
    keypoint_label = np.array([1])
    predictor = get_sam2_predictor()
    masks, scores, logits = predictor.predict(
        point_coords=keypoint,
        point_labels=keypoint_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    return masks[0]

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
    set_image_for_sam2(image)
    scores = []
    boxes = denormalize_bbox(boxes)
    for box_dict in boxes:
        box = box_dict['bbox_2d']
        masks, box_scores = segment_by_box(box)
        show_mask(image, masks[0], box_scores[0])
        scores.append(float(box_scores[0]))
    for box, score in zip(boxes, scores):
        box['score'] = score
    return boxes

def mask_to_bbox(mask: np.ndarray, padding: int = 0, min_area_ratio: float = 0.01) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary mask to bounding box coordinates, with noise removal.
    Keeps only the main connected component to avoid outlier noise.

    Args:
        mask: 2D numpy array (H, W), binary mask where 1 indicates the object (float32 or bool).
        padding: int, number of pixels to pad the bounding box on each side.
        min_area_ratio: float, minimum area ratio (relative to total mask) for a connected component to be kept.

    Returns:
        A tuple (x1, y1, x2, y2) for the cleaned main component,
        or None if no valid region is found.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    total_area = mask.sum()
    min_area = total_area * min_area_ratio
    # stats: [label, left, top, width, height, area]
    valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    if not valid_labels:
        return None

    main_label = max(valid_labels, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[main_label]

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding - 1, mask.shape[1] - 1)
    y2 = min(y + h + padding - 1, mask.shape[0] - 1)

    return (int(x1), int(y1), int(x2), int(y2))

# === legacy utils ===
_global_image_suffix = 0
def save_npy_image(raw_image, mask, score, save_root="tmp", random_color=False):
    """
    :param image: np.array
    """
    global _global_image_suffix
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # 如果是浮点类型（float32/float64），假设范围是 [0, 1] 或需要归一化
    if mask_image.dtype in [np.float32, np.float64]:
        # 方法1：如果数据在 [0, 1] 范围内
        mask_image = np.clip(mask_image, 0, 1)  # 防止越界
        mask_image = (mask_image * 255).astype(np.uint8)
        
        # 方法2（可选）：如果数据范围未知，可以归一化到 [0, 255]
        # image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # 如果是布尔类型（常见于 mask）
    elif mask_image.dtype == bool:
        mask_image = mask_image.astype(np.uint8) * 255
    
    # 确保通道数合理（PIL 对 4 通道支持有限，可转为 RGB 或 RGBA）
    if mask_image.shape[-1] == 4:
        # 保留 RGBA
        mode = 'RGBA'
    elif mask_image.shape[-1] == 3:
        mode = 'RGB'
    elif len(mask_image.shape) == 2:
        mode = 'L'
    else:
        raise ValueError(f"Unsupported image shape: {mask_image.shape}")
    
    # 转为 uint8（再次确保）
    mask_image = mask_image.astype(np.uint8)
    pil_image = Image.fromarray(mask_image, mode=mode)
    
    box = mask_to_bbox(mask)
    if box is not None:
        draw = ImageDraw.Draw(raw_image)
        font = ImageFont.load_default()
        draw.rectangle(box, outline="blue", width=2)
        text_position = (box[0] + 2, box[1] - 18 if box[1] > 18 else box[1] + 2)
        draw.text(text_position, f"{score:.2f}", fill="blue", font=font)

    # overlay mask on raw image
    blended = Image.blend(raw_image.convert("RGBA"), pil_image, alpha=0.6)
    pil_image = blended.convert("RGB")

    save_path = os.path.join(save_root, f"sam_rgb_{_global_image_suffix}.png")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    pil_image.save(save_path)
    print(f"Save to {save_path}")
    _global_image_suffix += 1

def show_mask(image, mask, score):
    mask = mask.astype(np.uint8)
    save_npy_image(image, mask, score)

def show_masks(image, masks, scores):
    for mask, score in zip(masks, scores):
        show_mask(image, mask, score)