import logging
import json, jsonlines
import time
import re
import hashlib
import decord, av, ffmpeg
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets.features.image import Image as HFImage
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from io import BytesIO
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from fastparquet import ParquetFile
from collections import OrderedDict
from decord import VideoReader, DECORDError

# ==== configuration ==== #
# --- set third-view and ego-view camera key names for different datasets --- #
EGO_CAMERA_MAP = {
    # 'dataset name': 'ego camera key name',
    'AgiBotAlpha_covt2lerobot_0818': 'observation.images.cam_2',
    'AgiBotBeta_covt2lerobot_0818': 'observation.images.cam_2',
    'austin_buds_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sailor_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sirius_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'stanford_hydra_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': 'observation.images.cam1',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam',
    'robo_set_new': 'observation.images.cam3',
    'fmb': 'observation.images.cam3',
    'berkeley_autolab_ur5': 'observation.images.cam1',
    'berkeley_fanuc_manipulation': 'observation.images.cam1',
    # 'qut_dexterous_manpulation': 'observation.images.cam1',
    'taco_play': 'observation.images.cam1',
    'jaco_play': 'observation.images.cam1',
    'utaustin_mutex': 'observation.images.cam1',
    'plex_robosuite': 'observation.images.cam1',
}
EXO_CAMERA_MAP = {
    # 'dataset name': 'third camera key name',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam1',
}
# ==== end of configuration ==== #



### Image and Video Processing Helpers ###
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
                    show_plans: bool = True, show_object_name: bool=True, font=None) -> List[Image.Image]:
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



# ===== Utility Functions ===    
def clean_string(s: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F]", "", s)

# copied from lerobot.datasets.utils
def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info

# We disabled sorting to make it faster
def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    for item in tasks: item["task"] = clean_string(item["task"])
    # tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    tasks = {item["task_index"]: item["task"] for item in tasks}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index

def load_cots(local_dir: Path) -> dict:
    cots = load_jsonlines(local_dir / COTS_PATH)
    # cots = {item["cot_index"]: item["cot"] for item in sorted(cots, key=lambda x: x["cot_index"])}
    cots = {item["cot_index"]: item["cot"] for item in cots}
    return cots

def load_sub_tasks(local_dir: Path) -> dict:
    sub_tasks = load_jsonlines(local_dir / SUB_TASKS_PATH)
    # sub_tasks = {item["sub_task_index"]: item["sub_task"] for item in sorted(sub_tasks, key=lambda x: x["sub_task_index"])}
    sub_tasks = {item["sub_task_index"]: item["sub_task"] for item in sub_tasks}
    return sub_tasks

def format_bboxes(bboxes: list) -> str:
    if len(bboxes) == 0:
        return ""
    for bbox_dict in bboxes:
        bbox_dict["bbox_2d"] = [round(number, 3) for number in bbox_dict["bbox_2d"]]
    bboxes_str = json.dumps(bboxes, indent=2, ensure_ascii=False)
    return bboxes_str

def load_bboxes(local_dir: Path) -> dict:
    bboxes = load_jsonlines(local_dir / BBOXES_PATH)
    # bboxes = {item["bbox_index"]: format_bboxes(item["bbox"]) for item in sorted(bboxes, key=lambda x: x["bbox_index"])}
    bboxes = {item["bbox_index"]: item["bbox"] for item in bboxes}
    return bboxes

def load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / EPISODES_PATH)
    # return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}
    return {item["episode_index"]: item for item in episodes}

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

def append_jsonlines(data: dict, fpath: Path | str) -> None:
    """Append a single JSON line to a file."""
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)



# ===== Marcos =====
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"
COTS_PATH = "meta/cots.jsonl"
SUB_TASKS_PATH = "meta/sub_tasks.jsonl"
BBOXES_PATH = "meta/bboxes.jsonl"



# ===== Default Features =====
logger = logging.getLogger('vl_reader')

class LeRobotDatasetMetadata:
    # modified version of lerobot.dataset.lerobot_dataset.LerobotDatasetMetadata
    # to improve cache efficiency and speed up loading, focusing on local files only
    def __init__(
        self,
        repo_id: str,
        root: Union[str, Path],
    ):
        self.repo_id = repo_id
        self.root = Path(root)

        try:
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError) as e:
            raise NotImplementedError(f"Cannot load dataset metadata locally: {e})")

    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)

        if (self.root/COTS_PATH).exists():
            self.cots = load_cots(self.root)
        else:
            self.cots = None

        if (self.root/SUB_TASKS_PATH).exists():
            self.sub_tasks = load_sub_tasks(self.root)
        else:
            self.sub_tasks = None

        if (self.root/BBOXES_PATH).exists():
            self.bboxes = load_bboxes(self.root)
        else:
            self.bboxes = None

        self.episodes = load_episodes(self.root)
        # We removed self.stats and self.episodes_stats because they are either optional or useless

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> Union[str, None]:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> Union[str, None]:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, Union[list, dict]]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]
    
    @property
    def state_dim(self) -> int:
        """Dimension size of observation state."""
        assert len(self.features["observation.state"]["shape"]) == 1

        return self.features["observation.state"]["shape"][0]

    @property
    def action_dim(self) -> int:
        """Dimension size of action."""
        assert len(self.features["action"]["shape"]) == 1
        
        return self.features["action"]["shape"][0]
    
    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> Union[int, None]:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    # We removed self.pull_from_repo, self.save_episode, self.update_video_info, classmethod create because they are not needed

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

class FastLerobotVLReader:
    """
    A fast and memory-efficient data reader for LeRobot datasets stored in Parquet format.

    Features:
        - Lazy loading of Parquet files with an LRU cache to limit memory usage.
        - On-the-fly decoding of frames from associated MP4 videos.
        - Pre-computation of a global index map for fast item access.
        - Automatic detection of camera keys and task/sub-task descriptions.
    
    Properties:
        - meta: LeRobotDatasetMetadata, metadata of the dataset.
        - loaded_camera_keys: Set[str], set of camera keys being loaded.
        - sub_task_available: bool, whether sub-task annotations are available.
        - ego_camera_available: bool, whether ego camera is available.
        - all_episode_indices: List[int], list of all episode indices in the dataset.
        - all_episode_paths: List[str], list of all episode parquet file paths in the dataset.
        - all_episode_names: List[str], list of all episode parquet file names (Path.stem) in the dataset.

    Functionality:
        - get_episode_range(ep_str): Convert episode string description to (start_index, end_index) tuple.
        - episode_index_to_path(ep_index): Convert episode index to full parquet path.
        - episode_string_to_index(ep_str): Convert episode string description to episode index.
        - set_skip_episode(skip_names): Skip specified episodes by their names (Path.stem).
        - set_skip_episode_by_index(skip_indices): Skip specified episodes by their indices.

    Iterable:
        - __len__(): Returns the total number of frames in the dataset.
        - __getitem__(index): Retrieves a data item by global index or slice. The returned item contains:
            - frame_index: int or np.int64, absolute frame index within the episode
            - task_index: int
            - task: str, task description
            - sub_task_index: int (if available)
            - sub_task: str, sub-task description (if available)
            - task_description: str, either sub-task or task description, prefer sub-task if available
            - exo_image: PIL.Image, image from the third-person camera
            - ego_image: PIL.Image, image from the ego camera (if available)
            - [other camera keys]: PIL.Image, images from other cameras, named by their camera keys (if load_all_camera_keys is True)
            - episode_path: str, path to the *.parquet file (if return_record_meta is True)
            - episode_len: int, number of frames in the episode (if return_record_meta is True)
            - chunk_name: str, chunk directory name (if return_record_meta is True)
            - episode_name: str, episode file name (if return_record_meta is True)
    """
    def __init__(
        self, 
        root: str, 
        ego_name: Optional[str] = None, 
        exo_name: Optional[str] = None, 
        image_resize: Optional[Tuple[int, int]] = None,
        load_all_camera_keys: bool = False,
        return_record_meta: bool = False,
        cache_num: int = 16, 
        prefetch_num: int = 5,
        use_gpu_decoding: bool = False,
        check_episode_integrity: bool = False,
    ):
        """
        Args:
            root (str): Root directory of the LeRobot dataset.
            ego_name (Optional[str]): Name of the ego camera key. If None, auto-detected.
            exo_name (Optional[str]): Name of the third-person camera key. If None, auto-detected.
            image_resize (Optional[Tuple[int, int]]): If provided, resize all images to this size (width, height). Set width or height to -1 to maintain aspect ratio.
            cache_num (int): Number of Parquet files to keep in memory cache.
            prefetch_num (int): Number of upcoming Parquet files to prefetch in the background.
            load_all_camera_keys (bool): Whether to load all camera keys or just ego and third-person.
            return_record_meta (bool): Whether to return metadata about the record (parquet path, offset, etc.) with each item.
        """
        start_time = time.time()
        self.root = Path(root)
        self.name = self.root.name
        self.meta = LeRobotDatasetMetadata(repo_id=self.name, root=root)

        # --- Reader Settings ---
        self.image_resize = image_resize
        self.load_all_camera_keys = load_all_camera_keys
        self.return_record_meta = return_record_meta
        self.cache_num = cache_num
        self.prefetch_num = prefetch_num
        if self.cache_num <= 0:
            self.cache_num = 1
            logger.warning(f"cache_num should be positive. Setting cache_num to {self.cache_num}.")
        if self.prefetch_num < 0:
            self.prefetch_num = 0
            logger.warning(f"prefetch_num should be non-negative. Setting prefetch_num to {self.prefetch_num}.")
        if not (self.cache_num >= 2 * (self.prefetch_num + 1)):
            self.prefetch_num = (self.cache_num // 2) - 1
            logger.warning(f"prefetch_num+1 should be at most half of cache_num. Setting prefetch_num to {self.prefetch_num}.")
        # if self.load_all_camera_keys and self.prefetch_num != 0:
        #     self.prefetch_num = 0
        #     logger.warning("Using prefetch_num > 0 may reduce performance when load_all_camera_keys is True. Setting prefetch_num to 0.")
        if use_gpu_decoding:
            self.decord_ctx = decord.gpu(0)
        else:
            self.decord_ctx = decord.cpu(0)

        # --- Caches and Threading ---
        self.df_cache = OrderedDict()
        self.frame_cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=2*(self.prefetch_num+1))
        self.prefetching = set()

        # --- Prepare Metadata ---
        self.task_index_to_task = self.meta.tasks
        self.sub_task_index_to_sub_task = self.meta.sub_tasks
        if self.sub_task_index_to_sub_task is None:
            logger.warning(f"'sub_tasks.jsonl' not found in {self.name}. Falling back to task descriptions.")
        self._prepare_camera_keys(ego_name, exo_name)
        self._prepare_episode(do_all_checks=check_episode_integrity) # Scan parquets to build index map
        end_time = time.time()
        logger.info(
            f'Initialized Visual Language Reader for dataset "{self.name}". '
            f'Overhead time: {end_time - start_time:.2f}s'
        )

    def __len__(self) -> int:
        return self.total_frames
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total_frames)
            return [self._get_by_global_index(i) for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self._get_by_global_index(index)
        raise TypeError(f"Invalid index type: {type(index)}. Must be int or slice.")

    def _get_by_global_index(self, global_index: int) -> Dict[str, Any]:
        if not 0 <= global_index < self.total_frames:
            raise IndexError(f"Index {global_index} is out of bounds for dataset with length {self.total_frames}")

        # 1. Find which parquet file and local index this global index corresponds to
        p_idx, local_index = self._find_parquet_for_index(global_index)
        parquet_info = self.parquet_info[p_idx]

        # 2. Get the DataFrame from cache or load it
        self._maybe_prefetch(p_idx)
        df = self.df_cache[p_idx]

        # 3. Get the specific row of data
        row = df.iloc[local_index]
        item = {}
        # Basic indices
        item['frame_index'] = row['frame_index']
        item['task_index'] = row['task_index']
        if 'sub_task_index' in row:
            item['sub_task_index'] = row['sub_task_index']
        item['task'] = self.task_index_to_task[int(row['task_index'])]
        
        # Get task description
        if self.sub_task_index_to_sub_task and int(row['sub_task_index']) in self.sub_task_index_to_sub_task:
            item['sub_task'] = self.sub_task_index_to_sub_task[int(row['sub_task_index'])]
        if 'sub_task' in item and item['sub_task']:
            item['task_description'] = item['sub_task']
        else:
            item['task_description'] = item['task']
    
        # Get ego image
        if self.ego_camera_key:
            item['ego_image'] = self._get_image(
                p_idx, local_index,
                row, self.ego_camera_key, parquet_info[self.ego_camera_key]
            )

        # Get third-person image
        if self.exo_camera_key:
            item['exo_image'] = self._get_image(
                p_idx, local_index,
                row, self.exo_camera_key, parquet_info[self.exo_camera_key]
            )
        
        # Optionally get all other camera images
        for cam_key in self.loaded_camera_keys:
            if cam_key == self.ego_camera_key:
                item[cam_key] = item['ego_image']
            elif cam_key == self.exo_camera_key:
                item[cam_key] = item['exo_image']
            else:
                item[cam_key] = self._get_image(
                    p_idx, local_index,
                    row, cam_key, parquet_info[cam_key]
                )
        
        # meta info
        if self.return_record_meta:
            item['episode_path'] = parquet_info['path']
            item['episode_len'] = parquet_info['num_frames']
            item['chunk_name'] = parquet_info['chunk_name']
            item['episode_name'] = parquet_info['episode_name']
            
        return item

    def _get_image(self, p_idx: int, local_index: int, row: pd.Series, camera_key: str, video_path: Optional[str]) -> Image.Image:
        """Get image from either embedded bytes or video file, with frame-level caching."""
        frame_cache_key = (p_idx, local_index, camera_key)
        if frame_cache_key in self.frame_cache:
            img = self.frame_cache[frame_cache_key]
        else:
            img = self._get_image_basic(row, camera_key, video_path)
        return img

    def _regularize_image(self, img: Image.Image) -> Image.Image:
        """Convert image to RGB and resize if needed."""
        if (self.image_resize != None):
            aim_w, aim_h = self.image_resize
            if aim_w != -1 or aim_h != -1:
                if aim_w == -1:
                    w, h = img.size
                    aim_w = int(w * (aim_h / h))
                elif aim_h == -1:
                    w, h = img.size
                    aim_h = int(h * (aim_w / w))
                if  (img.size != (aim_w, aim_h)):
                    img = img.resize((aim_w, aim_h), Image.Resampling.LANCZOS)
        return img
    
    def _get_video_length(self, video_path: str) -> int:
        try:
            vr = VideoReader(video_path)
            return len(vr)
        except DECORDError:
            pass  # Fallback to av
        with av.open(video_path) as container:
            return container.streams.video[0].frames

    def _iter_video_frames(self, video_path: str):
        # decord
        try:
            vr = VideoReader(video_path)
            for frame in vr:
                yield frame.asnumpy()
            return
        except DECORDError:
            pass
        # PyAV
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.options = {"strict": "experimental"}
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format='rgb24')
            return
        except Exception:
            pass
        # ffmpeg-python fallback
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        w, h = int(video_info["width"]), int(video_info["height"])
        out = (
            ffmpeg
            .input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )[0]
        buf = memoryview(out)
        frame_size = w * h * 3
        for i in range(0, len(buf), frame_size):
            yield np.frombuffer(buf[i:i+frame_size], np.uint8).reshape(h, w, 3)

    def _decode_frame(self, video_path: str, frame_index: int) -> np.ndarray:
        # decord
        try:
            vr = VideoReader(video_path)
            return vr[frame_index].asnumpy()
        except (DECORDError, IndexError):
            pass
        # PyAV
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.options = {"strict": "experimental"}
                for i, frame in enumerate(container.decode(stream)):
                    if i == frame_index:
                        return frame.to_ndarray(format='rgb24')
        except Exception:
            pass
        # ffmpeg-python fallback
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        w, h = int(video_info["width"]), int(video_info["height"])
        fps = eval(video_info["r_frame_rate"])
        out = (
            ffmpeg
            .input(video_path, ss=frame_index / fps)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )[0]
        return np.frombuffer(out, np.uint8).reshape(h, w, 3)

    def _get_image_basic(self, row: pd.Series, camera_key: str, video_path: Optional[str]) -> Image.Image:
        """Decode image from either embedded bytes or video file."""
        if video_path is None:
            img_data = row[camera_key]
            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, HFImage):
                img = img_data.to_pil()
            elif isinstance(img_data, bytes):
                img = Image.open(BytesIO(img_data))
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                img = Image.open(BytesIO(img_data['bytes']))
            else:
                raise TypeError(f"Unsupported embedded image format: {type(img_data)}")
        else:
            frame = self._decode_frame(video_path, int(row['frame_index']))
            img = Image.fromarray(frame)
        return self._regularize_image(img)

    def _find_parquet_for_index(self, global_index: int) -> Tuple[int, int]:
        """Finds the parquet file index and the local index within that file."""
        # `searchsorted` finds the index where the element should be inserted to maintain order.
        # This is exactly what we need to find which "bin" (parquet file) the index falls into.
        p_idx = np.searchsorted(self.cumulative_frames, global_index, side='right')
        
        # Calculate the local index
        if p_idx == 0:
            local_index = global_index
        else:
            local_index = global_index - self.cumulative_frames[p_idx - 1]
            
        return int(p_idx), int(local_index)

    def _load_df_from_parquet(self, parquet_info: Dict[str, Any]) -> pd.DataFrame:
        """Loads only the necessary columns from a parquet file."""
        required_columns = ['frame_index', 'task_index']
        if self.sub_task_index_to_sub_task is not None:
            required_columns.append('sub_task_index')

        pf = ParquetFile(parquet_info['path'])
        column_names = pf.columns
        needs_rename = False
        for key in self.loaded_camera_keys:
            if key in column_names:
                required_columns.append(key)
            elif key + '.bytes' in column_names:
                required_columns.append(key + '.bytes')
                needs_rename = True
            # use video if neither exists

        logger.debug(f"Loading {parquet_info['path']} with columns: {required_columns}")
        df = pf.to_pandas(columns=required_columns)
        if needs_rename:
            df.rename(columns={col: col[:-6] for col in df.columns if col.endswith('.bytes')}, inplace=True)
        return df
    
    # --- Cache or Prefetch Methods ---
    def _ensure_episode_cache(self, p_idx: int):
        try:
            # --- fast-path: if cached, no-op ---
            if p_idx in self.df_cache:
                return

            # --- load df and frames (slow path) ---
            parquet_info = self.parquet_info[p_idx]
            df = self._load_df_from_parquet(parquet_info)
            frames_to_insert = {}

            for cam_key in self.loaded_camera_keys:
                vpath = parquet_info[cam_key]
                if vpath is not None:
                    for local_idx, frame in zip(range(parquet_info["num_frames"]), self._iter_video_frames(vpath)):
                        frames_to_insert[(p_idx, local_idx, cam_key)] = self._regularize_image(Image.fromarray(frame))
                else:
                    for local_idx in range(parquet_info["num_frames"]):
                        row = df.iloc[local_idx]
                        frames_to_insert[(p_idx, local_idx, cam_key)] = self._get_image_basic(row, cam_key, None)

            # --- commit to cache ---
            self.df_cache[p_idx] = df
            self.frame_cache.update(frames_to_insert)
            # evict old
            if len(self.df_cache) > self.cache_num:
                old_p_idx, old_df = self.df_cache.popitem(last=False)
                for cam_key in self.loaded_camera_keys:
                    for local_idx in range(old_df.shape[0]):
                        self.frame_cache.pop((old_p_idx, local_idx, cam_key), None)
        finally:
            self.prefetching.discard(p_idx)
        
    def _maybe_prefetch(self, p_idx: int):
        """ toggle prefetching for next few parquets """
        # submit future prefetched tasks
        for offset in range(1, self.prefetch_num + 1):
            target_idx = p_idx + offset
            if target_idx >= len(self.parquet_info):
                break
            if target_idx not in self.prefetching:
                self.prefetching.add(target_idx)
                self.executor.submit(self._ensure_episode_cache, target_idx)
        # blocking behavior
        while True:
            if p_idx not in self.prefetching:
                self.prefetching.add(p_idx)
                self.executor.submit(self._ensure_episode_cache, p_idx)
            if p_idx in self.df_cache:
                break
            time.sleep(0.01)

    # --- Metadata Preparation Methods (largely from your original code, with fixes) ---
    def _prepare_camera_keys(self, ego_name=None, exo_name=None):
        self.camera_keys = self.meta.camera_keys
        assert len(self.camera_keys) > 0, f"No camera keys found in {self.name}."

        self.loaded_camera_keys = set()
        if self.load_all_camera_keys:
            self.loaded_camera_keys = set(self.camera_keys)

        self.ego_camera_key = ego_name if ego_name in self.camera_keys else None
        if self.ego_camera_key is None:
            for key in self.camera_keys:
                if any(k in key.lower() for k in ['wrist', 'ego', 'interior']):
                    self.ego_camera_key = key
                    self.loaded_camera_keys.add(self.ego_camera_key)
                    break
        else:
            self.loaded_camera_keys.add(self.ego_camera_key)
        
        self.exo_camera_key = exo_name if exo_name in self.camera_keys else None
        if self.exo_camera_key is None:
            for key in self.camera_keys:
                if any(k in key.lower() for k in ['front', 'exterior', 'third']):
                    self.exo_camera_key = key
                    break
            if self.exo_camera_key is None:
                self.exo_camera_key = self.camera_keys[0] # A reasonable fallback
                logger.warning(f"Could not auto-detect third-person camera. Defaulting to '{self.exo_camera_key}'.")
        self.loaded_camera_keys.add(self.exo_camera_key)

        logger.info(f"Using ego camera: '{self.ego_camera_key}', third camera: '{self.exo_camera_key}'")

    def _prepare_episode(self, do_all_checks: bool = False):
        # parquet_paths = load_parquet(self.root)
        # self.check_parquet_integrity(parquet_paths)
        self.parquet_info = []
        cumulative_frames = []
        total_frames = 0
        
        if do_all_checks:
            iterable = tqdm(self.all_episode_paths, desc=f"Checking parquets in {self.name}")
        else:
            iterable = self.all_episode_paths
        for p_str in iterable:
            episode_name = Path(p_str).stem
            chunk_name = Path(p_str).parent.stem
            pf = ParquetFile(p_str)
            num_frames = pf.count()
            total_frames += num_frames
            cumulative_frames.append(total_frames)
            
            column_names = pf.columns
            for i in range(len(column_names)):
                if column_names[i].endswith('.bytes'):
                    column_names[i] = column_names[i][:-6]
            info = {'path': p_str, 'num_frames': num_frames, 'columns': column_names, 'chunk_name': chunk_name, 'episode_name': episode_name}
            
            def check_video(camera_key):
                    if not camera_key or camera_key in column_names:
                        return None
                    video_path = self.root / "videos" / chunk_name / camera_key / f"{episode_name}.mp4"
                    assert video_path.exists(), f"Parquet {p_str} requires video {video_path}, but it was not found."
                    if do_all_checks:
                        video_len = self._get_video_length(str(video_path))
                        if video_len < num_frames:
                            raise RuntimeError(f"Video {video_path} has fewer frames ({video_len}) than parquet {p_str} ({num_frames}).")
                        elif video_len > num_frames:
                            logger.warning(f"Video {video_path} has more frames ({video_len}) than parquet {p_str} ({num_frames}). Using video.")
                    return str(video_path)

            if self.load_all_camera_keys:
                for cam_key in self.camera_keys:
                    info[cam_key] = check_video(cam_key)
            else:
                info[self.ego_camera_key] =  check_video(self.ego_camera_key)
                info[self.exo_camera_key] = check_video(self.exo_camera_key)

            self.parquet_info.append(info)

        self.total_frames = total_frames
        self.cumulative_frames = np.array(cumulative_frames, dtype=np.int64)
        logger.info(f"Found {len(self.all_episode_paths)} parquet files in {self.name}, total frames: {self.total_frames}")
    
    # --- Functions ---
    @property
    def sub_task_available(self) -> bool:
        return self.sub_task_index_to_sub_task is not None

    @property
    def ego_camera_available(self) -> bool:
        return self.ego_camera_key is not None

    @property
    def all_episode_indices(self) -> List[int]:
        return sorted(self.meta.episodes.keys())

    def episode_index_to_path(self, ep_index: int) -> Path:
        """
        Convert episode index to full parquet path.
        """
        chunk_index = self.meta.get_episode_chunk(ep_index)
        parquet_path = self.root / self.meta.data_path.format(episode_chunk=chunk_index, episode_index=ep_index)
        return parquet_path

    def episode_string_to_index(self, ep_str: str) -> int:
        """ 
        Convert episode string description to episode index. 
        This string can either be a full path or just the episode file name. 
        """
        ep_path = Path(ep_str)
        if ep_path.suffix == '.parquet':
            ep_name = ep_path.stem
        else:
            ep_name = ep_str
        match = re.search(r'episode_(\d+)$', ep_name)
        if not match:
            raise ValueError(f"Invalid episode string description format: {ep_str}")
        return int(match.group(1))
    
    def get_episode_range(self, ep_str: str) -> Tuple[int, int]:
        """ 
        Convert episode string description to (start_index, end_index) tuple. 
        This string can either be a full path or just the episode file name. 
        """
        ep_name = Path(ep_str).stem if Path(ep_str).suffix == '.parquet' else ep_str
        for p_idx, info in enumerate(self.parquet_info):
            if info["episode_name"] == ep_name:
                start_index = self.cumulative_frames[p_idx - 1] if p_idx > 0 else 0
                end_index = self.cumulative_frames[p_idx]
                return int(start_index), int(end_index)
        raise ValueError(f"Episode '{ep_str}' not found in the dataset.")

    @property
    def all_episode_paths(self) -> List[str]:
        parquet_paths = []
        for ep_index in self.all_episode_indices:
            parquet_path = self.episode_index_to_path(ep_index)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Expected parquet file {parquet_path} does not exist.")
            parquet_paths.append(str(parquet_path))
        return parquet_paths
    
    @property
    def all_episode_names(self) -> List[str]:
        return [Path(p).stem for p in self.all_episode_paths]

    def set_skip_episode(self, skip_names: List[str]):
        """
        Skip the specified episodes, given by their names (Path.stem), remove their parquet info
        """
        # clean parquet_info
        new_parquet_info = []
        skipped_names = []
        for info in self.parquet_info:
            if info["episode_name"] in skip_names:
                skipped_names.append(info["episode_name"])
            else:
                new_parquet_info.append(info)
        self.parquet_info = new_parquet_info

        # recompute cumulative_frames and total_frames
        cumulative_frames = []
        total_frames = 0
        for info in self.parquet_info:
            total_frames += info["num_frames"]
            cumulative_frames.append(total_frames)
        self.total_frames = total_frames
        self.cumulative_frames = np.array(cumulative_frames, dtype=np.int64)

        logger.info(
            f"Skipped {len(skipped_names)} episodes. "
            f"Remaining {len(self.parquet_info)} parquet files, total frames: {self.total_frames}."
        )
    
    def set_skip_episode_by_index(self, skip_indices: List[int]):
        """
        Skip the specified episodes, given by their index, remove their parquet info
        """
        ep_paths = [self.episode_index_to_path(idx) for idx in skip_indices]
        skip_names = {p.stem for p in ep_paths}
        self.set_skip_episode(skip_names)


import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_paths", type=str, nargs='+', help="List of dataset paths to process.", required=True)
    args = argparser.parse_args()
    for dataset_path in args.dataset_paths:
        dataset = FastLerobotVLReader(
            root=dataset_path,
            load_all_camera_keys=True,
            check_episode_integrity=True,
        )
