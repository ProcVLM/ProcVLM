import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

from evqa.model import batch_chat_with_value_head
from core.utils.common import images_to_video


PROGRESS_PATTERN = re.compile(
    r"<progress>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</progress>",
    re.IGNORECASE,
)

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


@dataclass(frozen=True)
class PlotBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class VizStyle:
    bg: tuple[int, int, int] = (255, 255, 255)
    panel_bg: tuple[int, int, int] = (250, 252, 250)
    panel_border: tuple[int, int, int] = (230, 235, 230)
    text: tuple[int, int, int] = (24, 24, 24)
    muted: tuple[int, int, int] = (105, 105, 105)
    grid: tuple[int, int, int] = (224, 228, 224)

    # Green style adapted from make_case_study_figures.py.
    curve: tuple[int, int, int] = (92, 154, 113)
    curve_fill: tuple[int, int, int] = (224, 241, 230)
    accent: tuple[int, int, int] = (82, 142, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize frame-wise progress prediction: for each sampled frame, run model inference "
            "with a multi-image window, then render a video with a bottom progress/text panel."
        )
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--task", type=str, required=True, help="Task instruction replacing {task} in procedural_c")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_path", type=str, default=None, help="Output visualization video path")

    parser.add_argument("--torch_dtype", type=str, default="bf16", help="Torch dtype")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max generation tokens per frame")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--video_codec", type=str, default="libx264", help="Preferred output video codec")
    parser.add_argument("--window_size", type=int, default=8, help="Number of images per inference window")

    parser.add_argument("--frame_dir", type=str, default=None, help="Directory to store extracted frames")
    parser.add_argument("--keep_frames", action="store_true", help="Keep extracted frames after completion")
    parser.add_argument("--max_frames", type=int, default=None, help="Only process first N frames")
    parser.add_argument(
        "--max_sampled_frames",
        type=int,
        default=512,
        help="Maximum number of frames used for inference/visualization, uniformly sampled",
    )
    parser.add_argument(
        "--enable_value_head",
        action="store_true",
        default=False,
        help="Whether to enable value head for progress prediction",
    )

    parser.add_argument("--min_video_width", type=int, default=960, help="Upscale video frames to at least this width")
    parser.add_argument("--min_video_height", type=int, default=540, help="Upscale video frames to at least this height")

    parser.add_argument(
        "--bottom_panel_ratio",
        type=float,
        default=0.32,
        help="Bottom panel height as a ratio of the resized video frame height",
    )
    parser.add_argument(
        "--plot_width_ratio",
        type=float,
        default=0.67,
        help="Progress plot width ratio inside the bottom panel",
    )
    parser.add_argument(
        "--hide_model_output",
        dest="show_model_output",
        action="store_false",
        help="Hide model output text in the bottom panel",
    )
    parser.set_defaults(show_model_output=True)

    return parser.parse_args()


def load_prompt_template(task_text: str) -> str:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "procedural_c.txt"
    template = template_path.read_text(encoding="utf-8").strip()
    return template.replace("{task}", task_text)


def extract_progress(text: str) -> float | None:
    match = PROGRESS_PATTERN.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    return float(max(0.0, min(100.0, value)))


def extract_reasoning_text(answer: str) -> str:
    """Remove the final <progress> sentence and keep the visible reasoning text."""
    if not answer:
        return ""

    raw = str(answer).strip()

    raw = re.sub(
        r"[^.\n]*<progress>\s*[+-]?\d+(?:\.\d+)?\s*%?\s*</progress>[^.\n]*\.?",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    kept_lines: list[str] = []
    for line in raw.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if re.match(r"^(therefore|thus|hence|so)\b", clean, flags=re.IGNORECASE):
            continue
        kept_lines.append(clean)

    return "\n".join(kept_lines).strip()


def build_window_paths(frame_paths: List[str], frame_idx: int, window_size: int) -> List[str]:
    start = frame_idx - window_size + 1
    paths: List[str] = []
    for i in range(window_size):
        idx = start + i
        idx = max(0, min(frame_idx, idx))
        paths.append(frame_paths[idx])
    return paths


def build_uniform_indices(total_frames: int, target_count: int) -> List[int]:
    if total_frames <= 0 or target_count <= 0:
        return []
    if target_count >= total_frames:
        return list(range(total_frames))
    if target_count == 1:
        return [0]

    vals = np.linspace(0, total_frames - 1, num=target_count)
    idx = np.round(vals).astype(np.int64).tolist()

    uniq: List[int] = []
    seen = set()
    for x in idx:
        if x not in seen:
            seen.add(x)
            uniq.append(int(x))

    cur = 0
    while len(uniq) < target_count and cur < total_frames:
        if cur not in seen:
            seen.add(cur)
            uniq.append(cur)
        cur += 1

    uniq.sort()
    return uniq[:target_count]


def resize_keep_aspect_min(img: Image.Image, min_width: int, min_height: int) -> Image.Image:
    if min_width <= 0 and min_height <= 0:
        return img

    w, h = img.size
    scale = max(
        min_width / max(w, 1) if min_width > 0 else 1.0,
        min_height / max(h, 1) if min_height > 0 else 1.0,
        1.0,
    )
    if scale <= 1.001:
        return img

    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, RESAMPLE_LANCZOS)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    repo_root = Path(__file__).resolve().parents[2]
    font_dir = repo_root / "assets" / "fonts"

    candidates = [
        font_dir / ("Ubuntu-Bold.ttf" if bold else "Ubuntu-Regular.ttf"),
        font_dir / "Ubuntu-Regular.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]

    for path in candidates:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text_centered_on_y(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    anchor_right: bool = False,
) -> None:
    w, h = text_size(draw, text, font)
    x = xy[0] - w if anchor_right else xy[0]
    y = xy[1] - h / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    w, h = text_size(draw, text, font)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def nice_step(span: float, target_ticks: int) -> float:
    raw = span / max(1, target_ticks)
    if raw <= 0:
        return 1.0

    exponent = math.floor(math.log10(raw))
    base = raw / (10**exponent)

    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10

    return nice * (10**exponent)


def tick_values(vmin: float, vmax: float, target_ticks: int) -> list[float]:
    step = nice_step(vmax - vmin, target_ticks)
    start = math.ceil(vmin / step) * step

    ticks = []
    value = start
    while value <= vmax + step * 0.25:
        ticks.append(round(value, 10))
        value += step

    return ticks


def format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def polyline_points(
    values: np.ndarray,
    frame_indices: np.ndarray,
    box: PlotBox,
    x_min: float,
    x_max: float,
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> list[tuple[float, float]]:
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    points = []
    for x_value, y_value in zip(frame_indices, values):
        x = box.left + ((float(x_value) - x_min) / x_span) * box.width
        y = box.bottom - ((float(y_value) - y_min) / y_span) * box.height
        points.append((x, y))

    return points


def draw_progress_plot_pil(
    draw: ImageDraw.ImageDraw,
    progress_history: list[float],
    frame_index_history: list[int],
    box: PlotBox,
    current_progress: float,
    current_frame_idx: int,
    style: VizStyle,
) -> None:
    title_font = load_font(max(22, box.height // 7), bold=True)
    axis_font = load_font(max(15, box.height // 11), bold=False)
    small_font = load_font(max(14, box.height // 13), bold=False)

    header_y = box.top
    plot_top = box.top + max(42, int(box.height * 0.22))
    plot_box = PlotBox(
        left=box.left + max(58, int(box.width * 0.08)),
        top=plot_top,
        right=box.right - max(14, int(box.width * 0.02)),
        bottom=box.bottom - max(34, int(box.height * 0.16)),
    )

    draw.text(
        (box.left, header_y),
        f"Progress {current_progress:.1f}%",
        font=title_font,
        fill=style.text,
    )

    frame_text = f"Frame {current_frame_idx}"
    fw, _ = text_size(draw, frame_text, small_font)
    draw.text(
        (box.right - fw, header_y + 4),
        frame_text,
        font=small_font,
        fill=style.muted,
    )

    x_min = 0
    x_max = max(1, current_frame_idx)
    points: list[tuple[float, float]] = []

    if progress_history:
        values = np.asarray(progress_history, dtype=np.float32)
        frames = np.asarray(frame_index_history, dtype=np.float32)
        points = polyline_points(
            values=values,
            frame_indices=frames,
            box=plot_box,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=0.0,
            y_max=100.0,
        )

        # Light green area fill under the curve.
        if len(points) >= 2:
            fill_poly = [
                (points[0][0], plot_box.bottom),
                *points,
                (points[-1][0], plot_box.bottom),
            ]
            draw.polygon(fill_poly, fill=style.curve_fill)

    # Horizontal grid only: 0 / 25 / 50 / 75 / 100.
    for tick in [0, 25, 50, 75, 100]:
        y = plot_box.bottom - (tick / 100.0) * plot_box.height
        draw.line(
            (plot_box.left, y, plot_box.right, y),
            fill=style.grid,
            width=1,
        )
        draw_text_centered_on_y(
            draw,
            (plot_box.left - 10, y),
            str(tick),
            axis_font,
            style.muted,
            anchor_right=True,
        )

    # X axis: only start and current frame.
    x_tick_y = plot_box.bottom + 10
    for x_value in [x_min, x_max]:
        x = plot_box.left + ((x_value - x_min) / max(x_max - x_min, 1e-6)) * plot_box.width
        tick_text = str(int(x_value))
        tw, _ = text_size(draw, tick_text, axis_font)
        draw.text((x - tw / 2, x_tick_y), tick_text, font=axis_font, fill=style.muted)

    if points:
        if len(points) >= 2:
            draw.line(
                points,
                fill=style.curve,
                width=max(3, box.height // 46),
                joint="curve",
            )
        else:
            x, y = points[0]
            r = max(2, box.height // 60)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=style.curve)

        # Current point.
        x, y = points[-1]
        r = max(4, box.height // 40)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=style.accent)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=2)

    label_font = load_font(max(15, box.height // 12), bold=False)
    draw_centered_text(
        draw,
        ((plot_box.left + plot_box.right) / 2, box.bottom - 6),
        "Frame index",
        label_font,
        style.muted,
    )

def wrap_paragraph_to_lines(
    draw: ImageDraw.ImageDraw,
    para: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    para = para.rstrip()
    if not para:
        return [""]

    words = para.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = word
        else:
            # Very long token: split by character.
            piece = ""
            for ch in word:
                candidate_piece = piece + ch
                bbox = draw.textbbox((0, 0), candidate_piece, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    piece = candidate_piece
                else:
                    if piece:
                        lines.append(piece)
                    piece = ch
            current = piece

    if current:
        lines.append(current)

    return lines

def wrap_text_strict(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> tuple[list[str], bool]:
    raw_paragraphs = str(text).splitlines()
    if not raw_paragraphs:
        return [], True

    lines: list[str] = []
    fits = True

    for para_idx, para in enumerate(raw_paragraphs):
        para_lines = wrap_paragraph_to_lines(draw, para, font, max_width)

        for line in para_lines:
            if len(lines) >= max_lines:
                fits = False
                return lines, fits
            lines.append(line)

        # Preserve explicit blank lines, but avoid appending one after the last paragraph.
        if para_idx < len(raw_paragraphs) - 1 and para.strip() == "":
            if len(lines) >= max_lines:
                fits = False
                return lines, fits
            lines.append("")

    return lines, fits


def truncate_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text

    text = text.rstrip()
    while text:
        candidate = text + "..."
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return candidate
        text = text[:-1].rstrip()

    return "..."


def fit_output_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    initial_size: int,
    min_size: int = 12,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    text = text.strip()
    if not text:
        font = load_font(initial_size, bold=False)
        return font, [], initial_size + 4

    for size in range(initial_size, min_size - 1, -1):
        font = load_font(size, bold=False)
        line_height = int(size * 1.35)
        max_lines = max(1, max_height // max(1, line_height))
        lines, fits = wrap_text_strict(draw, text, font, max_width, max_lines)

        if fits and lines:
            return font, lines, line_height

    font = load_font(min_size, bold=False)
    line_height = int(min_size * 1.35)
    max_lines = max(1, max_height // max(1, line_height))
    lines, _ = wrap_text_strict(draw, text, font, max_width, max_lines)
    lines = lines[:max_lines] or [""]
    lines[-1] = truncate_to_width(draw, lines[-1], font, max_width)

    return font, lines, line_height


def draw_output_text_panel(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: PlotBox,
    style: VizStyle,
) -> None:
    title = "Reasoning"
    title_font = load_font(max(19, box.height // 8), bold=True)

    draw.text(
        (box.left, box.top),
        title,
        font=title_font,
        fill=style.text,
    )

    title_h = text_size(draw, title, title_font)[1]
    body_top = box.top + title_h + max(8, box.height // 20)
    body_h = max(1, box.bottom - body_top)

    if not text.strip():
        text = "[no reasoning available]"

    font, lines, line_height = fit_output_text(
        draw=draw,
        text=text,
        max_width=box.width,
        max_height=body_h,
        initial_size=max(17, box.height // 9),
        min_size=14,
    )

    y = body_top
    for line in lines:
        draw.text((box.left, y), line, font=font, fill=style.text)
        y += line_height


def compose_frame_with_bottom_panel(
    frame_img: Image.Image,
    progress_history: list[float],
    frame_index_history: list[int],
    current_progress: float,
    current_frame_idx: int,
    reasoning_text: str,
    show_model_output: bool,
    bottom_panel_ratio: float,
    plot_width_ratio: float,
    style: VizStyle | None = None,
) -> Image.Image:
    style = style or VizStyle()

    frame = frame_img.convert("RGB")
    w, h = frame.size
    panel_h = max(190, int(h * bottom_panel_ratio))

    out = Image.new("RGB", (w, h + panel_h), style.bg)
    out.paste(frame, (0, 0))

    draw = ImageDraw.Draw(out)

    panel_top = h
    panel_bottom = h + panel_h

    draw.rectangle((0, panel_top, w, panel_bottom), fill=style.panel_bg)
    draw.line((0, panel_top, w, panel_top), fill=style.panel_border, width=1)

    margin_x = max(22, int(w * 0.028))
    margin_y = max(18, int(panel_h * 0.12))
    gap = max(22, int(w * 0.025))

    if show_model_output:
        plot_width_ratio = max(0.58, min(0.78, plot_width_ratio))
        plot_w = int((w - 2 * margin_x - gap) * plot_width_ratio)
        text_w = w - 2 * margin_x - gap - plot_w

        plot_box = PlotBox(
            left=margin_x,
            top=panel_top + margin_y,
            right=margin_x + plot_w,
            bottom=panel_bottom - margin_y,
        )
        text_box = PlotBox(
            left=plot_box.right + gap,
            top=panel_top + margin_y,
            right=w - margin_x,
            bottom=panel_bottom - margin_y,
        )

        draw.line(
            (text_box.left - gap // 2, panel_top + margin_y, text_box.left - gap // 2, panel_bottom - margin_y),
            fill=(232, 236, 232),
            width=1,
        )

        draw_progress_plot_pil(
            draw=draw,
            progress_history=progress_history,
            frame_index_history=frame_index_history,
            box=plot_box,
            current_progress=current_progress,
            current_frame_idx=current_frame_idx,
            style=style,
        )
        draw_output_text_panel(
            draw=draw,
            text=reasoning_text,
            box=text_box,
            style=style,
        )
    else:
        plot_box = PlotBox(
            left=margin_x,
            top=panel_top + margin_y,
            right=w - margin_x,
            bottom=panel_bottom - margin_y,
        )
        draw_progress_plot_pil(
            draw=draw,
            progress_history=progress_history,
            frame_index_history=frame_index_history,
            box=plot_box,
            current_progress=current_progress,
            current_frame_idx=current_frame_idx,
            style=style,
        )

    return out


def extract_frames_torchvision(video_path: str, frame_dir: Path) -> tuple[List[str], float]:
    """Extract frames using moviepy for broader environment compatibility."""
    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_paths: List[str] = []
        with VideoFileClip(video_path) as clip:
            fps = float(clip.fps) if clip.fps and clip.fps > 0 else 30.0
            for frame_idx, frame_np in enumerate(clip.iter_frames()):
                img = Image.fromarray(frame_np)
                frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
                img.save(str(frame_path), quality=95)
                frame_paths.append(str(frame_path))

        if not frame_paths:
            raise RuntimeError("No frames extracted by moviepy")

        return frame_paths, fps
    except Exception as e:
        raise RuntimeError(f"Failed to extract frames via moviepy: {e}") from e


def write_video_moviepy(
    output_path: Path,
    video_frames: List[Image.Image],
    fps: float,
    preferred_codec: str,
) -> str:
    if not video_frames:
        raise RuntimeError("No rendered frames to encode")

    images_to_video(
        images=video_frames,
        output_path=output_path,
        fps=max(1, int(round(float(fps)))),
        show_log=False,
    )
    return preferred_codec or "libx264"


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else (
        video_path.with_name(f"{video_path.stem}_progress_vis.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = Path(args.frame_dir).expanduser().resolve() if args.frame_dir else (
        output_path.parent / f"{video_path.stem}_frames"
    )

    prompt = load_prompt_template(args.task)

    print("Extracting frames using moviepy...")
    all_frame_paths, fps = extract_frames_torchvision(str(video_path), frame_dir)
    total_frames = len(all_frame_paths)

    source_frames = total_frames
    if args.max_frames is not None:
        source_frames = min(source_frames, max(args.max_frames, 0))

    max_sampled = max(int(args.max_sampled_frames), 1)
    sampled_count = min(source_frames, max_sampled)
    selected_indices = build_uniform_indices(source_frames, sampled_count)
    if not selected_indices:
        raise RuntimeError("no frames selected for processing")

    # Keep the output duration close to source duration after downsampling.
    duration_scale = sampled_count / max(source_frames, 1)
    output_fps = max(1.0, float(fps) * duration_scale)

    print(
        f"Frame policy: source={source_frames}, sampled={sampled_count}, "
        f"max_sampled={max_sampled}, output_fps={output_fps:.3f}"
    )

    selected_frame_paths = [all_frame_paths[i] for i in selected_indices]

    frame_paths: List[str] = []
    all_items = []
    all_answers: List[str] = []
    all_progress: List[float] = []

    pbar = tqdm(total=len(selected_frame_paths), desc="Preparing frames")
    try:
        for output_frame_idx, frame_path in enumerate(selected_frame_paths):
            if not Path(frame_path).exists():
                print(f"  Warning: frame missing {frame_path}, skipping")
                pbar.update(1)
                continue

            frame_paths.append(frame_path)
            window_paths = build_window_paths(frame_paths, output_frame_idx, max(1, args.window_size))
            all_items.append(
                {
                    "image": window_paths,
                    "conversations": [{"from": "human", "value": prompt}],
                }
            )

            pbar.update(1)

        pbar.close()

        if not frame_paths:
            raise RuntimeError("No frames available for processing")

        print("Running batched model inference...")
        all_answers = batch_chat_with_value_head(
            batch_items=all_items,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            torch_dtype=args.torch_dtype,
            enable_value_head=args.enable_value_head,
        )

        last_progress = 0.0
        for answer in all_answers:
            progress = extract_progress(answer)
            if progress is None:
                progress = last_progress
            else:
                last_progress = progress
            all_progress.append(progress)

        render_pbar = tqdm(total=len(frame_paths), desc="Rendering and encoding")
        progress_history: List[float] = []
        frame_index_history: List[int] = []
        video_frames: List[Image.Image] = []

        for frame_idx, frame_path in enumerate(frame_paths):
            frame_img = Image.open(frame_path).convert("RGB")
            frame_img = resize_keep_aspect_min(
                frame_img,
                min_width=args.min_video_width,
                min_height=args.min_video_height,
            )

            source_frame_idx = selected_indices[frame_idx]
            progress = all_progress[frame_idx] if frame_idx < len(all_progress) else 0.0

            progress_history.append(progress)
            frame_index_history.append(source_frame_idx)

            reasoning_text = ""
            if frame_idx < len(all_answers):
                reasoning_text = extract_reasoning_text(all_answers[frame_idx])

            rendered = compose_frame_with_bottom_panel(
                frame_img=frame_img,
                progress_history=progress_history,
                frame_index_history=frame_index_history,
                current_progress=progress,
                current_frame_idx=source_frame_idx,
                reasoning_text=reasoning_text,
                show_model_output=args.show_model_output,
                bottom_panel_ratio=args.bottom_panel_ratio,
                plot_width_ratio=args.plot_width_ratio,
            )

            video_frames.append(rendered)
            render_pbar.update(1)

        render_pbar.close()

        print("Writing video using moviepy/common.py pipeline...")
        used_codec = write_video_moviepy(
            output_path=output_path,
            video_frames=video_frames,
            fps=float(output_fps),
            preferred_codec=args.video_codec,
        )
        print(f"Video encoded with codec: {used_codec}")

    finally:
        if "pbar" in locals() and not pbar.disable and pbar.n < pbar.total:
            pbar.close()

    if not args.keep_frames:
        shutil.rmtree(frame_dir, ignore_errors=True)

    print(f"Done. Output video: {output_path}")


if __name__ == "__main__":
    main()