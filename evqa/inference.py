from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


PROGRESS_PATTERN = re.compile(
    r"<progress>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</progress>",
    re.IGNORECASE,
)

VLLM_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ProcVLM progress reward inference on sampled video frames and "
            "save frame-wise outputs to a JSONL file."
        )
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--task", type=str, required=True, help="Task instruction replacing {task} in procedural_c")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSONL path")

    parser.add_argument("--torch_dtype", type=str, default="bf16", help="Torch dtype")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max generation tokens per frame")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--window_size", type=int, default=8, help="Number of images per inference window")
    parser.add_argument("--tp", type=int, default=1, help="Parallel degree for model inference")

    parser.add_argument("--frame_dir", type=str, default=None, help="Directory to store extracted frames")
    parser.add_argument("--keep_frames", action="store_true", help="Keep extracted frames after completion")
    parser.add_argument("--max_frames", type=int, default=None, help="Only process first N source frames")
    parser.add_argument(
        "--max_sampled_frames",
        type=int,
        default=512,
        help="Maximum number of frames used for inference, uniformly sampled",
    )
    parser.add_argument(
        "--enable_value_head",
        action="store_true",
        default=False,
        help="Whether to enable value head for progress prediction",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for model inference",
    )

    return parser.parse_args()


def load_prompt_template(task_text: str) -> str:
    template_path = Path(__file__).resolve().parent / "templates" / "procedural_c.txt"
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


def build_window_values(values: list[Any], frame_idx: int, window_size: int) -> list[Any]:
    start = frame_idx - window_size + 1
    window: list[Any] = []
    for i in range(window_size):
        idx = start + i
        idx = max(0, min(frame_idx, idx))
        window.append(values[idx])
    return window


def build_uniform_indices(total_frames: int, target_count: int) -> list[int]:
    if total_frames <= 0 or target_count <= 0:
        return []
    if target_count >= total_frames:
        return list(range(total_frames))
    if target_count == 1:
        return [0]

    vals = np.linspace(0, total_frames - 1, num=target_count)
    idx = np.round(vals).astype(np.int64).tolist()

    uniq: list[int] = []
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


def extract_frames_moviepy(video_path: str, frame_dir: Path) -> tuple[list[str], float]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_paths: list[str] = []
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


def make_output_record(
    *,
    video_path: Path,
    task: str,
    fps: float,
    sample_index: int,
    frame_index: int,
    window_frame_indices: list[int],
    answer: str,
    parsed_progress: float | None,
    progress: float,
) -> dict[str, Any]:
    return {
        "video_path": str(video_path),
        "task": task,
        "sample_index": sample_index,
        "frame_index": frame_index,
        "timestamp_sec": frame_index / max(float(fps), 1e-6),
        "window_frame_indices": window_frame_indices,
        "progress": progress,
        "parsed_progress": parsed_progress,
        "progress_source": "model" if parsed_progress is not None else "previous",
        "reasoning": extract_reasoning_text(answer),
        "model_output": answer,
    }


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_vllm_engine_kwargs(
    torch_dtype: str,
    engine_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kwargs = dict(engine_kwargs or {})
    if "dtype" not in kwargs:
        kwargs["dtype"] = VLLM_DTYPE_ALIASES.get(str(torch_dtype).lower(), torch_dtype)
    return kwargs


def run_batch_progress_inference(
    *,
    batch_items: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    enable_value_head: bool = False,
    use_lora: bool = False,
    torch_dtype: str = "bf16",
    tp: int = 1,
    sampling_kwargs: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Run model inference for already-prepared progress-reward batch items."""
    if enable_value_head or use_lora:
        from evqa.model import batch_chat_with_value_head

        return batch_chat_with_value_head(
            batch_items=batch_items,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            dp=tp,
            enable_value_head=True,
        )

    from evqa.model import batch_chat_with_vllm

    return batch_chat_with_vllm(
        batch_items=batch_items,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        tp=tp,
        sampling_kwargs=sampling_kwargs,
        engine_kwargs=build_vllm_engine_kwargs(torch_dtype, engine_kwargs),
    )


def infer_progress_from_video(
    *,
    video_path: str | Path,
    task: str,
    model_path: str,
    output_path: str | Path | None = None,
    window_size: int = 8,
    torch_dtype: str = "bf16",
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    frame_dir: str | Path | None = None,
    keep_frames: bool = False,
    max_frames: int | None = None,
    max_sampled_frames: int = 512,
    enable_value_head: bool = False,
    use_lora: bool = False,
    tp: int = 1,
    sampling_kwargs: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Infer frame-wise progress rewards from a video.

    Returns one record for each selected ``frame_index``. If ``output_path`` is
    provided, the same records are also written as JSONL.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    resolved_output_path = Path(output_path).expanduser().resolve() if output_path else None
    frame_root = resolved_output_path.parent if resolved_output_path else video_path.parent
    frame_dir = Path(frame_dir).expanduser().resolve() if frame_dir else (
        frame_root / f"{video_path.stem}_frames"
    )

    prompt = load_prompt_template(task)

    print("Extracting frames using moviepy...")
    all_frame_paths, fps = extract_frames_moviepy(str(video_path), frame_dir)
    total_frames = len(all_frame_paths)

    source_frames = total_frames
    if max_frames is not None:
        source_frames = min(source_frames, max(max_frames, 0))

    max_sampled = max(int(max_sampled_frames), 1)
    sampled_count = min(source_frames, max_sampled)
    selected_indices = build_uniform_indices(source_frames, sampled_count)
    if not selected_indices:
        raise RuntimeError("no frames selected for processing")

    print(
        f"Frame policy: source={source_frames}, sampled={sampled_count}, "
        f"max_sampled={max_sampled}, fps={fps:.3f}"
    )

    selected_frame_paths = [all_frame_paths[i] for i in selected_indices]

    frame_paths: list[str] = []
    frame_indices: list[int] = []
    all_items: list[dict[str, Any]] = []
    window_frame_indices_list: list[list[int]] = []
    records: list[dict[str, Any]] = []

    pbar = tqdm(total=len(selected_frame_paths), desc="Preparing frames", disable=not show_progress)
    try:
        for sample_index, frame_path in enumerate(selected_frame_paths):
            if not Path(frame_path).exists():
                print(f"  Warning: frame missing {frame_path}, skipping")
                pbar.update(1)
                continue

            frame_paths.append(frame_path)
            frame_indices.append(selected_indices[sample_index])

            window_paths = build_window_values(frame_paths, len(frame_paths) - 1, max(1, window_size))
            window_frame_indices = build_window_values(frame_indices, len(frame_indices) - 1, max(1, window_size))
            window_frame_indices_list.append(window_frame_indices)
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
        all_answers = run_batch_progress_inference(
            batch_items=all_items,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_value_head=enable_value_head,
            use_lora=use_lora,
            torch_dtype=torch_dtype,
            tp=tp,
            sampling_kwargs=sampling_kwargs,
            engine_kwargs=engine_kwargs,
        )
        if len(all_answers) != len(frame_indices):
            raise RuntimeError(
                f"model returned {len(all_answers)} answers for {len(frame_indices)} prepared frames"
            )

        last_progress = 0.0
        for sample_index, answer in enumerate(all_answers):
            parsed_progress = extract_progress(answer)
            if parsed_progress is None:
                progress = last_progress
            else:
                progress = parsed_progress
                last_progress = progress

            records.append(
                make_output_record(
                    video_path=video_path,
                    task=task,
                    fps=fps,
                    sample_index=sample_index,
                    frame_index=frame_indices[sample_index],
                    window_frame_indices=window_frame_indices_list[sample_index],
                    answer=answer,
                    parsed_progress=parsed_progress,
                    progress=progress,
                )
            )

        if resolved_output_path is not None:
            write_jsonl(records, resolved_output_path)

    finally:
        if "pbar" in locals() and not pbar.disable and pbar.n < pbar.total:
            pbar.close()
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)

    return records


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else (
        video_path.with_name(f"{video_path.stem}_progress.jsonl")
    )

    records = infer_progress_from_video(
        video_path=video_path,
        task=args.task,
        model_path=args.model_path,
        output_path=output_path,
        window_size=args.window_size,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        frame_dir=args.frame_dir,
        keep_frames=args.keep_frames,
        max_frames=args.max_frames,
        max_sampled_frames=args.max_sampled_frames,
        enable_value_head=args.enable_value_head,
        tp=args.tp,
        use_lora=args.use_lora,
    )

    print(f"Done. Wrote {len(records)} records to: {output_path}")


if __name__ == "__main__":
    main()
