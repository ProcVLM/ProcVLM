"""
python evqa/one-shot/annotator.py \
    --video_path path/to/your/video.mp4 \
    --task "Task description for the video (e.g., 'Make a sandwich')" \
    --data_output_dir path/to/save/
"""

import argparse
import io
import os
import threading
import time
import av
import random
from typing import List, Dict
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)

GLOBAL_FRAMES = []
DATA_OUTPUT_DIR = ""
TASK_DESCRIPTION = ""
SAMPLE_TAG = ""


def decode_video(video_path):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
    return frames


def load_html_template():
    """Load HTML from the external file located in the same directory."""
    current_dir = Path(__file__).parent
    html_path = current_dir / "annotator.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Template file '{html_path}' not found. Ensure it is in the same directory.")
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/')
def index():
    html_template = load_html_template()
    return render_template_string(
        html_template,
        total_frames=len(GLOBAL_FRAMES),
        task_description=TASK_DESCRIPTION,
    )


@app.route('/frame/<int:idx>')
def get_frame(idx):
    if 0 <= idx < len(GLOBAL_FRAMES):
        img_io = io.BytesIO()
        GLOBAL_FRAMES[idx].save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    return "Frame not found", 404


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    sub_tasks = data.get('sub_tasks', [])
    completion_reason = data.get('reason', "")
    is_done = data.get('is_done', False)
    success_cutoff = data.get('success_cutoff', 0.75)

    final_frames = []
    task_idx = 0

    for i, img in enumerate(GLOBAL_FRAMES):
        if i > sub_tasks[task_idx][1]:
            task_idx += 1

        desc = sub_tasks[task_idx][0]

        # Skip deleted segments
        if desc.strip() == '/delete':
            continue

        final_frames.append({
            'exo_image': img,
            'sub_task': desc
        })

    threading.Timer(
        1.0,
        lambda: finalize_and_exit(
            final_frames,
            DATA_OUTPUT_DIR,
            TASK_DESCRIPTION,
            completion_reason,
            is_done,
            SAMPLE_TAG,
            success_cutoff,
        ),
    ).start()
    return jsonify({"status": "success"})


def ensure_image_on_disk(data_output_dir: Path, image_root: Path, frame_id: int, image_data: Image) -> str:
    possible_path = Path(image_root) / f"{frame_id:06d}.jpg"
    if possible_path.exists():
        return str(possible_path.relative_to(data_output_dir))

    save_path = possible_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_data.save(save_path)
    return str(save_path.relative_to(data_output_dir))


_cached_templates: Dict[str, List[str]] = {}


def load_templates(template_name: str) -> List[str]:
    global _cached_templates
    if template_name in _cached_templates:
        return _cached_templates[template_name]

    template_path = Path(f"evqa/templates/{template_name}.txt")
    if not template_path.exists():
        raise FileNotFoundError(f"Template '{template_path}' not found.")

    with open(template_path, 'r') as f:
        templates = [line.strip() for line in f if line.strip()]

    _cached_templates[template_name] = templates
    return templates


def sample_previous_frames(cur_id: int, single_frame_P: float = 0.3, min_frames: int = 4, max_frames: int = 16) -> List[int]:
    if random.random() < single_frame_P:
        return [cur_id]

    num_frames = random.randint(min_frames, max_frames)
    start_id = max(0, cur_id - num_frames)
    return list(range(start_id, cur_id + 1))


def build_sample_tag(video_path: str) -> str:
    """Build a run-unique tag to isolate image paths across multiple annotations."""
    stem = Path(video_path).stem.replace(' ', '_')
    safe_stem = ''.join(ch if (ch.isalnum() or ch in ['_', '-']) else '_' for ch in stem)
    timestamp = int(time.time() * 1000)
    return f"{safe_stem}_{timestamp}"


def finalize_and_exit(
    frames: List[Dict],
    data_output_dir: Path,
    task: str,
    completion_reason: str,
    is_done: bool,
    sample_tag: str,
    success_cutoff: float,
):
    data_output_dir = Path(data_output_dir)
    if len(frames) == 0:
        print("\n[Server] Error: No frames left after deletion! Shutting down...")
        os._exit(1)

    print(f"\n[Server] Output path: {data_output_dir}")
    print(f"[Server] Task completion status: {'Done' if is_done else 'Not Done'}")
    if is_done and completion_reason:
        print(f"[Server] Task completion reason: {completion_reason}")

    if not isinstance(success_cutoff, (int, float)):
        success_cutoff = 0.75
    success_cutoff = max(0.0, min(1.0, float(success_cutoff)))
    if not is_done:
        print(f"[Server] Success cutoff: {success_cutoff}")

    from core.utils.common import append_jsonlines
    from evqa.tools.functions.progress import p as p_function
    prg = p_function(frames)

    # Scale progress down if task is ultimately incomplete
    if not is_done:
        prg = [p * success_cutoff for p in prg]

    for frame, p in zip(frames, prg):
        frame['progress'] = float(p)

    following_actions = []

    # Trace planned actions backwards
    for frame in reversed(frames):
        if frame['sub_task'].lower() not in ['done', 'not done'] and frame['sub_task'] not in following_actions:
            following_actions.insert(0, frame['sub_task'])
        frame['planned_actions'] = following_actions.copy()

    task_prefix = task[:15].replace(' ', '_')
    image_root = data_output_dir / "images" / task_prefix / sample_tag
    image_root.mkdir(parents=True, exist_ok=True)
    jl_path = data_output_dir / "qa_pairs.jsonl"

    template = load_templates('procedural_c')[0]
    QA_pairs = []

    no_skip_ids, finished_ids = [], []
    for cur_id in range(len(frames)):
        if frames[cur_id]['progress'] < 1:
            no_skip_ids.append(cur_id)
        else:
            finished_ids.append(cur_id)

    random.shuffle(finished_ids)
    finished_ids = finished_ids[:10]
    no_skip_ids = sorted(no_skip_ids + finished_ids)
    dropped_no_action = 0

    for cur_id in no_skip_ids:
        prg = frames[cur_id]['progress']
        finished = prg >= 1
        repeat_times = 3 if finished else 1

        # For unfinished states, drop samples with no actionable future steps.
        if not finished and not frames[cur_id].get('planned_actions'):
            dropped_no_action += 1
            continue

        for _ in range(repeat_times):
            sampled_fids = sample_previous_frames(cur_id)
            sampled_image_paths = [
                ensure_image_on_disk(data_output_dir, image_root, fid, frames[fid]['exo_image'])
                for fid in sampled_fids
            ]

            prompt = template.format(task=task)
            question = ''.join(['<image>\n' for _ in sampled_fids]) + prompt

            if finished:
                answer = "Image shows the final result."
                if completion_reason:
                    answer += f" Reason: {completion_reason}."
                answer += "\nTherefore, the estimated progress is <progress>100%</progress>."
            else:
                steps_str = '\n'.join([f"{i + 1}. {action}" for i, action in enumerate(frames[cur_id]['planned_actions'])])
                answer = f"The following actions are required: \n{steps_str}\nTherefore, the estimated progress is <progress>{prg * 100:.2f}%</progress>."

            QA_pairs.append({
                "image": sampled_image_paths,
                "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
            })

    if dropped_no_action > 0:
        print(f"[Server] Dropped {dropped_no_action} unfinished samples with no following actions.")

    append_jsonlines(QA_pairs, jl_path)
    print("[Server] Processing complete. Shutting down annotator...")
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Web-based Video Annotator")
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Path to video")
    parser.add_argument("--data_output_dir", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--task", "-t", type=str, required=True, help="Task description")
    parser.add_argument("--port", "-p", type=int, default=5110, help="Port to run the web server on")
    args = parser.parse_args()

    DATA_OUTPUT_DIR, TASK_DESCRIPTION = args.data_output_dir, args.task
    SAMPLE_TAG = build_sample_tag(args.video_path)

    try:
        GLOBAL_FRAMES = decode_video(args.video_path)
    except Exception as e:
        GLOBAL_FRAMES = [Image.new('RGB', (640, 480), color=(i % 255, 100, 100)) for i in range(100)]

    print(f"Decoded {len(GLOBAL_FRAMES)} frames. Access: http://<server-ip>:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
